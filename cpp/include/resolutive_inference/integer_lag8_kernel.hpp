#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

namespace resolutive_inference {

struct IntegerLag8Model {
    int16_t means_q[4][7];
    uint16_t inv_variances_q[7];
    uint16_t lut_values[128];
    int16_t initial_costs[4];
    int16_t transition1_costs[4][4];
    int16_t transition2_costs[4][4][4];
    int32_t observation_scale;
    int32_t inverse_variance_scale;
    int32_t max_distance_q;
    int32_t cost_scale;

    static constexpr std::size_t persistent_table_bytes() noexcept { return 510; }
};

class IntegerLag8Kernel {
public:
    static constexpr std::size_t kStates = 4;
    static constexpr std::size_t kFeatures = 7;
    static constexpr std::size_t kLag = 8;
    static constexpr std::size_t runtime_buffer_bytes() noexcept { return 272; }
    static constexpr std::size_t core_bytes() noexcept {
        return IntegerLag8Model::persistent_table_bytes() + runtime_buffer_bytes();
    }

    bool decode(const IntegerLag8Model& model, const int16_t* observations_q,
                std::size_t length, uint8_t* out_states) noexcept {
        if (!observations_q || !out_states || length < 2 ||
            model.inverse_variance_scale <= 0 || model.max_distance_q <= 0) {
            return false;
        }

        for (std::size_t i = 0; i < length; ++i) out_states[i] = 0xFF;
        int32_t e0[4], e1[4];
        emission_costs(model, observations_q, e0);
        emission_costs(model, observations_q + 7, e1);

        for (std::size_t a = 0; a < 4; ++a) {
            for (std::size_t b = 0; b < 4; ++b) {
                dp_[a * 4 + b] = e0[a] + e1[b] + model.initial_costs[a] +
                                  model.transition1_costs[a][b];
            }
        }
        normalize(dp_);
        ring_count_ = 0;
        ring_head_ = 0;

        for (std::size_t t = 2; t < length; ++t) {
            emission_costs(model, observations_q + t * 7, emission_);
            int64_t global_min = std::numeric_limits<int64_t>::max();

            for (std::size_t b = 0; b < 4; ++b) {
                for (std::size_t c = 0; c < 4; ++c) {
                    int64_t best = std::numeric_limits<int64_t>::max();
                    uint8_t best_a = 0;
                    for (std::size_t a = 0; a < 4; ++a) {
                        const int64_t value = static_cast<int64_t>(dp_[a * 4 + b]) +
                                              model.transition2_costs[a][b][c] +
                                              emission_[c];
                        if (value < best) {
                            best = value;
                            best_a = static_cast<uint8_t>(a);
                        }
                    }
                    next_[b * 4 + c] = static_cast<int32_t>(best);
                    ring_[ring_head_][b * 4 + c] = best_a;
                    if (best < global_min) global_min = best;
                }
            }

            for (std::size_t i = 0; i < 16; ++i) {
                next_[i] =
                    static_cast<int32_t>(static_cast<int64_t>(next_[i]) - global_min);
                dp_[i] = next_[i];
            }
            ring_head_ = static_cast<uint8_t>((ring_head_ + 1) % 8);
            if (ring_count_ < 8) ++ring_count_;

            if (t >= kLag + 1) {
                uint8_t b, c;
                best_pair(dp_, b, c);
                trace_ring(b, c);
                const std::size_t target = t - kLag;
                if (target == 1) out_states[0] = b;
                out_states[target] = c;
            }
        }

        uint8_t b, c;
        best_pair(dp_, b, c);
        out_states[length - 1] = c;
        out_states[length - 2] = b;
        std::size_t write = length >= 3 ? length - 3 : 0;
        for (std::size_t r = 0; r < ring_count_ && length >= 3; ++r) {
            const std::size_t idx = ring_index_from_newest(r);
            const uint8_t a = ring_[idx][b * 4 + c];
            if (write < length && out_states[write] == 0xFF) out_states[write] = a;
            if (write == 0) break;
            --write;
            c = b;
            b = a;
        }

        const uint8_t fallback = static_cast<uint8_t>(best_pair_flat(dp_) % 4);
        for (std::size_t i = 0; i < length; ++i) {
            if (out_states[i] == 0xFF) out_states[i] = fallback;
        }
        return true;
    }

private:
    int32_t dp_[16]{};
    int32_t next_[16]{};
    uint8_t ring_[8][16]{};
    int32_t emission_[4]{};
    uint8_t ring_count_{0};
    uint8_t ring_head_{0};

    static void normalize(int32_t* scores) noexcept {
        int32_t minimum = scores[0];
        for (std::size_t i = 1; i < 16; ++i) {
            if (scores[i] < minimum) minimum = scores[i];
        }
        for (std::size_t i = 0; i < 16; ++i) scores[i] -= minimum;
    }

    void emission_costs(const IntegerLag8Model& model, const int16_t* x,
                        int32_t out[4]) const noexcept {
        for (std::size_t state = 0; state < 4; ++state) {
            int64_t distance_q = 0;
            for (std::size_t feature = 0; feature < 7; ++feature) {
                const int32_t diff = static_cast<int32_t>(x[feature]) -
                                     model.means_q[state][feature];
                const int64_t squared = static_cast<int64_t>(diff) * diff;
                distance_q +=
                    (squared * model.inv_variances_q[feature]) /
                    model.inverse_variance_scale;
            }
            const int64_t numerator = distance_q * 127 + model.max_distance_q / 2;
            int64_t index = numerator / model.max_distance_q;
            if (index < 0) index = 0;
            if (index > 127) index = 127;
            out[state] = model.lut_values[index];
        }
    }

    std::size_t ring_index_from_newest(std::size_t offset) const noexcept {
        return (static_cast<std::size_t>(ring_head_) + 8 - 1 - offset) % 8;
    }

    void trace_ring(uint8_t& b, uint8_t& c) const noexcept {
        for (std::size_t r = 0; r < ring_count_; ++r) {
            const std::size_t idx = ring_index_from_newest(r);
            const uint8_t a = ring_[idx][b * 4 + c];
            c = b;
            b = a;
        }
    }

    static std::size_t best_pair_flat(const int32_t* scores) noexcept {
        std::size_t best = 0;
        for (std::size_t i = 1; i < 16; ++i) {
            if (scores[i] < scores[best]) best = i;
        }
        return best;
    }

    static void best_pair(const int32_t* scores, uint8_t& b, uint8_t& c) noexcept {
        const std::size_t best = best_pair_flat(scores);
        b = static_cast<uint8_t>(best / 4);
        c = static_cast<uint8_t>(best % 4);
    }
};

}  // namespace resolutive_inference
