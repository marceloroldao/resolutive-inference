#include "resolutive_inference/integer_lag8_kernel.hpp"

#include <cstdint>
#include <cstring>

using resolutive_inference::IntegerLag8Kernel;
using resolutive_inference::IntegerLag8Model;

int main() {
    IntegerLag8Model model{};

    const int16_t means[28] = {
        0, 6, 6, 12, 12, 17, 17, 23, 23, 29, 35, 35, 40, 40,
        46, 46, 52, 52, 58, 63, 63, 69, 69, 75, 75, 81, 81, 86,
    };
    const uint16_t inverse_variances[7] = {183, 164, 142, 125, 116, 105, 98};
    const uint16_t lut[128] = {
        0,146,246,322,383,434,479,518,552,584,612,638,663,685,706,726,
        744,762,779,794,809,824,838,851,864,876,887,899,910,920,931,941,
        950,960,969,977,986,994,1003,1011,1018,1026,1033,1040,1048,1054,
        1061,1068,1074,1081,1087,1093,1099,1105,1111,1117,1122,1128,1133,
        1138,1144,1149,1154,1159,1164,1168,1173,1178,1183,1187,1192,1196,
        1200,1205,1209,1213,1217,1221,1225,1229,1233,1237,1241,1245,1249,
        1252,1256,1260,1263,1267,1270,1274,1277,1281,1284,1287,1291,1294,
        1297,1300,1303,1307,1310,1313,1316,1319,1322,1325,1328,1331,1333,
        1336,1339,1342,1345,1347,1350,1353,1356,1358,1361,1363,1366,1369,
        1371,1374,1376,1379,
    };
    const int16_t initial[4] = {53, 53, 53, 53};
    const int16_t transition1[16] = {
        16,87,87,87, 87,16,87,87, 87,87,16,87, 87,87,87,16,
    };
    int16_t transition2[64];
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            for (int c = 0; c < 4; ++c) {
                transition2[(a * 4 + b) * 4 + c] = (a == c) ? 32 : 57;
            }
        }
    }

    std::memcpy(model.means_q, means, sizeof(means));
    std::memcpy(model.inv_variances_q, inverse_variances, sizeof(inverse_variances));
    std::memcpy(model.lut_values, lut, sizeof(lut));
    std::memcpy(model.initial_costs, initial, sizeof(initial));
    std::memcpy(model.transition1_costs, transition1, sizeof(transition1));
    std::memcpy(model.transition2_costs, transition2, sizeof(transition2));
    model.observation_scale = 32;
    model.inverse_variance_scale = 128;
    model.max_distance_q = 225280;
    model.cost_scale = 64;

    const int16_t observations[24][7] = {
        {-5,1,13,11,17,19,16},{3,2,5,10,5,22,16},{5,4,14,6,11,18,9},
        {4,11,12,13,12,22,24},{2,3,5,4,19,6,17},{23,30,37,37,34,42,36},
        {29,25,26,29,35,37,53},{19,26,29,32,35,36,39},{18,18,30,37,33,36,40},
        {20,19,35,34,32,40,39},{21,27,33,39,43,37,39},{45,49,48,51,62,69,63},
        {42,47,46,53,60,60,70},{46,50,50,61,52,59,58},{49,52,49,62,56,54,63},
        {47,47,62,59,66,67,56},{65,57,70,82,76,80,88},{71,69,72,81,81,82,88},
        {68,72,83,75,77,90,83},{75,69,74,80,86,88,85},{65,72,74,74,82,88,87},
        {75,72,73,77,77,78,89},{63,69,73,79,82,83,77},{66,67,73,82,84,84,89},
    };
    const uint8_t expected[24] = {
        0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,3,3,3,
    };
    uint8_t output[24]{};

    IntegerLag8Kernel kernel;
    if (!kernel.decode(model, &observations[0][0], 24, output)) return 2;
    for (int i = 0; i < 24; ++i) {
        if (output[i] != expected[i]) return 3;
    }

    if (IntegerLag8Model::persistent_table_bytes() != 510) return 4;
    if (IntegerLag8Kernel::runtime_buffer_bytes() != 272) return 5;
    if (IntegerLag8Kernel::core_bytes() != 782) return 6;
    if (sizeof(IntegerLag8Kernel) > 288) return 7;
    return 0;
}
