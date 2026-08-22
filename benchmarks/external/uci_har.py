"""External UCI HAR benchmark for CompactRobust119 and HMM controls.

The benchmark deliberately uses only training-split information for feature selection
and parameter estimation. Four activities are retained to match the fixed four-state
CompactRobust119 reference: WALKING, SITTING, STANDING and LAYING.

Dataset: UCI Human Activity Recognition Using Smartphones (dataset 240).
License: CC BY 4.0. Dataset files are downloaded at runtime and are not vendored.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from resolutive_inference.baselines import GaussianHMM, StudentTHMM
from resolutive_inference.compact_robust import CompactRobust119

DATASET_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ACTIVITIES = {1: 0, 4: 1, 5: 2, 6: 3}  # walking, sitting, standing, laying
FEATURE_COUNT = 7


def _download_and_extract(cache_dir: Path) -> Path:
    root = cache_dir / "UCI HAR Dataset"
    if root.exists():
        return root
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "uci_har.zip"
    if not archive.exists():
        urllib.request.urlretrieve(DATASET_URL, archive)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(cache_dir)
    return root


def _load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stem = "train" if split == "train" else "test"
    folder = root / stem
    x = np.loadtxt(folder / f"X_{stem}.txt", dtype=float)
    y = np.loadtxt(folder / f"y_{stem}.txt", dtype=int)
    subject = np.loadtxt(folder / f"subject_{stem}.txt", dtype=int)
    return x, y, subject


def _select_features(x_train: np.ndarray) -> np.ndarray:
    """Select seven highest-variance features using training data only."""
    variances = np.var(x_train, axis=0)
    order = np.argsort(variances)[::-1]
    return np.sort(order[:FEATURE_COUNT])


def _segments(y: np.ndarray, subject: np.ndarray) -> list[np.ndarray]:
    """Return contiguous retained-row index segments without bridging removed classes."""
    retained = np.array([int(label) in ACTIVITIES for label in y], dtype=bool)
    segments: list[np.ndarray] = []
    start: int | None = None
    for index in range(y.size):
        valid = retained[index]
        continues = (
            start is not None
            and valid
            and index > 0
            and retained[index - 1]
            and subject[index] == subject[index - 1]
        )
        if valid and start is None:
            start = index
        elif valid and not continues:
            if start is not None:
                segments.append(np.arange(start, index))
            start = index
        elif not valid and start is not None:
            segments.append(np.arange(start, index))
            start = None
    if start is not None:
        segments.append(np.arange(start, y.size))
    return [segment for segment in segments if segment.size >= 3]


def _map_labels(labels: np.ndarray) -> np.ndarray:
    return np.array([ACTIVITIES[int(value)] for value in labels], dtype=int)


def _estimate_models(
    x: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
) -> tuple[CompactRobust119, GaussianHMM, StudentTHMM]:
    segments = _segments(y, subject)
    retained_idx = np.concatenate(segments)
    retained_x = x[retained_idx]
    retained_y = _map_labels(y[retained_idx])

    means = np.vstack([retained_x[retained_y == state].mean(axis=0) for state in range(4)])
    residuals = retained_x - means[retained_y]
    shared_variances = residuals.var(axis=0) + 1e-6
    per_state_variances = np.vstack(
        [retained_x[retained_y == state].var(axis=0) + 1e-6 for state in range(4)]
    )

    transition = np.full((4, 4), 0.5, dtype=float)
    transition2 = np.full((4, 4, 4), 0.2, dtype=float)
    initial = np.full(4, 0.5, dtype=float)
    for segment in segments:
        labels = _map_labels(y[segment])
        initial[labels[0]] += 1.0
        np.add.at(transition, (labels[:-1], labels[1:]), 1.0)
        if labels.size >= 3:
            np.add.at(transition2, (labels[:-2], labels[1:-1], labels[2:]), 1.0)

    compact = CompactRobust119(means, shared_variances, transition, transition2, initial)
    gaussian = GaussianHMM(transition, means, per_state_variances)
    student = StudentTHMM(transition, means, per_state_variances, degrees_of_freedom=3.0)
    return compact, gaussian, student


def _evaluate_compact(
    model: CompactRobust119,
    x: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
) -> tuple[float, float, int]:
    correct = 0
    total = 0
    started = time.perf_counter_ns()
    for segment in _segments(y, subject):
        truth = _map_labels(y[segment])
        pred = model.decode(x[segment])
        correct += int(np.sum(pred == truth))
        total += int(truth.size)
    elapsed = time.perf_counter_ns() - started
    return correct / total, elapsed / 1_000 / total, total


def _evaluate_hmm(
    model_factory: type[GaussianHMM] | type[StudentTHMM],
    template: GaussianHMM | StudentTHMM,
    x: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
) -> tuple[float, float, int]:
    correct = 0
    total = 0
    started = time.perf_counter_ns()
    for segment in _segments(y, subject):
        kwargs = {
            "transition": template.transition.copy(),
            "means": template.means.copy(),
            "variances": template.variances.copy(),
        }
        if model_factory is StudentTHMM:
            kwargs["degrees_of_freedom"] = template.degrees_of_freedom
        model = model_factory(**kwargs)
        truth = _map_labels(y[segment])
        pred = np.array([np.argmax(model.step(row)) for row in x[segment]], dtype=int)
        correct += int(np.sum(pred == truth))
        total += int(truth.size)
    elapsed = time.perf_counter_ns() - started
    return correct / total, elapsed / 1_000 / total, total


def run(cache_dir: Path) -> dict[str, object]:
    root = _download_and_extract(cache_dir)
    x_train_full, y_train, subject_train = _load_split(root, "train")
    x_test_full, y_test, subject_test = _load_split(root, "test")

    feature_idx = _select_features(x_train_full)
    x_train = x_train_full[:, feature_idx]
    x_test = x_test_full[:, feature_idx]

    compact, gaussian, student = _estimate_models(x_train, y_train, subject_train)
    c_acc, c_us, count = _evaluate_compact(compact, x_test, y_test, subject_test)
    g_acc, g_us, _ = _evaluate_hmm(GaussianHMM, gaussian, x_test, y_test, subject_test)
    t_acc, t_us, _ = _evaluate_hmm(StudentTHMM, student, x_test, y_test, subject_test)

    return {
        "dataset": "UCI Human Activity Recognition Using Smartphones",
        "dataset_id": 240,
        "dataset_url": DATASET_URL,
        "activities": ["WALKING", "SITTING", "STANDING", "LAYING"],
        "selected_feature_indices_zero_based": feature_idx.tolist(),
        "feature_selection": "top-7 training-split variance only",
        "test_observations_evaluated": count,
        "results": {
            "compact_robust119": {
                "accuracy": c_acc,
                "us_per_observation": c_us,
                "stored_statistics": compact.statistic_count,
            },
            "gaussian_hmm": {
                "accuracy": g_acc,
                "us_per_observation": g_us,
                "stored_parameters": gaussian.parameter_count,
            },
            "student_t_hmm": {
                "accuracy": t_acc,
                "us_per_observation": t_us,
                "stored_parameters": student.parameter_count,
            },
        },
        "methodology": (
            "official train/test split; feature selection and parameter estimation use train only; "
            "evaluation preserves contiguous within-subject segments and never bridges excluded classes"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.cache_dir is None:
        with tempfile.TemporaryDirectory(prefix="resolutive-uci-har-") as directory:
            result = run(Path(directory))
    else:
        result = run(args.cache_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
