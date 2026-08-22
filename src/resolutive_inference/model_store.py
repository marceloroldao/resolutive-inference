"""Versioned on-disk model storage for the PC/server API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .compact_robust import CompactRobust119


@dataclass(frozen=True)
class StoredModelRef:
    model_id: str
    version: int
    path: Path


class JsonModelStore:
    """Simple deterministic JSON store for named model versions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_model_id(model_id: str) -> str:
        if not model_id or len(model_id) > 128:
            raise ValueError("model_id must contain 1..128 characters")
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        if any(ch not in allowed for ch in model_id):
            raise ValueError("model_id may contain only letters, digits, '-', '_' and '.'")
        return model_id

    def _model_dir(self, model_id: str) -> Path:
        return self.root / self._validate_model_id(model_id)

    def versions(self, model_id: str) -> list[int]:
        directory = self._model_dir(model_id)
        if not directory.exists():
            return []
        versions: list[int] = []
        for path in directory.glob("*.json"):
            if path.name == "latest.json":
                continue
            try:
                versions.append(int(path.stem))
            except ValueError:
                continue
        return sorted(versions)

    def latest_version(self, model_id: str) -> int:
        versions = self.versions(model_id)
        if not versions:
            raise KeyError(f"unknown model_id: {model_id}")
        return versions[-1]

    def list_models(self) -> list[str]:
        result: list[str] = []
        if not self.root.exists():
            return result
        for path in self.root.iterdir():
            if path.is_dir() and self.versions(path.name):
                result.append(path.name)
        return sorted(result)

    @staticmethod
    def _serialize(model: CompactRobust119) -> dict[str, object]:
        return {
            "schema": "resolutive-inference.compact-robust119.v1",
            "means": np.asarray(model.means, dtype=float).tolist(),
            "shared_variances": np.asarray(model.shared_variances, dtype=float).tolist(),
            "transition": np.asarray(model.transition, dtype=float).tolist(),
            "transition2": np.asarray(model.transition2, dtype=float).tolist(),
            "initial": np.asarray(model.initial, dtype=float).tolist(),
            "degrees_of_freedom": float(model.degrees_of_freedom),
        }

    @staticmethod
    def _deserialize(payload: dict[str, object]) -> CompactRobust119:
        if payload.get("schema") != "resolutive-inference.compact-robust119.v1":
            raise ValueError("unsupported model schema")
        return CompactRobust119(
            means=np.asarray(payload["means"], dtype=float),
            shared_variances=np.asarray(payload["shared_variances"], dtype=float),
            transition=np.asarray(payload["transition"], dtype=float),
            transition2=np.asarray(payload["transition2"], dtype=float),
            initial=np.asarray(payload["initial"], dtype=float),
            degrees_of_freedom=float(payload["degrees_of_freedom"]),
        )

    def put(self, model_id: str, model: CompactRobust119) -> StoredModelRef:
        directory = self._model_dir(model_id)
        directory.mkdir(parents=True, exist_ok=True)
        current = self.versions(model_id)
        version = (current[-1] + 1) if current else 1
        target = directory / f"{version}.json"
        temp = directory / f".{version}.tmp"
        temp.write_text(
            json.dumps(self._serialize(model), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temp.replace(target)
        (directory / "latest.json").write_text(
            json.dumps({"version": version}, sort_keys=True), encoding="utf-8"
        )
        return StoredModelRef(model_id=model_id, version=version, path=target)

    def get(self, model_id: str, version: int | None = None) -> CompactRobust119:
        selected = self.latest_version(model_id) if version is None else int(version)
        if selected < 1:
            raise KeyError(f"unknown model version: {model_id}@{selected}")
        path = self._model_dir(model_id) / f"{selected}.json"
        if not path.exists():
            raise KeyError(f"unknown model version: {model_id}@{selected}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid persisted model payload")
        return self._deserialize(payload)

    def delete_version(self, model_id: str, version: int) -> None:
        path = self._model_dir(model_id) / f"{int(version)}.json"
        if not path.exists():
            raise KeyError(f"unknown model version: {model_id}@{version}")
        path.unlink()
        versions = self.versions(model_id)
        latest_path = self._model_dir(model_id) / "latest.json"
        if versions:
            latest_path.write_text(
                json.dumps({"version": versions[-1]}, sort_keys=True), encoding="utf-8"
            )
        elif latest_path.exists():
            latest_path.unlink()
