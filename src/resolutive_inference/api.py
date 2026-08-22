"""FastAPI service for PC/server deployments of Resolutive Inference."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .compact_robust import CompactRobust119
from .edge_compact import Q4CompactRobust119, StudentTCostLUT
from .integer_runtime import IntegerLag8Decoder
from .model_store import JsonModelStore

EngineName = Literal["float", "q4_lut128", "integer_lag8"]
SUPPORTED_ENGINES: list[EngineName] = ["float", "q4_lut128", "integer_lag8"]
MAX_BATCH_SEQUENCES = 64
MAX_SEQUENCE_LENGTH = 4096
MAX_BATCH_OBSERVATIONS = 65536


class ModelPayload(BaseModel):
    means: list[list[float]]
    shared_variances: list[float]
    transition: list[list[float]]
    transition2: list[list[list[float]]]
    initial: list[float]
    degrees_of_freedom: float = Field(default=3.0, gt=0.0)

    def to_model(self) -> CompactRobust119:
        return CompactRobust119(
            means=np.asarray(self.means, dtype=float),
            shared_variances=np.asarray(self.shared_variances, dtype=float),
            transition=np.asarray(self.transition, dtype=float),
            transition2=np.asarray(self.transition2, dtype=float),
            initial=np.asarray(self.initial, dtype=float),
            degrees_of_freedom=float(self.degrees_of_freedom),
        )


class InferenceRequest(BaseModel):
    observations: list[list[float]]
    engine: EngineName = "float"


class BatchInferenceRequest(BaseModel):
    sequences: list[list[list[float]]]
    engine: EngineName = "float"


class InferenceResponse(BaseModel):
    model_id: str
    version: int
    engine: EngineName
    states: list[int]
    observation_count: int


class BatchSequenceResult(BaseModel):
    states: list[int]
    observation_count: int


class BatchInferenceResponse(BaseModel):
    model_id: str
    version: int
    engine: EngineName
    sequence_count: int
    total_observation_count: int
    results: list[BatchSequenceResult]


class ModelInfo(BaseModel):
    model_id: str
    latest_version: int
    versions: list[int]
    statistic_count: int
    supported_engines: list[EngineName]


@dataclass
class ModelRegistry:
    models: dict[str, CompactRobust119] = field(default_factory=dict)
    versions_by_id: dict[str, int] = field(default_factory=dict)

    def put(self, model_id: str, model: CompactRobust119) -> int:
        if not model_id or len(model_id) > 128:
            raise ValueError("model_id must contain 1..128 characters")
        self.models[model_id] = model
        version = self.versions_by_id.get(model_id, 0) + 1
        self.versions_by_id[model_id] = version
        return version

    def get(self, model_id: str, version: int | None = None) -> CompactRobust119:
        if version not in (None, self.versions_by_id.get(model_id)):
            raise KeyError(f"unknown model version: {model_id}@{version}")
        try:
            return self.models[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown model_id: {model_id}") from exc

    def versions(self, model_id: str) -> list[int]:
        version = self.versions_by_id.get(model_id)
        return [] if version is None else [version]

    def latest_version(self, model_id: str) -> int:
        try:
            return self.versions_by_id[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown model_id: {model_id}") from exc

    def list_ids(self) -> list[str]:
        return sorted(self.models)


class PersistentRegistry:
    def __init__(self, root: str | Path) -> None:
        self.store = JsonModelStore(root)

    def put(self, model_id: str, model: CompactRobust119) -> int:
        return self.store.put(model_id, model).version

    def get(self, model_id: str, version: int | None = None) -> CompactRobust119:
        return self.store.get(model_id, version)

    def versions(self, model_id: str) -> list[int]:
        return self.store.versions(model_id)

    def latest_version(self, model_id: str) -> int:
        return self.store.latest_version(model_id)

    def list_ids(self) -> list[str]:
        return self.store.list_models()


def _decode_observations(
    model: CompactRobust119,
    observations_payload: list[list[float]],
    engine: EngineName,
) -> np.ndarray:
    observations = np.asarray(observations_payload, dtype=float)
    if observations.ndim != 2 or observations.shape[1:] != (7,) or observations.shape[0] < 2:
        raise ValueError("observations must have shape (length>=2, 7)")
    if observations.shape[0] > MAX_SEQUENCE_LENGTH:
        raise ValueError(f"sequence length must be <= {MAX_SEQUENCE_LENGTH}")
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations must be finite")

    if engine == "float":
        return model.decode(observations)

    q4 = Q4CompactRobust119.from_model(model)
    lut = StudentTCostLUT.build(128, degrees_of_freedom=model.degrees_of_freedom)
    if engine == "q4_lut128":
        return q4.decode(observations, lut=lut)
    if engine == "integer_lag8":
        return IntegerLag8Decoder.compile(q4, lut, lag=8).decode_from_float(observations)
    raise ValueError(f"unsupported engine: {engine}")


def _decode(model: CompactRobust119, request: InferenceRequest) -> np.ndarray:
    return _decode_observations(model, request.observations, request.engine)


def create_app(
    registry: ModelRegistry | PersistentRegistry | None = None,
    *,
    model_store_path: str | Path | None = None,
) -> FastAPI:
    if registry is not None and model_store_path is not None:
        raise ValueError("provide registry or model_store_path, not both")
    if registry is None:
        registry = PersistentRegistry(model_store_path) if model_store_path else ModelRegistry()

    app = FastAPI(
        title="Resolutive Inference API",
        version="0.1.0-dev",
        description="Experimental PC/server API for compact sequential inference.",
    )
    app.state.registry = registry

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "model_count": len(registry.list_ids())}

    @app.get("/v1/info")
    def info() -> dict[str, object]:
        return {
            "api_version": "0.1.0-dev",
            "maturity": "pre-alpha",
            "engines": SUPPORTED_ENGINES,
            "persistence": "json-versioned" if isinstance(registry, PersistentRegistry) else "process-memory",
            "limits": {
                "max_batch_sequences": MAX_BATCH_SEQUENCES,
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "max_batch_observations": MAX_BATCH_OBSERVATIONS,
            },
        }

    @app.get("/v1/models")
    def list_models() -> list[ModelInfo]:
        result: list[ModelInfo] = []
        for model_id in registry.list_ids():
            model = registry.get(model_id)
            versions = registry.versions(model_id)
            result.append(
                ModelInfo(
                    model_id=model_id,
                    latest_version=registry.latest_version(model_id),
                    versions=versions,
                    statistic_count=model.statistic_count,
                    supported_engines=SUPPORTED_ENGINES,
                )
            )
        return result

    @app.get("/v1/models/{model_id}/versions")
    def model_versions(model_id: str) -> dict[str, object]:
        versions = registry.versions(model_id)
        if not versions:
            raise HTTPException(status_code=404, detail="model not found")
        return {
            "model_id": model_id,
            "latest_version": registry.latest_version(model_id),
            "versions": versions,
        }

    @app.put("/v1/models/{model_id}", response_model=ModelInfo)
    def put_model(model_id: str, payload: ModelPayload) -> ModelInfo:
        try:
            model = payload.to_model()
            version = registry.put(model_id, model)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ModelInfo(
            model_id=model_id,
            latest_version=version,
            versions=registry.versions(model_id),
            statistic_count=model.statistic_count,
            supported_engines=SUPPORTED_ENGINES,
        )

    @app.post("/v1/infer/{model_id}", response_model=InferenceResponse)
    def infer(
        model_id: str,
        request: InferenceRequest,
        version: int | None = Query(default=None, ge=1),
    ) -> InferenceResponse:
        try:
            selected_version = registry.latest_version(model_id) if version is None else version
            model = registry.get(model_id, selected_version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model or version not found") from exc
        try:
            states = _decode(model, request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return InferenceResponse(
            model_id=model_id,
            version=selected_version,
            engine=request.engine,
            states=[int(value) for value in states],
            observation_count=len(request.observations),
        )

    @app.post("/v1/infer-batch/{model_id}", response_model=BatchInferenceResponse)
    def infer_batch(
        model_id: str,
        request: BatchInferenceRequest,
        version: int | None = Query(default=None, ge=1),
    ) -> BatchInferenceResponse:
        if not request.sequences:
            raise HTTPException(status_code=422, detail="batch must contain at least one sequence")
        if len(request.sequences) > MAX_BATCH_SEQUENCES:
            raise HTTPException(
                status_code=422,
                detail=f"batch sequence count must be <= {MAX_BATCH_SEQUENCES}",
            )
        total_observations = sum(len(sequence) for sequence in request.sequences)
        if total_observations > MAX_BATCH_OBSERVATIONS:
            raise HTTPException(
                status_code=422,
                detail=f"batch observation count must be <= {MAX_BATCH_OBSERVATIONS}",
            )
        try:
            selected_version = registry.latest_version(model_id) if version is None else version
            model = registry.get(model_id, selected_version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model or version not found") from exc

        results: list[BatchSequenceResult] = []
        try:
            for sequence in request.sequences:
                states = _decode_observations(model, sequence, request.engine)
                results.append(
                    BatchSequenceResult(
                        states=[int(value) for value in states],
                        observation_count=len(sequence),
                    )
                )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return BatchInferenceResponse(
            model_id=model_id,
            version=selected_version,
            engine=request.engine,
            sequence_count=len(results),
            total_observation_count=total_observations,
            results=results,
        )

    return app


def _default_app() -> FastAPI:
    store_path = os.getenv("RESOLUTIVE_MODEL_STORE")
    return create_app(model_store_path=store_path) if store_path else create_app()


app = _default_app()
