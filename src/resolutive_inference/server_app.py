"""Composed PC/server application with stateful inference-session endpoints."""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .api import EngineName, PersistentRegistry, _decode_observations, create_app
from .edge_compact import Q4CompactRobust119, StudentTCostLUT
from .incremental_runtime import IncrementalIntegerLag8
from .integer_runtime import IntegerLag8Decoder
from .sessions import MAX_SESSION_OBSERVATIONS, MAX_SESSIONS, SessionManager

API_KEY_HEADER = "X-API-Key"


class SessionCreateRequest(BaseModel):
    model_id: str
    engine: EngineName = "integer_lag8"
    version: int | None = None


class SessionInfo(BaseModel):
    session_id: str
    model_id: str
    version: int
    engine: EngineName
    observation_count: int
    compute_mode: str


class SessionObservationsRequest(BaseModel):
    observations: list[list[float]]


class SessionInferenceResponse(SessionInfo):
    ready: bool
    states: list[int]
    latest_state: int | None


def _build_incremental_runtime(model, engine: EngineName):
    if engine != "integer_lag8":
        return None
    q4 = Q4CompactRobust119.from_model(model)
    lut = StudentTCostLUT.build(128, degrees_of_freedom=model.degrees_of_freedom)
    decoder = IntegerLag8Decoder.compile(q4, lut, lag=8)
    return IncrementalIntegerLag8(decoder)


def create_server_app(
    *,
    model_store_path: str | Path | None = None,
    sessions: SessionManager | None = None,
    api_key: str | None = None,
) -> FastAPI:
    if api_key is not None and not api_key:
        raise ValueError("api_key must not be empty")

    app = create_app(model_store_path=model_store_path)
    manager = sessions or SessionManager()
    app.state.sessions = manager
    app.state.api_key_enabled = api_key is not None

    if api_key is not None:
        expected_key = api_key

        @app.middleware("http")
        async def require_api_key(request: Request, call_next):
            if request.url.path.startswith("/v1/"):
                provided = request.headers.get(API_KEY_HEADER)
                if provided is None or not hmac.compare_digest(provided, expected_key):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "invalid or missing API key"},
                        headers={"WWW-Authenticate": "ApiKey"},
                    )
            return await call_next(request)

    def _session_response(session) -> SessionInferenceResponse:
        states: list[int] = []
        if session.observation_count >= 2:
            if session.runtime is not None:
                states = [int(value) for value in session.runtime.current_states()]
            else:
                registry = app.state.registry
                model = registry.get(session.model_id, session.version)
                decoded = _decode_observations(
                    model,
                    session.observations,
                    session.engine,  # type: ignore[arg-type]
                )
                states = [int(value) for value in decoded]
        return SessionInferenceResponse(
            session_id=session.session_id,
            model_id=session.model_id,
            version=session.version,
            engine=session.engine,  # type: ignore[arg-type]
            observation_count=session.observation_count,
            compute_mode=session.compute_mode,
            ready=bool(states),
            states=states,
            latest_state=states[-1] if states else None,
        )

    @app.post("/v1/sessions", response_model=SessionInfo)
    def create_session(request: SessionCreateRequest) -> SessionInfo:
        registry = app.state.registry
        try:
            selected_version = (
                registry.latest_version(request.model_id)
                if request.version is None
                else int(request.version)
            )
            model = registry.get(request.model_id, selected_version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model or version not found") from exc
        try:
            runtime = _build_incremental_runtime(model, request.engine)
            session = manager.create(
                model_id=request.model_id,
                version=selected_version,
                engine=request.engine,
                runtime=runtime,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return SessionInfo(
            session_id=session.session_id,
            model_id=session.model_id,
            version=session.version,
            engine=session.engine,  # type: ignore[arg-type]
            observation_count=session.observation_count,
            compute_mode=session.compute_mode,
        )

    @app.get("/v1/sessions/{session_id}", response_model=SessionInfo)
    def get_session(session_id: str) -> SessionInfo:
        try:
            session = manager.get(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        return SessionInfo(
            session_id=session.session_id,
            model_id=session.model_id,
            version=session.version,
            engine=session.engine,  # type: ignore[arg-type]
            observation_count=session.observation_count,
            compute_mode=session.compute_mode,
        )

    @app.post("/v1/sessions/{session_id}/observations", response_model=SessionInferenceResponse)
    def append_observations(
        session_id: str,
        request: SessionObservationsRequest,
    ) -> SessionInferenceResponse:
        try:
            session = manager.append(session_id, request.observations)
            return _session_response(session)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.websocket("/v1/ws/sessions/{session_id}")
    async def websocket_session(websocket: WebSocket, session_id: str) -> None:
        if api_key is not None:
            provided = websocket.headers.get(API_KEY_HEADER)
            if provided is None or not hmac.compare_digest(provided, api_key):
                await websocket.close(code=4401, reason="invalid or missing API key")
                return
        try:
            session = manager.get(session_id)
        except KeyError:
            await websocket.close(code=4404, reason="session not found")
            return

        await websocket.accept()
        await websocket.send_json(
            {
                "type": "session",
                "session_id": session.session_id,
                "model_id": session.model_id,
                "version": session.version,
                "engine": session.engine,
                "compute_mode": session.compute_mode,
                "observation_count": session.observation_count,
            }
        )

        try:
            while True:
                payload = await websocket.receive_json()
                if payload.get("type") == "close":
                    await websocket.send_json({"type": "closed", "session_id": session_id})
                    await websocket.close(code=1000)
                    return
                if payload.get("type") != "observations":
                    await websocket.send_json(
                        {"type": "error", "detail": "expected type='observations' or type='close'"}
                    )
                    continue
                observations = payload.get("observations")
                if not isinstance(observations, list):
                    await websocket.send_json({"type": "error", "detail": "observations must be a list"})
                    continue
                try:
                    session = manager.append(session_id, observations)
                    response = _session_response(session)
                except (KeyError, ValueError) as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                    continue
                await websocket.send_json({"type": "inference", **response.model_dump()})
        except WebSocketDisconnect:
            return

    @app.delete("/v1/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, str]:
        try:
            manager.delete(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        return {"status": "deleted", "session_id": session_id}

    @app.get("/v1/server-info")
    def server_info() -> dict[str, object]:
        return {
            "session_mode": "process-memory",
            "integer_lag8_session_compute": "incremental-lag8",
            "other_session_compute": "buffered-redecode",
            "transports": ["http", "websocket"],
            "websocket_path": "/v1/ws/sessions/{session_id}",
            "session_count": manager.count(),
            "authentication": "api-key" if api_key is not None else "disabled",
            "api_key_header": API_KEY_HEADER if api_key is not None else None,
            "limits": {
                "max_sessions": MAX_SESSIONS,
                "max_session_observations": MAX_SESSION_OBSERVATIONS,
            },
            "persistent_models": isinstance(app.state.registry, PersistentRegistry),
        }

    return app


def _default_server_app() -> FastAPI:
    store_path = os.getenv("RESOLUTIVE_MODEL_STORE")
    api_key = os.getenv("RESOLUTIVE_API_KEY")
    return create_server_app(model_store_path=store_path, api_key=api_key)


app = _default_server_app()
