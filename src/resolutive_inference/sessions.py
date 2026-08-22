"""In-memory session state for the PC/server API."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any
from uuid import uuid4

import numpy as np

MAX_SESSIONS = 1024
MAX_SESSION_OBSERVATIONS = 4096


@dataclass
class InferenceSession:
    session_id: str
    model_id: str
    version: int
    engine: str
    observations: list[list[float]] = field(default_factory=list)
    runtime: Any | None = None
    processed_observations: int = 0

    @property
    def observation_count(self) -> int:
        if self.runtime is not None:
            return self.processed_observations
        return len(self.observations)

    @property
    def compute_mode(self) -> str:
        if self.runtime is not None:
            return str(getattr(self.runtime, "compute_mode", "incremental"))
        return "buffered-redecode"


class SessionManager:
    """Thread-safe, process-local inference-session registry."""

    def __init__(self, *, max_sessions: int = MAX_SESSIONS) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")
        self.max_sessions = int(max_sessions)
        self._sessions: dict[str, InferenceSession] = {}
        self._lock = RLock()

    def create(
        self,
        *,
        model_id: str,
        version: int,
        engine: str,
        runtime: Any | None = None,
    ) -> InferenceSession:
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise RuntimeError("session limit reached")
            session_id = uuid4().hex
            session = InferenceSession(
                session_id=session_id,
                model_id=model_id,
                version=int(version),
                engine=engine,
                runtime=runtime,
            )
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> InferenceSession:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"unknown session_id: {session_id}") from exc

    def append(self, session_id: str, observations: list[list[float]]) -> InferenceSession:
        if not observations:
            raise ValueError("observations must not be empty")
        values = np.asarray(observations, dtype=float)
        if values.ndim != 2 or values.shape[1:] != (7,):
            raise ValueError("observations must have shape (length, 7)")
        if not np.all(np.isfinite(values)):
            raise ValueError("observations must be finite")

        with self._lock:
            session = self.get(session_id)
            if session.observation_count + len(observations) > MAX_SESSION_OBSERVATIONS:
                raise ValueError(
                    f"session observation count must be <= {MAX_SESSION_OBSERVATIONS}"
                )
            if session.runtime is not None:
                for row in values:
                    session.runtime.append_float(row)
                    session.processed_observations += 1
            else:
                session.observations.extend(values.tolist())
            return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"unknown session_id: {session_id}")
            del self._sessions[session_id]

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)
