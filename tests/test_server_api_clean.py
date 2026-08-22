from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from resolutive_inference.api import create_app
from resolutive_inference.server_app import create_server_app


def _model_payload(offset: float = 0.0) -> dict[str, object]:
    transition = (np.eye(4) + 0.2).tolist()
    transition2 = np.ones((4, 4, 4), dtype=float)
    return {
        "means": (np.arange(28, dtype=float).reshape(4, 7) / 10.0 + offset).tolist(),
        "shared_variances": np.linspace(0.7, 1.3, 7).tolist(),
        "transition": transition,
        "transition2": transition2.tolist(),
        "initial": [1.0, 1.0, 1.0, 1.0],
        "degrees_of_freedom": 3.0,
    }


def _observations(length: int = 12) -> list[list[float]]:
    return np.tile(np.linspace(0.0, 0.6, 7), (length, 1)).tolist()


def test_rest_register_and_infer() -> None:
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok", "model_count": 0}
    registered = client.put("/v1/models/demo", json=_model_payload())
    assert registered.status_code == 200
    assert registered.json()["statistic_count"] == 119

    for engine in ("float", "q4_lut128", "integer_lag8"):
        response = client.post(
            "/v1/infer/demo",
            json={"observations": _observations(), "engine": engine},
        )
        assert response.status_code == 200
        assert response.json()["observation_count"] == 12
        assert len(response.json()["states"]) == 12


def test_persistent_versions_survive_restart(tmp_path: Path) -> None:
    store = tmp_path / "models"
    first = TestClient(create_app(model_store_path=store))
    assert first.put("/v1/models/demo", json=_model_payload(0.0)).status_code == 200
    second_version = first.put("/v1/models/demo", json=_model_payload(0.5))
    assert second_version.json()["latest_version"] == 2

    restarted = TestClient(create_app(model_store_path=store))
    versions = restarted.get("/v1/models/demo/versions").json()
    assert versions["versions"] == [1, 2]
    old = restarted.post(
        "/v1/infer/demo?version=1",
        json={"observations": _observations(), "engine": "float"},
    )
    assert old.status_code == 200
    assert old.json()["version"] == 1


def test_api_key_protects_v1_routes() -> None:
    client = TestClient(create_server_app(api_key="secret"))
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"X-API-Key": "secret"}).status_code == 200


def test_incremental_session_and_websocket() -> None:
    client = TestClient(create_server_app())
    assert client.put("/v1/models/demo", json=_model_payload()).status_code == 200
    opened = client.post(
        "/v1/sessions",
        json={"model_id": "demo", "engine": "integer_lag8"},
    )
    assert opened.status_code == 200
    session_id = opened.json()["session_id"]
    assert opened.json()["compute_mode"] == "incremental-lag8"

    pushed = client.post(
        f"/v1/sessions/{session_id}/observations",
        json={"observations": _observations(4)},
    )
    assert pushed.status_code == 200
    assert pushed.json()["ready"] is True
    assert pushed.json()["observation_count"] == 4

    with client.websocket_connect(f"/v1/ws/sessions/{session_id}") as websocket:
        hello = websocket.receive_json()
        assert hello["type"] == "session"
        websocket.send_json({"type": "observations", "observations": _observations(2)})
        inference = websocket.receive_json()
        assert inference["type"] == "inference"
        assert inference["observation_count"] == 6
        websocket.send_json({"type": "close"})
        assert websocket.receive_json()["type"] == "closed"
