"""Tests for the local web UI's FastAPI backend.

`/api/validate` and `/ws/run` are thin wrappers around report.validate_job_config() /
runner.run_job() -- the same functions the CLI uses -- so the actual attack logic is mocked
out here and these tests focus on the request/response contract.
"""
from __future__ import annotations

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from privaudit.webapp import server  # noqa: E402


@pytest.fixture()
def client():
    return fastapi_testclient.TestClient(server.app)


def _valid_config() -> dict:
    return {
        "model": {"source": "huggingface", "identifier": "gpt2", "access": "white_box"},
        "data": {
            "role": "member_nonmember_split", "path": "/data",
            "schema": {"format": "csv", "member_file": "train.csv", "nonmember_file": "test.csv", "text_field": "note"},
        },
        "attacks": [{"family": "EM", "variant": "default"}],
    }


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Privacy Audit" in response.text


def test_api_validate_accepts_a_valid_config(client):
    response = client.post("/api/validate", json={"config": _valid_config()})
    assert response.json() == {"valid": True, "error": None}


def test_api_validate_reports_schema_errors(client):
    response = client.post("/api/validate", json={"config": {"model": {}}})
    body = response.json()
    assert body["valid"] is False
    assert "required property" in body["error"]


def test_ws_run_streams_progress_then_a_result(client, monkeypatch):
    def _fake_run_job(job_cfg, on_progress=None):
        assert job_cfg["model"]["identifier"] == "gpt2"
        if on_progress:
            on_progress("Loading model...")
            on_progress("Running EM...")
        return {"job_id": "x", "model_id": "gpt2", "metrics": {"EM": {"mem_at_50": 0.0}}, "canonical_scores": {"EM": 0.0}}

    monkeypatch.setattr(server, "run_job", _fake_run_job)

    with client.websocket_connect("/ws/run") as ws:
        ws.send_json({"config": _valid_config(), "llm": {"enabled": False}})

        messages = []
        while True:
            msg = ws.receive_json()
            messages.append(msg)
            if msg["type"] in ("result", "error"):
                break

    progress_messages = [m["message"] for m in messages if m["type"] == "progress"]
    assert "Loading model..." in progress_messages
    assert "Running EM..." in progress_messages

    result = messages[-1]
    assert result["type"] == "result"
    assert result["report"]["metrics"] == {"EM": {"mem_at_50": 0.0}}
    assert result["summary"]["families"][0]["family"] == "EM"
    assert result["narration"] is None
    assert result["narration_error"] is None


def test_ws_run_reports_config_errors_without_calling_run_job(client, monkeypatch):
    called = []
    monkeypatch.setattr(server, "run_job", lambda job_cfg, on_progress=None: called.append(1))

    with client.websocket_connect("/ws/run") as ws:
        ws.send_json({"config": {"model": {}}, "llm": {"enabled": False}})
        msg = ws.receive_json()

    assert msg["type"] == "error"
    assert "Invalid job config" in msg["message"]
    assert called == []


def test_ws_run_surfaces_run_job_failures(client, monkeypatch):
    def _boom(job_cfg, on_progress=None):
        if on_progress:
            on_progress("Loading model...")
        raise RuntimeError("model download failed")

    monkeypatch.setattr(server, "run_job", _boom)

    with client.websocket_connect("/ws/run") as ws:
        ws.send_json({"config": _valid_config(), "llm": {"enabled": False}})
        messages = []
        while True:
            msg = ws.receive_json()
            messages.append(msg)
            if msg["type"] in ("result", "error"):
                break

    final = messages[-1]
    assert final["type"] == "error"
    assert "model download failed" in final["message"]
    assert "traceback" in final


def test_ws_run_appends_llm_narration_when_enabled(client, monkeypatch):
    def _fake_run_job(job_cfg, on_progress=None):
        return {"job_id": "x", "model_id": "gpt2", "metrics": {"EM": {"mem_at_50": 0.0}}, "canonical_scores": {"EM": 0.0}}

    monkeypatch.setattr(server, "run_job", _fake_run_job)

    captured = {}

    def _fake_explain_with_llm(report, config):
        captured["config"] = config
        return "This model looks fine based on what was tested."

    monkeypatch.setattr(server, "explain_with_llm", _fake_explain_with_llm)

    with client.websocket_connect("/ws/run") as ws:
        ws.send_json({
            "config": _valid_config(),
            "llm": {"enabled": True, "provider": "openai", "api_key": "sk-test", "model": "gpt-4o-mini", "base_url": None},
        })
        messages = []
        while True:
            msg = ws.receive_json()
            messages.append(msg)
            if msg["type"] in ("result", "error"):
                break

    result = messages[-1]
    assert result["type"] == "result"
    assert result["narration"] == "This model looks fine based on what was tested."
    assert captured["config"].provider == "openai"
    assert captured["config"].api_key == "sk-test"


def test_ws_run_llm_narration_failure_is_reported_without_failing_the_run(client, monkeypatch):
    monkeypatch.setattr(server, "run_job", lambda job_cfg, on_progress=None: {
        "job_id": "x", "model_id": "gpt2", "metrics": {"EM": {"mem_at_50": 0.0}}, "canonical_scores": {"EM": 0.0},
    })

    def _boom(report, config):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(server, "explain_with_llm", _boom)

    with client.websocket_connect("/ws/run") as ws:
        ws.send_json({"config": _valid_config(), "llm": {"enabled": True, "provider": "anthropic"}})
        messages = []
        while True:
            msg = ws.receive_json()
            messages.append(msg)
            if msg["type"] in ("result", "error"):
                break

    result = messages[-1]
    assert result["type"] == "result"
    assert result["narration"] is None
    assert "network unreachable" in result["narration_error"]
