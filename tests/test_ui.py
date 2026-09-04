"""Tests for the optional Gradio UI's request/response logic.

`_validate`/`_run` are plain functions wrapping report.validate_job_config()/runner.run_job(),
so they're tested directly without needing a running Gradio server. `build_app()` needs the
`ui` extra installed; skipped automatically when gradio isn't available.
"""
from __future__ import annotations

import json

import pytest

from privaudit import ui


def _valid_config_text() -> str:
    return json.dumps({
        "model": {"source": "huggingface", "identifier": "gpt2", "access": "white_box"},
        "data": {"role": "member_nonmember_split", "known_dataset": "MedQA", "path": "/data"},
        "attacks": [{"family": "EM", "variant": "default"}],
    })


def test_validate_reports_malformed_json():
    result = ui._validate("{not valid json")
    assert result.startswith("Invalid JSON:")


def test_validate_reports_schema_errors():
    result = ui._validate(json.dumps({"model": {}}))
    assert "required property" in result


def test_validate_reports_success():
    assert ui._validate(_valid_config_text()) == "Config is valid."


def test_run_reports_malformed_json_as_json_error():
    result = json.loads(ui._run("{not valid json"))
    assert "Invalid JSON" in result["error"]


def test_run_reports_schema_errors_as_json_error():
    result = json.loads(ui._run(json.dumps({"model": {}})))
    assert "error" in result


def test_run_delegates_to_run_job_and_returns_its_report(monkeypatch):
    captured = {}

    def _fake_run_job(job_cfg):
        captured["job_cfg"] = job_cfg
        return {"job_id": "abc", "metrics": {}}

    monkeypatch.setattr(ui, "run_job", _fake_run_job)
    result = json.loads(ui._run(_valid_config_text()))

    assert result == {"job_id": "abc", "metrics": {}}
    assert captured["job_cfg"]["model"]["identifier"] == "gpt2"


def test_run_wraps_unexpected_exceptions_from_run_job(monkeypatch):
    def _boom(job_cfg):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(ui, "run_job", _boom)
    result = json.loads(ui._run(_valid_config_text()))

    assert "RuntimeError" in result["error"]
    assert "model download failed" in result["error"]
    assert "traceback" in result


def test_build_app_requires_gradio_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "gradio":
            raise ImportError("No module named 'gradio'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    with pytest.raises(ImportError, match="pip install -e '.\\[ui\\]'"):
        ui.build_app()


def test_build_app_returns_a_gradio_blocks_instance():
    gradio = pytest.importorskip("gradio")
    app = ui.build_app()
    assert isinstance(app, gradio.Blocks)
