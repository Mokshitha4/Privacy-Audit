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
        "data": {
            "role": "member_nonmember_split", "path": "/data",
            "schema": {"format": "csv", "member_file": "train.csv", "nonmember_file": "test.csv", "text_field": "note"},
        },
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


# ---------------------------------------------------------------------------
# Form -> job config assembly
# ---------------------------------------------------------------------------

_FORM_DEFAULTS = {
    "model_source": "huggingface", "model_identifier": "gpt2", "model_access": "white_box",
    "model_base_model": "", "model_api_key": "", "model_base_url": "",
    "ft_enabled": False, "ft_regime": "full_ft", "ft_loss": "full", "ft_epsilon": "",
    "ref_source": "huggingface", "ref_identifier": "", "ref_access": "white_box",
    "ref_base_model": "", "ref_api_key": "", "ref_base_url": "",
    "data_role": "member_nonmember_split", "data_path": "/data",
    "schema_format": "csv", "schema_member_file": "train.csv", "schema_nonmember_file": "test.csv",
    "schema_text_field": "note", "schema_text_template": "",
    "em_enabled": True, "em_prefix_len": 50, "em_continuation_len": 500, "em_max_samples": "",
    "em_ngram_ns": "10, 20, 30, 50",
    "mia_enabled": False, "mia_num_members": "", "mia_num_nonmembers": "", "mia_max_length": 512,
    "mia_k_percent": 20, "mia_n_folds": 5, "mia_batch_size": 8, "mia_seed": 42,
    "ezmia_enabled": False, "ezmia_num_members": "", "ezmia_num_nonmembers": "",
    "ezmia_sequence_length": 128, "ezmia_batch_size": 8, "ezmia_seed": 42,
    "return_raw_generations": False,
}


def _config_from_form(**overrides):
    fields = {**_FORM_DEFAULTS, **overrides}
    return ui._form_config_dict(*(fields[name] for name in ui._FIELD_ORDER))


def test_form_defaults_assemble_into_a_valid_config():
    config = _config_from_form()
    ui.validate_job_config(config)  # must not raise
    assert config["model"] == {"source": "huggingface", "identifier": "gpt2", "access": "white_box"}
    assert config["attacks"] == [
        {"family": "EM", "variant": "default", "params": {"prefix_len": 50, "continuation_len": 500, "ngram_ns": [10, 20, 30, 50]}}
    ]
    assert config["data"]["schema"] == {
        "format": "csv", "member_file": "train.csv", "nonmember_file": "test.csv", "text_field": "note"
    }
    assert config["output"] == {"return_raw_generations": False}
    assert "finetuning" not in config["model"]
    assert "reference_model" not in config


def test_blank_optional_numbers_are_omitted_not_sent_as_null():
    config = _config_from_form(mia_enabled=True)
    mia_params = next(a["params"] for a in config["attacks"] if a["family"] == "MIA")
    assert "num_members" not in mia_params and "num_nonmembers" not in mia_params
    assert mia_params == {"max_length": 512, "k_percent": 20, "n_folds": 5, "batch_size": 8, "seed": 42}


def test_num_members_cap_is_forwarded_when_set():
    config = _config_from_form(mia_enabled=True, mia_num_members="200", mia_num_nonmembers="200")
    mia_params = next(a["params"] for a in config["attacks"] if a["family"] == "MIA")
    assert mia_params["num_members"] == 200 and mia_params["num_nonmembers"] == 200


def test_non_numeric_cap_field_is_reported_as_a_field_error():
    with pytest.raises(ValueError, match="number"):
        _config_from_form(mia_enabled=True, mia_num_members="lots")


def test_text_template_takes_precedence_over_text_field():
    config = _config_from_form(schema_text_template="Q: {q}\nA: {a}")
    assert config["data"]["schema"]["text_template"] == "Q: {q}\nA: {a}"
    assert "text_field" not in config["data"]["schema"]


def test_reference_model_block_added_when_ezmia_enabled():
    config = _config_from_form(
        ezmia_enabled=True, ref_source="local_checkpoint", ref_identifier="/ckpt", ref_base_model="Qwen/Qwen3-0.6B",
    )
    assert config["reference_model"] == {
        "source": "local_checkpoint", "identifier": "/ckpt", "access": "white_box", "base_model": "Qwen/Qwen3-0.6B"
    }


def test_reference_model_block_absent_when_ezmia_disabled():
    # MIA doesn't need a reference model; enabling it alone must not add one.
    config = _config_from_form(mia_enabled=True, ref_identifier="gpt2")
    assert "reference_model" not in config


def test_openai_model_block_includes_api_key_and_base_url():
    config = _config_from_form(
        model_source="openai", model_identifier="gpt-4o-mini", model_access="black_box",
        model_api_key="sk-test", model_base_url="https://my-proxy.example.com/v1",
    )
    assert config["model"] == {
        "source": "openai", "identifier": "gpt-4o-mini", "access": "black_box",
        "api_key": "sk-test", "base_url": "https://my-proxy.example.com/v1",
    }


def test_openai_model_block_omits_blank_api_key_and_base_url():
    config = _config_from_form(model_source="openai", model_identifier="gpt-4o-mini", model_access="black_box")
    assert config["model"] == {"source": "openai", "identifier": "gpt-4o-mini", "access": "black_box"}


def test_api_key_and_base_url_ignored_for_non_openai_sources():
    # A user could type into these fields and then switch source away from openai; the
    # assembled config must not leak them into an unrelated source.
    config = _config_from_form(model_api_key="sk-leftover", model_base_url="https://leftover.example.com")
    assert "api_key" not in config["model"] and "base_url" not in config["model"]


def test_reference_openai_block_includes_api_key():
    config = _config_from_form(
        ezmia_enabled=True, ref_source="openai", ref_identifier="gpt-4o-mini", ref_access="black_box",
        ref_api_key="sk-ref",
    )
    assert config["reference_model"]["api_key"] == "sk-ref"


def test_finetuning_block_added_only_when_recorded():
    config = _config_from_form(ft_enabled=True, ft_regime="dp_sgd", ft_loss="masked", ft_epsilon=8)
    assert config["model"]["finetuning"] == {"regime": "dp_sgd", "loss": "masked", "epsilon": 8}


def test_all_three_attacks_can_be_enabled_together():
    config = _config_from_form(mia_enabled=True, ezmia_enabled=True, ref_identifier="gpt2")
    assert [a["family"] for a in config["attacks"]] == ["EM", "MIA", "EZ_MIA"]


def test_disabling_every_attack_yields_an_empty_list_that_fails_validation():
    config = _config_from_form(em_enabled=False)
    assert config["attacks"] == []
    with pytest.raises(ValueError):
        ui.validate_job_config(config)


def test_bad_ngram_ns_is_reported_as_a_field_error():
    with pytest.raises(ValueError, match="ngram_ns"):
        _config_from_form(em_ngram_ns="10, oops, 30")


def test_ngram_ns_accepts_spaces_or_commas_and_blank_omits_it():
    assert _config_from_form(em_ngram_ns="10 20 40")["attacks"][0]["params"]["ngram_ns"] == [10, 20, 40]
    assert "ngram_ns" not in _config_from_form(em_ngram_ns="  ")["attacks"][0]["params"]


def test_validate_form_returns_status_and_the_assembled_json():
    fields = [_FORM_DEFAULTS[name] for name in ui._FIELD_ORDER]
    status, config_text = ui._validate_form(*fields)
    assert status == "Config is valid."
    assert json.loads(config_text)["model"]["identifier"] == "gpt2"


# ---------------------------------------------------------------------------
# Streaming Run: live progress + plain-language summary
# ---------------------------------------------------------------------------

_LLM_FORM_DEFAULTS = {"llm_enabled": False, "llm_provider": "anthropic", "llm_api_key": "", "llm_model": ""}


def _drain(generator):
    """Collect every yielded frame from a streaming UI generator."""
    return list(generator)


def _all_fields(**overrides):
    """Job-config fields (defaults + overrides) followed by LLM fields, in _run_form_streaming's
    expected order."""
    job = {**_FORM_DEFAULTS, **{k: v for k, v in overrides.items() if k in _FORM_DEFAULTS}}
    llm = {**_LLM_FORM_DEFAULTS, **{k: v for k, v in overrides.items() if k in _LLM_FORM_DEFAULTS}}
    return [job[name] for name in ui._FIELD_ORDER] + [llm[name] for name in ui._LLM_FIELD_ORDER]


def test_run_form_streaming_delegates_to_run_job_and_streams_progress(monkeypatch):
    captured = {}

    def _fake_run_job(job_cfg, on_progress=None):
        captured["cfg"] = job_cfg
        if on_progress:
            on_progress("Loading model...")
            on_progress("Running EM...")
        return {"job_id": "x", "metrics": {"EM": {"mem_at_50": 0.0}}}

    monkeypatch.setattr(ui, "run_job", _fake_run_job)
    frames = _drain(ui._run_form_streaming(*_all_fields()))

    assert captured["cfg"]["attacks"][0]["family"] == "EM"

    # Intermediate frames show progress messages arriving, with no result yet.
    status_logs = [f[0] for f in frames]
    assert any("Loading model" in s for s in status_logs)
    assert any("Running EM" in s for s in status_logs)
    assert all(f[2] == "" for f in frames[:-1])  # report box empty until the final frame

    # Final frame has the report, a rendered plain-language summary, and "Done."
    final_status, final_summary, final_report, final_config = frames[-1]
    assert "Done." in final_status
    assert json.loads(final_report) == {"job_id": "x", "metrics": {"EM": {"mem_at_50": 0.0}}}
    assert "Exact Memorization" in final_summary
    assert json.loads(final_config)["attacks"][0]["family"] == "EM"


def test_run_form_streaming_reports_config_errors_without_calling_run_job(monkeypatch):
    called = []
    monkeypatch.setattr(ui, "run_job", lambda job_cfg, on_progress=None: called.append(1))

    frames = _drain(ui._run_form_streaming(*_all_fields(em_ngram_ns="not, a, list")))

    assert len(frames) == 1
    assert "ngram_ns" in frames[0][0]
    assert called == []


def test_run_form_streaming_surfaces_run_job_failures(monkeypatch):
    def _boom(job_cfg, on_progress=None):
        if on_progress:
            on_progress("Loading model...")
        raise RuntimeError("model download failed")

    monkeypatch.setattr(ui, "run_job", _boom)
    frames = _drain(ui._run_form_streaming(*_all_fields()))

    final_status, final_summary, final_report, _ = frames[-1]
    assert "Failed" in final_status
    error = json.loads(final_report)
    assert "model download failed" in error["error"]
    # No explanation to show for a failed run; the initial placeholder stays.
    assert final_summary == ui._INITIAL_SUMMARY


# ---------------------------------------------------------------------------
# Optional LLM narration on top of the rule-based summary
# ---------------------------------------------------------------------------

def _fake_run_job_em_only(job_cfg, on_progress=None):
    return {"job_id": "x", "metrics": {"EM": {"mem_at_50": 0.0}}}


def test_llm_narration_off_by_default_does_not_call_explain_with_llm(monkeypatch):
    monkeypatch.setattr(ui, "run_job", _fake_run_job_em_only)
    called = []
    monkeypatch.setattr(ui, "explain_with_llm", lambda report, config: called.append(1))

    frames = _drain(ui._run_form_streaming(*_all_fields()))

    assert called == []
    assert "AI narration" not in frames[-1][1]


def test_llm_narration_appended_to_summary_when_enabled(monkeypatch):
    monkeypatch.setattr(ui, "run_job", _fake_run_job_em_only)
    captured = {}

    def _fake_explain_with_llm(report, config):
        captured["config"] = config
        return "This model looks fine based on what was tested."

    monkeypatch.setattr(ui, "explain_with_llm", _fake_explain_with_llm)

    frames = _drain(ui._run_form_streaming(*_all_fields(
        llm_enabled=True, llm_provider="openai", llm_api_key="sk-test", llm_model="gpt-4o-mini",
    )))

    final_status, final_summary, _, _ = frames[-1]
    assert "Exact Memorization" in final_summary  # rule-based summary still present
    assert "AI narration (openai)" in final_summary
    assert "This model looks fine" in final_summary
    assert "Done." in final_status
    assert captured["config"].provider == "openai"
    assert captured["config"].api_key == "sk-test"
    assert captured["config"].model == "gpt-4o-mini"


def test_llm_narration_blank_api_key_and_model_become_none(monkeypatch):
    monkeypatch.setattr(ui, "run_job", _fake_run_job_em_only)
    captured = {}
    monkeypatch.setattr(ui, "explain_with_llm", lambda report, config: captured.setdefault("config", config) or "ok")

    _drain(ui._run_form_streaming(*_all_fields(llm_enabled=True)))

    assert captured["config"].api_key is None
    assert captured["config"].model is None


def test_llm_narration_failure_falls_back_to_rule_based_summary(monkeypatch):
    monkeypatch.setattr(ui, "run_job", _fake_run_job_em_only)

    def _boom(report, config):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(ui, "explain_with_llm", _boom)

    frames = _drain(ui._run_form_streaming(*_all_fields(llm_enabled=True, llm_api_key="sk-test")))

    final_status, final_summary, final_report, _ = frames[-1]
    assert "Exact Memorization" in final_summary  # rule-based summary intact
    assert "AI narration" not in final_summary
    assert "AI narration failed" in final_status
    assert "network unreachable" in final_status
    assert json.loads(final_report) == {"job_id": "x", "metrics": {"EM": {"mem_at_50": 0.0}}}
