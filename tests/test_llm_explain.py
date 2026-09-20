"""Tests for the optional LLM narration layer. No real network calls: urllib.request.urlopen
is monkeypatched with a fake response, so these run offline and don't need an API key."""
from __future__ import annotations

import json

import pytest

from privaudit import llm_explain
from privaudit.llm_explain import LLMExplainConfig, explain_with_llm


class _FakeResponse:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _NoRawGenerationsAccess(dict):
    """Raises if anything ever reads report["raw_generations"] -- the LLM call must not need it."""

    def get(self, key, default=None):
        if key == "raw_generations":
            raise AssertionError("explain_with_llm() must not read raw_generations")
        return super().get(key, default)

    def __getitem__(self, key):
        if key == "raw_generations":
            raise AssertionError("explain_with_llm() must not read raw_generations")
        return super().__getitem__(key)


def _report(metrics: dict, raw_generations=None) -> dict:
    data = {"metrics": metrics}
    if raw_generations is not None:
        data["raw_generations"] = raw_generations
    return _NoRawGenerationsAccess(data)


def test_explain_with_llm_requires_a_known_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        explain_with_llm(_report({"EM": {"mem_at_50": 0.0}}), LLMExplainConfig(provider="cohere"))


def test_explain_with_llm_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        explain_with_llm(_report({"EM": {"mem_at_50": 0.0}}), LLMExplainConfig(provider="anthropic"))


def test_explain_with_llm_falls_back_to_env_var_api_key(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        return _FakeResponse({"content": [{"type": "text", "text": "ok"}]})

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    monkeypatch.setattr(llm_explain.urllib.request, "urlopen", fake_urlopen)

    explain_with_llm(_report({"EM": {"mem_at_50": 0.0}}), LLMExplainConfig(provider="anthropic"))
    assert captured["headers"]["x-api-key"] == "sk-ant-from-env"


def test_call_anthropic_sends_expected_request_and_parses_response(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["body"] = json.loads(req.data)
        return _FakeResponse({"content": [{"type": "text", "text": "Hello there."}]})

    monkeypatch.setattr(llm_explain.urllib.request, "urlopen", fake_urlopen)

    text = explain_with_llm(
        _report({"EM": {"mem_at_50": 0.2}}),
        LLMExplainConfig(provider="anthropic", api_key="sk-ant-test", model="claude-sonnet-5"),
    )

    assert text == "Hello there."
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["body"]["model"] == "claude-sonnet-5"
    assert "system" in captured["body"]
    # the payload sent is the rule-based summary, not the raw report
    sent_summary = json.loads(captured["body"]["messages"][0]["content"])
    assert sent_summary["families"][0]["family"] == "EM"


def test_call_openai_sends_expected_request_and_parses_response(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["body"] = json.loads(req.data)
        return _FakeResponse({"choices": [{"message": {"content": "Hi!"}}]})

    monkeypatch.setattr(llm_explain.urllib.request, "urlopen", fake_urlopen)

    text = explain_with_llm(
        _report({"MIA": {"tpr_at_5pct_fpr": 0.05, "roc_auc": 0.5}}),
        LLMExplainConfig(provider="openai", api_key="sk-test"),
    )

    assert text == "Hi!"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer sk-test"
    assert captured["body"]["model"] == "gpt-4o-mini"  # default when config.model is unset


def test_base_url_override_is_used_for_openai_compatible_endpoints(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(llm_explain.urllib.request, "urlopen", fake_urlopen)

    explain_with_llm(
        _report({"EM": {"mem_at_50": 0.0}}),
        LLMExplainConfig(provider="openai", api_key="sk-test", base_url="https://my-proxy.example.com/v1"),
    )
    assert captured["url"] == "https://my-proxy.example.com/v1/chat/completions"


@pytest.mark.parametrize(
    ("provider", "env_var", "default_base_url", "default_model"),
    [
        ("huggingface", "HF_TOKEN", "https://router.huggingface.co/v1", "meta-llama/Meta-Llama-3-8B-Instruct"),
        ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini"),
    ],
)
def test_huggingface_and_openrouter_use_the_shared_openai_compatible_caller(
    monkeypatch, provider, env_var, default_base_url, default_model
):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["body"] = json.loads(req.data)
        return _FakeResponse({"choices": [{"message": {"content": "Looks fine."}}]})

    monkeypatch.setattr(llm_explain.urllib.request, "urlopen", fake_urlopen)

    text = explain_with_llm(
        _report({"EM": {"mem_at_50": 0.0}}),
        LLMExplainConfig(provider=provider, api_key="test-key"),
    )

    assert text == "Looks fine."
    assert captured["url"] == f"{default_base_url}/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == default_model


def test_huggingface_falls_back_to_hf_token_env_var(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["headers"] = dict(req.header_items())
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    monkeypatch.setattr(llm_explain.urllib.request, "urlopen", fake_urlopen)

    explain_with_llm(_report({"EM": {"mem_at_50": 0.0}}), LLMExplainConfig(provider="huggingface"))
    assert captured["headers"]["Authorization"] == "Bearer hf_from_env"


def test_openrouter_requires_openrouter_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        explain_with_llm(_report({"EM": {"mem_at_50": 0.0}}), LLMExplainConfig(provider="openrouter"))


def test_huggingface_and_openrouter_respect_a_custom_model_and_base_url(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data)
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(llm_explain.urllib.request, "urlopen", fake_urlopen)

    explain_with_llm(
        _report({"EM": {"mem_at_50": 0.0}}),
        LLMExplainConfig(
            provider="openrouter", api_key="test-key", model="meta-llama/llama-3.1-8b-instruct",
            base_url="https://my-openrouter-proxy.example.com/v1",
        ),
    )
    assert captured["url"] == "https://my-openrouter-proxy.example.com/v1/chat/completions"
    assert captured["body"]["model"] == "meta-llama/llama-3.1-8b-instruct"


def test_explain_with_llm_never_touches_raw_generations(monkeypatch):
    monkeypatch.setattr(
        llm_explain.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResponse({"content": [{"type": "text", "text": "ok"}]}),
    )
    report = _report(
        {"EM": {"mem_at_50": 0.3}},
        raw_generations={"EM": [{"prefix": "p", "groundtruth": "g", "generation": "o"}]},
    )
    explain_with_llm(report, LLMExplainConfig(provider="anthropic", api_key="sk-test"))  # must not raise


def test_malformed_provider_response_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr(llm_explain.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse({"unexpected": "shape"}))
    with pytest.raises(RuntimeError, match="Unexpected Anthropic API response"):
        explain_with_llm(_report({"EM": {"mem_at_50": 0.0}}), LLMExplainConfig(provider="anthropic", api_key="sk-test"))
