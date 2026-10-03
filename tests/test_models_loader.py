"""LoRA/PEFT adapter detection and base-model resolution in load_model().

Adapter detection now goes through `PeftConfig.from_pretrained`, which resolves a local
directory and an HF hub repo id the same way `AutoModel.from_pretrained` does -- so a
fine-tuned adapter published on the hub (model.source == "huggingface") is handled the same
way as one saved to a local checkpoint directory, with no separate code path. These tests
only exercise local directories (no network needed): PeftConfig.from_pretrained resolves a
local path without any hub lookup, so the detection logic is fully testable offline.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from privaudit.models.loader import LoadedModel, ModelConfig, _try_load_peft_config, load_model


def _write_adapter_config(directory, base_model_name_or_path=None):
    config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8, "target_modules": ["c_attn"]}
    if base_model_name_or_path is not None:
        config["base_model_name_or_path"] = base_model_name_or_path
    (directory / "adapter_config.json").write_text(json.dumps(config))


def test_try_load_peft_config_detects_local_adapter_dir(tmp_path):
    _write_adapter_config(tmp_path, "gpt2")
    peft_config = _try_load_peft_config(str(tmp_path))
    assert peft_config is not None
    assert peft_config.base_model_name_or_path == "gpt2"


def test_try_load_peft_config_returns_none_for_non_adapter_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt2"}))
    assert _try_load_peft_config(str(tmp_path)) is None


def test_load_model_raises_when_local_checkpoint_path_missing(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_model({"source": "local_checkpoint", "identifier": str(missing), "access": "white_box"})


def test_load_model_raises_clear_error_when_adapter_has_no_resolvable_base_model(tmp_path):
    _write_adapter_config(tmp_path, base_model_name_or_path=None)  # adapter_config.json present, but no base model recorded
    with pytest.raises(ValueError, match="model.base_model"):
        load_model({"source": "local_checkpoint", "identifier": str(tmp_path), "access": "white_box"})


def test_load_model_raises_when_base_model_set_but_not_an_adapter_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt2"}))  # looks like a full checkpoint, not an adapter
    with pytest.raises(ValueError, match="doesn't look like a LoRA/PEFT checkpoint"):
        load_model({
            "source": "local_checkpoint",
            "identifier": str(tmp_path),
            "access": "white_box",
            "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        })


def test_load_model_raises_on_unknown_source():
    with pytest.raises(ValueError, match="Unknown model.source"):
        load_model({"source": "carrier_pigeon", "identifier": "x", "access": "white_box"})


def test_load_model_api_endpoint_requires_black_box_access():
    with pytest.raises(ValueError, match="black_box"):
        load_model({"source": "api_endpoint", "identifier": "https://example.com/generate", "access": "white_box"})


# ---------------------------------------------------------------------------
# OpenAI source
# ---------------------------------------------------------------------------

def test_load_model_openai_requires_black_box_access():
    with pytest.raises(ValueError, match="black_box"):
        load_model({"source": "openai", "identifier": "gpt-4o-mini", "access": "white_box", "api_key": "sk-test"})


def test_load_model_openai_requires_an_api_key_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_model({"source": "openai", "identifier": "gpt-4o-mini", "access": "black_box"})


def test_load_model_openai_succeeds_with_explicit_api_key():
    loaded = load_model({"source": "openai", "identifier": "gpt-4o-mini", "access": "black_box", "api_key": "sk-test"})
    assert loaded.cfg.api_key == "sk-test"
    assert not loaded.is_white_box


def test_load_model_openai_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    loaded = load_model({"source": "openai", "identifier": "gpt-4o-mini", "access": "black_box"})
    assert loaded.cfg.api_key == "sk-from-env"


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_openai_generate_posts_chat_completions_and_parses_response(monkeypatch):
    from privaudit.models.loader import LoadedModel

    captured = {}

    def _fake_urlopen(req):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.headers.items()}
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse({"choices": [{"message": {"content": "hello back"}}]})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    cfg = ModelConfig(source="openai", identifier="gpt-4o-mini", access="black_box", api_key="sk-test")
    result = LoadedModel(cfg).generate_greedy("Hello", max_new_tokens=50)

    assert result == "hello back"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer sk-test"
    assert captured["body"]["model"] == "gpt-4o-mini"
    assert captured["body"]["messages"] == [{"role": "user", "content": "Hello"}]
    assert captured["body"]["max_tokens"] == 50


def test_openai_generate_respects_custom_base_url(monkeypatch):
    from privaudit.models.loader import LoadedModel

    captured = {}

    def _fake_urlopen(req):
        captured["url"] = req.full_url
        return _FakeHTTPResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    cfg = ModelConfig(
        source="openai", identifier="my-model", access="black_box", api_key="sk-test",
        base_url="https://my-proxy.example.com/v1/",
    )
    LoadedModel(cfg).generate_greedy("hi", max_new_tokens=10)
    assert captured["url"] == "https://my-proxy.example.com/v1/chat/completions"


# ---------------------------------------------------------------------------
# max_context_length() / sequence_stats() truncation-length clamping
#
# Long ("many chars") dataset text tokenizes to many tokens; without clamping,
# MIA's/EZ-MIA's max_length/sequence_length params could exceed the actual loaded model's
# position-embedding table and crash deep inside model.forward() with an opaque IndexError
# instead of just truncating further, the way `truncation=True` already does for every
# shorter case.
# ---------------------------------------------------------------------------

def _white_box_cfg():
    return ModelConfig(source="local_checkpoint", identifier="x", access="white_box")


class _FakeTorchModel:
    """Duck-typed stand-in for a loaded transformers model: only exposes what
    max_context_length()/sequence_stats() actually touch."""

    def __init__(self, max_position_embeddings=None, vocab_size=4):
        self.config = SimpleNamespace()
        if max_position_embeddings is not None:
            self.config.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size

    def eval(self):
        return self

    def __call__(self, **kwargs):
        batch, seq_len = kwargs["input_ids"].shape
        return SimpleNamespace(logits=torch.zeros(batch, seq_len, self.vocab_size))


class _FakeTokenizer:
    def __init__(self, model_max_length=None, calls=None):
        if model_max_length is not None:
            self.model_max_length = model_max_length
        self._calls = calls if calls is not None else []

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=512):
        self._calls.append(max_length)
        seq_len = min(max_length, 3)
        n = len(texts)
        return {
            "input_ids": torch.randint(0, 4, (n, seq_len)),
            "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
        }


def test_max_context_length_none_when_model_not_loaded():
    loaded = LoadedModel(ModelConfig(source="openai", identifier="x", access="black_box"))
    assert loaded.max_context_length() is None


def test_max_context_length_reads_model_config():
    loaded = LoadedModel(_white_box_cfg(), model=_FakeTorchModel(max_position_embeddings=1024), tokenizer=_FakeTokenizer())
    assert loaded.max_context_length() == 1024


def test_max_context_length_ignores_tokenizer_sentinel():
    # transformers reports int(1e30) for model_max_length when a tokenizer's config never set
    # a real one -- that must not be mistaken for an actual (tiny) limit.
    loaded = LoadedModel(
        _white_box_cfg(),
        model=_FakeTorchModel(max_position_embeddings=1024),
        tokenizer=_FakeTokenizer(model_max_length=int(1e30)),
    )
    assert loaded.max_context_length() == 1024


def test_max_context_length_takes_the_smaller_of_config_and_tokenizer():
    loaded = LoadedModel(
        _white_box_cfg(),
        model=_FakeTorchModel(max_position_embeddings=1024),
        tokenizer=_FakeTokenizer(model_max_length=256),
    )
    assert loaded.max_context_length() == 256


def test_max_context_length_none_when_neither_source_reports_a_limit():
    loaded = LoadedModel(_white_box_cfg(), model=_FakeTorchModel(), tokenizer=_FakeTokenizer())
    assert loaded.max_context_length() is None


def test_sequence_stats_clamps_max_length_to_the_models_actual_context_window():
    calls = []
    loaded = LoadedModel(
        _white_box_cfg(),
        model=_FakeTorchModel(max_position_embeddings=8),
        tokenizer=_FakeTokenizer(calls=calls),
        device=torch.device("cpu"),
    )

    loaded.sequence_stats(["a very long piece of text" * 50] * 2, batch_size=2, max_length=4096, k_percent=20)

    # Clamped from the requested 4096 down to the model's real 8-token window, not passed
    # through as-is.
    assert calls == [8]


def test_sequence_stats_leaves_max_length_untouched_when_it_is_already_smaller():
    calls = []
    loaded = LoadedModel(
        _white_box_cfg(),
        model=_FakeTorchModel(max_position_embeddings=1024),
        tokenizer=_FakeTokenizer(calls=calls),
        device=torch.device("cpu"),
    )

    loaded.sequence_stats(["short text"], batch_size=1, max_length=64, k_percent=20)

    assert calls == [64]
