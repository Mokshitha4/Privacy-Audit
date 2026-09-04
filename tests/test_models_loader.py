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

import pytest

from privaudit.models.loader import _try_load_peft_config, load_model


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
