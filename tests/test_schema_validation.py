from __future__ import annotations

import pytest

from privaudit.report import validate_job_config


def _base_config(**attacks_kwargs):
    return {
        "model": {"source": "huggingface", "identifier": "gpt2", "access": "white_box"},
        "data": {
            "role": "member_nonmember_split",
            "path": "/data/medqa",
            "schema": {"format": "csv", "member_file": "train.csv", "nonmember_file": "test.csv", "text_field": "question"},
        },
        "attacks": [{"family": "EM", "variant": "default"}],
    }


def test_valid_minimal_config_passes():
    validate_job_config(_base_config())  # should not raise


def test_missing_required_top_level_field_fails_fast():
    cfg = _base_config()
    del cfg["data"]
    with pytest.raises(ValueError, match="'data' is a required property"):
        validate_job_config(cfg)


def test_missing_data_schema_fails_fast():
    cfg = _base_config()
    del cfg["data"]["schema"]
    with pytest.raises(ValueError, match="'schema' is a required property"):
        validate_job_config(cfg)


def test_missing_model_identifier_fails_fast():
    cfg = _base_config()
    del cfg["model"]["identifier"]
    with pytest.raises(ValueError, match="identifier"):
        validate_job_config(cfg)


def test_unknown_attack_family_rejected():
    cfg = _base_config()
    cfg["attacks"] = [{"family": "PA", "variant": "multi_choice"}]
    with pytest.raises(ValueError):
        validate_job_config(cfg)


def test_duplicate_attack_family_rejected():
    cfg = _base_config()
    cfg["attacks"] = [{"family": "EM", "variant": "default"}, {"family": "EM", "variant": "default"}]
    with pytest.raises(ValueError, match="Duplicate attack families"):
        validate_job_config(cfg)


def test_invalid_model_source_rejected():
    cfg = _base_config()
    cfg["model"]["source"] = "some_random_place"
    with pytest.raises(ValueError):
        validate_job_config(cfg)


def test_unexpected_top_level_field_rejected():
    cfg = _base_config()
    cfg["typo_field"] = True
    with pytest.raises(ValueError):
        validate_job_config(cfg)


def test_openai_source_is_accepted():
    cfg = _base_config()
    cfg["model"] = {"source": "openai", "identifier": "gpt-4o-mini", "access": "black_box", "api_key": "sk-test"}
    validate_job_config(cfg)  # should not raise


def test_openai_base_url_is_accepted():
    cfg = _base_config()
    cfg["model"] = {
        "source": "openai", "identifier": "gpt-4o-mini", "access": "black_box",
        "api_key": "sk-test", "base_url": "https://my-proxy.example.com/v1",
    }
    validate_job_config(cfg)  # should not raise
