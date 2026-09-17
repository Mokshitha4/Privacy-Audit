"""End-to-end integration test: one job config through the real pipeline.

Unlike tests/test_em.py, test_mia.py, test_ez_mia.py -- which call each attack module's
run() directly -- this exercises the actual glue code: job-config schema validation,
runner.run_job()'s orchestration, load_member_split() reading real CSV files from a temp
directory, all three attack families running together (including a reference_model for
EZ_MIA), and the final build_report()/validate_report() round trip. Two things are mocked:
model loading itself (privaudit.runner.load_model), via one fake "LoadedModel" that
implements the same duck-typed interface (tokenizer, .model, .device, generate_greedy(),
sequence_stats()) the real one does; and EM's BERTScore-F1 call, which otherwise downloads a
real embedding model over the network on first use (exact BERTScore behavior is out of scope
for a pipeline-wiring test, and is unrelated to anything mocked here). Nothing in this file
needs a GPU, network access, or real checkpoints. Also runs the same config through the
actual `privaudit run` CLI entry point.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from privaudit import cli, runner
from privaudit.attacks import em
from privaudit.report import REPORT_SCHEMA, validate_report


class _FakeTokenizer:
    """Deterministic whitespace tokenizer: real tensor output for the EZ-MIA raw-tokenizer
    path, plus tokenize()/convert_tokens_to_string() for EM.
    """

    def __init__(self, vocab_size=40):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def _token_id(self, token: str) -> int:
        return (sum(ord(c) for c in token) % (self.vocab_size - 1)) + 1  # 1..vocab_size-1; 0 is pad

    def __call__(self, texts, padding=None, truncation=None, max_length=16, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        rows = [[self._token_id(tok) for tok in text.split()] for text in texts]
        if truncation:
            rows = [r[:max_length] for r in rows]
        target_len = max_length if padding == "max_length" else max((len(r) for r in rows), default=1)
        input_ids = torch.zeros(len(rows), target_len, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), target_len, dtype=torch.long)
        for i, ids in enumerate(rows):
            n = min(len(ids), target_len)
            if n > 0:
                input_ids[i, :n] = torch.tensor(ids[:n], dtype=torch.long)
                attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeCausalLM:
    """Uniform (all-zero) logits: enough for EZ-MIA's forward pass to run and produce finite,
    well-defined (if uninteresting) scores -- correctness of the EZ-MIA math itself is already
    pinned in tests/test_ez_mia.py.
    """

    def __init__(self, vocab_size=40):
        self.vocab_size = vocab_size

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask=None, labels=None):
        batch, seq_len = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(batch, seq_len, self.vocab_size), loss=torch.tensor(0.0))


class _FakeLoadedModel:
    """Duck-typed stand-in for models.loader.LoadedModel covering every attack's needs."""

    def __init__(self, seed=0, vocab_size=40):
        self.tokenizer = _FakeTokenizer(vocab_size=vocab_size)
        self.device = torch.device("cpu")
        self.model = _FakeCausalLM(vocab_size=vocab_size)
        self._rng = np.random.RandomState(seed)

    def generate_greedy(self, prompt, max_new_tokens):
        return " ".join(f"gen{i}" for i in range(min(max_new_tokens, 10)))

    def sequence_stats(self, texts, batch_size=8, max_length=512, k_percent=20):
        is_member = np.array([t.startswith("MEMBER") for t in texts])
        noise = self._rng.normal(0, 0.05, size=len(texts))
        loss = np.where(is_member, 0.5, 3.0) + noise
        return {
            "loss": loss,
            "perplexity": np.exp(loss),
            "confidence": np.where(is_member, 0.9, 0.2) + noise,
            "min_k_prob": np.where(is_member, 0.8, 0.1) + noise,
        }


def _write_split_csvs(data_dir, n_members=25, n_nonmembers=25):
    # Long enough (17 whitespace tokens) to clear EM's default prefix_len(5)+continuation_len(10)
    # threshold; "MEMBER_"/"NONMEMBER_" prefixes drive the fake sequence_stats' separable signal.
    member_rows = [{"note": f"MEMBER_{i} p0 p1 p2 p3 p4 c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 extra"} for i in range(n_members)]
    nonmember_rows = [{"note": f"NONMEMBER_{i} words words words words words words words words"} for i in range(n_nonmembers)]
    pd.DataFrame(member_rows).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(nonmember_rows).to_csv(data_dir / "test.csv", index=False)


def _build_job_config(data_dir) -> dict:
    return {
        "model": {"source": "huggingface", "identifier": "fake/target-model", "access": "white_box"},
        "reference_model": {"source": "huggingface", "identifier": "fake/reference-model", "access": "white_box"},
        "data": {
            "role": "member_nonmember_split",
            "schema": {"format": "csv", "member_file": "train.csv", "nonmember_file": "test.csv", "text_field": "note"},
            "path": str(data_dir),
        },
        "attacks": [
            {"family": "EM", "variant": "default", "params": {"prefix_len": 5, "continuation_len": 10}},
            {"family": "MIA", "variant": "default", "params": {"num_members": 20, "num_nonmembers": 20}},
            {"family": "EZ_MIA", "variant": "default", "params": {"sequence_length": 16, "num_members": 5, "num_nonmembers": 5}},
        ],
        "output": {"return_raw_generations": True},
    }


@pytest.fixture
def _mocked_model_loading(monkeypatch):
    monkeypatch.setattr(runner, "load_model", lambda model_cfg: _FakeLoadedModel())
    monkeypatch.setattr(em, "_bertscore_f1", lambda predictions, references: [0.5] * len(predictions))


def test_run_job_end_to_end_all_three_attacks(tmp_path, _mocked_model_loading):
    _write_split_csvs(tmp_path)
    job_cfg = _build_job_config(tmp_path)

    report = runner.run_job(job_cfg)

    assert set(report["metrics"]) == {"EM", "MIA", "EZ_MIA"}
    assert set(report["canonical_scores"]) == {"EM", "MIA", "EZ_MIA"}
    assert report["model_id"] == "fake/target-model"

    # Report is already validated inside build_report(); re-check directly too.
    validate_report(report)

    for family, metrics in report["metrics"].items():
        for metric_name, value in metrics.items():
            assert isinstance(value, (int, float)), f"{family}.{metric_name} is not numeric: {value!r}"
            assert value == value, f"{family}.{metric_name} is NaN"  # NaN != NaN

    assert "EM" in report["raw_generations"]
    assert len(report["raw_generations"]["EM"]) > 0


def test_run_job_respects_return_raw_generations_false(tmp_path, _mocked_model_loading):
    _write_split_csvs(tmp_path)
    job_cfg = _build_job_config(tmp_path)
    job_cfg["output"]["return_raw_generations"] = False
    job_cfg["attacks"] = [job_cfg["attacks"][0]]  # EM only, to keep this test fast

    report = runner.run_job(job_cfg)
    assert "raw_generations" not in report


def test_run_job_reports_progress_through_each_stage(tmp_path, _mocked_model_loading):
    _write_split_csvs(tmp_path)
    job_cfg = _build_job_config(tmp_path)
    messages = []

    runner.run_job(job_cfg, on_progress=messages.append)

    joined = " ".join(messages)
    assert any("model" in m.lower() for m in messages)
    assert any("reference model" in m.lower() for m in messages)
    assert any("dataset" in m.lower() for m in messages)
    assert "Running EM" in joined and "Running MIA" in joined and "Running EZ_MIA" in joined
    assert any("report" in m.lower() for m in messages)


def test_run_job_on_progress_defaults_to_a_no_op(tmp_path, _mocked_model_loading):
    _write_split_csvs(tmp_path)
    job_cfg = _build_job_config(tmp_path)
    job_cfg["attacks"] = [job_cfg["attacks"][0]]
    runner.run_job(job_cfg)  # must not raise with no on_progress given


def test_run_job_releases_model_weights_after_a_successful_run(tmp_path, monkeypatch):
    """A long-lived caller (the UI server) runs many jobs in one process; without releasing
    weights after each job, memory grows unbounded across runs. See runner._release()."""
    loaded_instances = []

    def _tracking_load_model(model_cfg):
        instance = _FakeLoadedModel()
        loaded_instances.append(instance)
        return instance

    monkeypatch.setattr(runner, "load_model", _tracking_load_model)
    monkeypatch.setattr(em, "_bertscore_f1", lambda predictions, references: [0.5] * len(predictions))

    _write_split_csvs(tmp_path)
    job_cfg = _build_job_config(tmp_path)

    runner.run_job(job_cfg)

    assert len(loaded_instances) == 2  # model + reference_model
    for instance in loaded_instances:
        assert instance.model is None
        assert instance.tokenizer is None


def test_run_job_releases_model_weights_even_when_an_attack_raises(tmp_path, monkeypatch):
    loaded_instances = []

    def _tracking_load_model(model_cfg):
        instance = _FakeLoadedModel()
        loaded_instances.append(instance)
        return instance

    monkeypatch.setattr(runner, "load_model", _tracking_load_model)

    _write_split_csvs(tmp_path, n_members=1, n_nonmembers=1)  # too few for MIA's 5-fold CV -> raises
    job_cfg = _build_job_config(tmp_path)
    job_cfg["attacks"] = [job_cfg["attacks"][1]]  # MIA only

    with pytest.raises(ValueError, match="5-fold"):
        runner.run_job(job_cfg)

    # Both model and reference_model are loaded unconditionally by run_job (before any attack
    # runs), even though only MIA -- which doesn't use a reference model -- was requested here.
    assert len(loaded_instances) == 2
    assert all(instance.model is None for instance in loaded_instances)


def test_run_job_rejects_rag_corpus_role(tmp_path, _mocked_model_loading):
    job_cfg = _build_job_config(tmp_path)
    job_cfg["data"] = {
        "role": "rag_corpus", "path": str(tmp_path),
        "schema": {"format": "csv", "member_file": "train.csv", "nonmember_file": "test.csv", "text_field": "note"},
    }
    with pytest.raises(ValueError, match="not supported"):
        runner.run_job(job_cfg)


def test_cli_run_end_to_end_writes_valid_report_file(tmp_path, _mocked_model_loading):
    _write_split_csvs(tmp_path)
    job_cfg = _build_job_config(tmp_path)
    job_cfg["attacks"] = [job_cfg["attacks"][0], job_cfg["attacks"][1]]  # EM + MIA, keep it fast

    config_path = tmp_path / "job.json"
    output_path = tmp_path / "report.json"
    config_path.write_text(json.dumps(job_cfg))

    cli.main(["run", "--config", str(config_path), "--output", str(output_path)])

    report = json.loads(output_path.read_text())
    assert set(report["metrics"]) == {"EM", "MIA"}
    import jsonschema
    jsonschema.validate(report, REPORT_SCHEMA)


def test_cli_validate_end_to_end(tmp_path, capsys):
    job_cfg = _build_job_config(tmp_path)
    config_path = tmp_path / "job.json"
    config_path.write_text(json.dumps(job_cfg))

    cli.main(["validate", "--config", str(config_path)])
    assert "Config is valid." in capsys.readouterr().out


def test_cli_run_prints_progress(tmp_path, capsys, _mocked_model_loading):
    _write_split_csvs(tmp_path)
    job_cfg = _build_job_config(tmp_path)
    job_cfg["attacks"] = [job_cfg["attacks"][0]]  # EM only, keep it fast
    config_path = tmp_path / "job.json"
    output_path = tmp_path / "report.json"
    config_path.write_text(json.dumps(job_cfg))

    cli.main(["run", "--config", str(config_path), "--output", str(output_path)])

    out = capsys.readouterr().out
    assert "[privaudit]" in out
    assert "Running EM" in out


def test_cli_explain_prints_markdown_summary(tmp_path, capsys, _mocked_model_loading):
    _write_split_csvs(tmp_path)
    job_cfg = _build_job_config(tmp_path)
    job_cfg["attacks"] = [job_cfg["attacks"][0]]  # EM only
    report = runner.run_job(job_cfg)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    cli.main(["explain", "--report", str(report_path)])

    out = capsys.readouterr().out
    assert "Exact Memorization" in out
    assert "generated locally" in out


def test_cli_explain_survives_a_non_utf8_console_encoding(tmp_path, monkeypatch, _mocked_model_loading):
    """Regression test: explain's output includes emoji badges (see privaudit/explain.py's
    _VERDICT_BADGE), which crash a plain print() on a Windows console defaulting to cp1252 --
    reproduced here with a real cp1252-encoded stream standing in for that console, rather than
    pytest's own (UTF-8-capable) capsys."""
    _write_split_csvs(tmp_path)
    job_cfg = _build_job_config(tmp_path)
    job_cfg["attacks"] = [job_cfg["attacks"][0]]  # EM only
    report = runner.run_job(job_cfg)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    import io
    buffer = io.BytesIO()
    fake_console = io.TextIOWrapper(buffer, encoding="cp1252")
    monkeypatch.setattr("sys.stdout", fake_console)

    cli.main(["explain", "--report", str(report_path)])  # must not raise UnicodeEncodeError

    fake_console.flush()
    assert b"Exact Memorization" in buffer.getvalue()


def test_cli_explain_writes_to_output_file_when_given(tmp_path, capsys, _mocked_model_loading):
    _write_split_csvs(tmp_path)
    job_cfg = _build_job_config(tmp_path)
    job_cfg["attacks"] = [job_cfg["attacks"][0]]
    report = runner.run_job(job_cfg)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    summary_path = tmp_path / "summary.md"

    cli.main(["explain", "--report", str(report_path), "--output", str(summary_path)])

    assert "Exact Memorization" in summary_path.read_text()
    assert "written to" in capsys.readouterr().out


def test_cli_explain_llm_requires_provider(tmp_path, capsys, _mocked_model_loading):
    _write_split_csvs(tmp_path)
    job_cfg = _build_job_config(tmp_path)
    job_cfg["attacks"] = [job_cfg["attacks"][0]]
    report = runner.run_job(job_cfg)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    with pytest.raises(SystemExit):
        cli.main(["explain", "--report", str(report_path), "--llm"])
    assert "--provider" in capsys.readouterr().err


def test_cli_explain_llm_appends_narration_on_success(tmp_path, capsys, monkeypatch, _mocked_model_loading):
    _write_split_csvs(tmp_path)
    job_cfg = _build_job_config(tmp_path)
    job_cfg["attacks"] = [job_cfg["attacks"][0]]
    report = runner.run_job(job_cfg)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    from privaudit import llm_explain
    monkeypatch.setattr(llm_explain, "explain_with_llm", lambda report, config: "A friendly narration.")

    cli.main(["explain", "--report", str(report_path), "--llm", "--provider", "anthropic", "--api-key", "sk-test"])

    out = capsys.readouterr().out
    assert "Exact Memorization" in out  # rule-based summary still present
    assert "AI narration (anthropic)" in out
    assert "A friendly narration." in out


def test_cli_explain_llm_failure_falls_back_to_rule_based_summary(tmp_path, capsys, monkeypatch, _mocked_model_loading):
    _write_split_csvs(tmp_path)
    job_cfg = _build_job_config(tmp_path)
    job_cfg["attacks"] = [job_cfg["attacks"][0]]
    report = runner.run_job(job_cfg)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    from privaudit import llm_explain

    def _boom(report, config):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(llm_explain, "explain_with_llm", _boom)

    cli.main(["explain", "--report", str(report_path), "--llm", "--provider", "anthropic", "--api-key", "sk-test"])

    captured = capsys.readouterr()
    assert "Exact Memorization" in captured.out  # rule-based summary still shown
    assert "network unreachable" in captured.err
    assert "AI narration" not in captured.out
