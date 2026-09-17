"""Tests for the rule-based, no-network report explanation."""
from __future__ import annotations

from privaudit import explain


class _NoRawGenerationsAccess(dict):
    """Raises if anything ever reads report["raw_generations"] -- explain.py must not need it."""

    def get(self, key, default=None):
        if key == "raw_generations":
            raise AssertionError("summarize() must not read raw_generations")
        return super().get(key, default)

    def __getitem__(self, key):
        if key == "raw_generations":
            raise AssertionError("summarize() must not read raw_generations")
        return super().__getitem__(key)


def _report(metrics: dict, raw_generations=None) -> dict:
    data = {"metrics": metrics}
    if raw_generations is not None:
        data["raw_generations"] = raw_generations
    return _NoRawGenerationsAccess(data)


def test_em_no_memorization():
    result = explain._explain_em({"mem_at_10": 0.0, "mem_at_50": 0.0, "rouge_l": 0.1, "bertscore_f1": 0.6})
    assert result["verdict"] == "none"
    assert "No exact memorization" in result["headline"]


def test_em_weak_memorization():
    # Band is computed on the largest n-gram window present (mem_at_50), matching the
    # canonical-score convention used elsewhere (attacks/em.py); mem_at_10 here is a distractor.
    result = explain._explain_em({"mem_at_10": 0.9, "mem_at_50": 0.02})
    assert result["verdict"] == "weak"


def test_em_strong_memorization():
    result = explain._explain_em({"mem_at_10": 0.3, "mem_at_50": 0.1})
    assert result["verdict"] == "strong"
    # uses the largest n-gram window present, not the first one
    assert "50-token" in result["detail"][0]


def test_mia_chance_level_is_no_evidence():
    result = explain._explain_mia({"roc_auc": 0.5, "tpr_at_5pct_fpr": 0.05, "tpr_at_1pct_fpr": 0.01})
    assert result["verdict"] == "none"


def test_mia_weak_signal():
    result = explain._explain_mia({"roc_auc": 0.6, "tpr_at_5pct_fpr": 0.10, "tpr_at_1pct_fpr": 0.02})
    assert result["verdict"] == "weak"


def test_mia_strong_signal():
    result = explain._explain_mia({"roc_auc": 0.8, "tpr_at_5pct_fpr": 0.25, "tpr_at_1pct_fpr": 0.15})
    assert result["verdict"] == "strong"


def test_ez_mia_bands():
    assert explain._explain_ez_mia({"score": 0.5})["verdict"] == "none"
    assert explain._explain_ez_mia({"score": 0.6})["verdict"] == "weak"
    assert explain._explain_ez_mia({"score": 0.9})["verdict"] == "strong"
    assert explain._explain_ez_mia({"score": 0.48})["verdict"] == "none"
    # distance-from-chance is symmetric: a below-chance score is banded the same as above-chance
    assert explain._explain_ez_mia({"score": 0.3})["verdict"] == "strong"  # |0.3-0.5| == 0.2 >= strong_at


def test_summarize_only_includes_families_present_in_metrics():
    report = _report({"EM": {"mem_at_50": 0.0}})
    result = explain.summarize(report)
    assert [f["family"] for f in result["families"]] == ["EM"]


def test_summarize_never_touches_raw_generations():
    report = _report(
        {"EM": {"mem_at_50": 0.2}, "MIA": {"tpr_at_5pct_fpr": 0.2, "roc_auc": 0.7}},
        raw_generations={"EM": [{"prefix": "p", "groundtruth": "g", "generation": "o"}]},
    )
    result = explain.summarize(report)  # must not raise
    assert len(result["families"]) == 2


def test_agreement_note_absent_for_a_single_family():
    report = _report({"EM": {"mem_at_50": 0.0}})
    assert explain.summarize(report)["agreement_note"] is None


def test_agreement_note_when_families_agree():
    report = _report({"EM": {"mem_at_50": 0.0}, "MIA": {"tpr_at_5pct_fpr": 0.05, "roc_auc": 0.5}})
    note = explain.summarize(report)["agreement_note"]
    assert "no signal" in note.lower()


def test_agreement_note_when_families_disagree():
    report = _report({"EM": {"mem_at_50": 0.3}, "MIA": {"tpr_at_5pct_fpr": 0.05, "roc_auc": 0.5}})
    note = explain.summarize(report)["agreement_note"]
    assert "disagree" in note.lower()
    assert "EM" in note and "MIA" in note


def test_disclaimer_always_present_and_mentions_locality():
    report = _report({"EM": {"mem_at_50": 0.0}})
    disclaimer = explain.summarize(report)["disclaimer"]
    assert "locally" in disclaimer
    assert "not sent anywhere" in disclaimer or "no data" in disclaimer.lower()


def test_render_markdown_includes_headline_badge_and_disclaimer():
    report = _report({"EM": {"mem_at_50": 0.2}})
    markdown = explain.render_markdown(explain.summarize(report))
    assert "Exact Memorization" in markdown
    assert "🔴" in markdown or "🟡" in markdown or "⚪" in markdown
    assert "generated locally" in markdown
