"""EM pipeline test against a fake model + whitespace tokenizer (no real transformer needed).

Exercises attacks/em.py end-to-end: sample construction (prefix/continuation split, short
sequences discarded), the fixed-instruction generation call, and cell-level metric averaging.
`_bertscore_f1` is monkeypatched since BERTScore needs a real embedding model/network and
isn't something a hermetic unit test should depend on.
"""
from __future__ import annotations

import pytest

from privaudit.attacks import em
from privaudit.data.member_split import MemberSplit

_PREFIX_LEN = 5
_CONTINUATION_LEN = 10


class _FakeTokenizer:
    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)


class _FakeModel:
    """Regurgitates the exact continuation for the "memorized" prefix, something unrelated otherwise."""

    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def generate_greedy(self, prompt, max_new_tokens):
        if "p0 p1 p2 p3 p4" in prompt:
            return "c0 c1 c2 c3 c4 c5 c6 c7 c8 c9"  # verbatim match of the groundtruth continuation
        return "zz0 zz1 zz2 zz3 zz4 zz5 zz6 zz7 zz8 zz9"  # disjoint from any groundtruth


@pytest.fixture(autouse=True)
def _stub_bertscore(monkeypatch):
    monkeypatch.setattr(em, "_bertscore_f1", lambda predictions, references: [1.0] * len(predictions))


def test_build_samples_discards_short_sequences_and_splits_correctly():
    tokenizer = _FakeTokenizer()
    member_texts = [
        "p0 p1 p2 p3 p4 c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 extra",  # 16 tokens: > 5+10 threshold
        "too short only ten tok ens here yes ok",  # 9 tokens: discarded
    ]
    samples = em._build_samples(member_texts, tokenizer, _PREFIX_LEN, _CONTINUATION_LEN, max_samples=None)
    assert len(samples) == 1
    assert samples[0]["prefix"] == "p0 p1 p2 p3 p4"
    assert samples[0]["groundtruth"] == "c0 c1 c2 c3 c4 c5 c6 c7 c8 c9"


def test_em_run_metrics_over_memorized_and_non_memorized_pairs():
    member_split = MemberSplit(
        member_texts=[
            "p0 p1 p2 p3 p4 c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 extra",  # generates a verbatim match
            "q0 q1 q2 q3 q4 g0 g1 g2 g3 g4 g5 g6 g7 g8 g9 extra",  # generates something disjoint
            "short seq under threshold",  # discarded (too short)
        ],
        nonmember_texts=[],
    )
    attack_cfg = {"family": "EM", "variant": "default", "params": {"prefix_len": _PREFIX_LEN, "continuation_len": _CONTINUATION_LEN}}

    output = em.run(_FakeModel(), member_split, attack_cfg, output_cfg={})

    assert output.family == "EM"
    # 2 samples total; only the first is a verbatim (10-token) memorization event.
    assert output.metrics["mem_at_10"] == pytest.approx(0.5)
    # groundtruth continuations are only 10 tokens long, so n=20/30/50 can never match.
    assert output.metrics["mem_at_20"] == 0.0
    assert output.metrics["mem_at_30"] == 0.0
    assert output.metrics["mem_at_50"] == 0.0
    assert output.metrics["rouge_l"] == pytest.approx(0.5)  # 1.0 (verbatim) + 0.0 (disjoint), averaged
    assert output.canonical_score == output.metrics["mem_at_50"]
    assert output.raw_generations is None  # off by default


def test_em_run_returns_raw_generations_when_requested():
    member_split = MemberSplit(
        member_texts=["p0 p1 p2 p3 p4 c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 extra"],
        nonmember_texts=[],
    )
    attack_cfg = {"family": "EM", "variant": "default", "params": {"prefix_len": _PREFIX_LEN, "continuation_len": _CONTINUATION_LEN}}
    output = em.run(_FakeModel(), member_split, attack_cfg, output_cfg={"return_raw_generations": True})
    assert output.raw_generations == [
        {"prefix": "p0 p1 p2 p3 p4", "groundtruth": "c0 c1 c2 c3 c4 c5 c6 c7 c8 c9", "generation": "c0 c1 c2 c3 c4 c5 c6 c7 c8 c9"}
    ]


def test_em_run_raises_when_no_sequence_is_long_enough():
    member_split = MemberSplit(member_texts=["too short"], nonmember_texts=[])
    attack_cfg = {"family": "EM", "variant": "default", "params": {"prefix_len": _PREFIX_LEN, "continuation_len": _CONTINUATION_LEN}}
    with pytest.raises(ValueError, match="No training sequences"):
        em.run(_FakeModel(), member_split, attack_cfg, output_cfg={})


def test_em_run_respects_custom_ngram_ns_and_tracks_canonical_score():
    member_split = MemberSplit(
        member_texts=["p0 p1 p2 p3 p4 c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 extra"],
        nonmember_texts=[],
    )
    attack_cfg = {
        "family": "EM", "variant": "default",
        "params": {"prefix_len": _PREFIX_LEN, "continuation_len": _CONTINUATION_LEN, "ngram_ns": [3, 7]},
    }
    output = em.run(_FakeModel(), member_split, attack_cfg, output_cfg={})
    assert set(k for k in output.metrics if k.startswith("mem_at_")) == {"mem_at_3", "mem_at_7"}
    assert "mem_at_50" not in output.metrics
    # canonical score tracks the largest requested n, not a hardcoded mem_at_50.
    assert output.canonical_score == output.metrics["mem_at_7"]


def test_em_run_max_samples_caps_pairs_built():
    member_texts = [f"p{i}0 p{i}1 p{i}2 p{i}3 p{i}4 c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 extra" for i in range(5)]
    member_split = MemberSplit(member_texts=member_texts, nonmember_texts=[])
    attack_cfg = {
        "family": "EM", "variant": "default",
        "params": {"prefix_len": _PREFIX_LEN, "continuation_len": _CONTINUATION_LEN, "max_samples": 2},
    }
    output = em.run(_FakeModel(), member_split, attack_cfg, output_cfg={"return_raw_generations": True})
    assert len(output.raw_generations) == 2
