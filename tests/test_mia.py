"""MIA pipeline test against a fake white-box model (no real transformer needed).

This checks the RandomForest + 5-fold-CV + TPR@FPR wiring end-to-end (attacks/mia.py) with
clearly separable synthetic per-sequence features. It is a sanity/correctness check on the
pipeline, not a reproduction of the paper's numbers -- those require the real fine-tuned
checkpoints (Llama-3-8B-Instruct, Med-LLaMA, etc.) and a GPU, neither available in this
environment. See test_metrics.py for the exact-reproduction test against EZ-MIA/results.csv.
"""
from __future__ import annotations

import numpy as np
import pytest

from privaudit.attacks import mia
from privaudit.data.member_split import MemberSplit


class _FakeWhiteBoxModel:
    """Duck-typed stand-in for LoadedModel: separable features keyed off text content."""

    def __init__(self):
        self.calls = []  # records (len(texts), max_length, k_percent) per sequence_stats call

    def sequence_stats(self, texts, batch_size=8, max_length=512, k_percent=20):
        self.calls.append((len(texts), max_length, k_percent))
        rng = np.random.RandomState(0)
        is_member = np.array([t.startswith("MEMBER") for t in texts])
        noise = rng.normal(0, 0.05, size=len(texts))
        loss = np.where(is_member, 0.5, 3.0) + noise
        perplexity = np.exp(loss)
        confidence = np.where(is_member, 0.9, 0.2) + noise
        min_k_prob = np.where(is_member, 0.8, 0.1) + noise
        return {"loss": loss, "perplexity": perplexity, "confidence": confidence, "min_k_prob": min_k_prob}


def test_mia_run_recovers_clearly_separable_membership_signal():
    member_split = MemberSplit(
        member_texts=[f"MEMBER_{i}" for i in range(30)],
        nonmember_texts=[f"NONMEMBER_{i}" for i in range(30)],
    )
    attack_cfg = {"family": "MIA", "variant": "default", "params": {"batch_size": 4, "seed": 42}}

    output = mia.run(_FakeWhiteBoxModel(), member_split, attack_cfg, output_cfg={})

    assert output.family == "MIA"
    assert set(output.metrics) == {"roc_auc", "accuracy", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr"}
    assert output.metrics["roc_auc"] > 0.9
    assert output.metrics["accuracy"] > 0.9
    assert output.canonical_score == output.metrics["tpr_at_5pct_fpr"]


def test_mia_run_raises_on_too_few_sequences_for_5_fold_cv():
    member_split = MemberSplit(member_texts=["MEMBER_0", "MEMBER_1"], nonmember_texts=["NONMEMBER_0", "NONMEMBER_1"])
    attack_cfg = {"family": "MIA", "variant": "default"}
    with pytest.raises(ValueError, match="5-fold"):
        mia.run(_FakeWhiteBoxModel(), member_split, attack_cfg, output_cfg={})


def test_mia_num_members_and_num_nonmembers_cap_the_sample_count():
    member_split = MemberSplit(
        member_texts=[f"MEMBER_{i}" for i in range(100)],
        nonmember_texts=[f"NONMEMBER_{i}" for i in range(100)],
    )
    fake_model = _FakeWhiteBoxModel()
    attack_cfg = {
        "family": "MIA", "variant": "default",
        "params": {"num_members": 12, "num_nonmembers": 40, "seed": 42},
    }
    mia.run(fake_model, member_split, attack_cfg, output_cfg={})

    # min(12, 40, 100, 100) == 12 sequences drawn per class, regardless of the 100 available.
    assert fake_model.calls == [(12, 512, 20)] * 2


def test_mia_max_length_and_k_percent_params_are_passed_through():
    member_split = MemberSplit(
        member_texts=[f"MEMBER_{i}" for i in range(10)],
        nonmember_texts=[f"NONMEMBER_{i}" for i in range(10)],
    )
    fake_model = _FakeWhiteBoxModel()
    attack_cfg = {
        "family": "MIA", "variant": "default",
        "params": {"max_length": 256, "k_percent": 10, "n_folds": 5},
    }
    mia.run(fake_model, member_split, attack_cfg, output_cfg={})
    assert fake_model.calls == [(10, 256, 10.0)] * 2


def test_mia_validate_requires_member_nonmember_split_and_white_box():
    job_cfg = {"model": {"access": "black_box"}, "data": {"role": "member_nonmember_split"}}
    with pytest.raises(ValueError, match="white_box"):
        mia.validate(job_cfg, {"family": "MIA", "variant": "default"})

    job_cfg = {"model": {"access": "white_box"}, "data": {"role": "rag_corpus"}}
    with pytest.raises(ValueError, match="member_nonmember_split"):
        mia.validate(job_cfg, {"family": "MIA", "variant": "default"})
