"""Deterministic unit test for the EZ-MIA scoring core (error_zone_pos_neg_sum_ratio).

This is the same tensor-math kernel ported from the original EZ-MIA/ez_score.py; the
values below are hand-computed so the test pins the exact formula rather than just its
sign/shape.
"""
from __future__ import annotations

import pytest
import torch

from privaudit.attacks import ez_mia
from privaudit.attacks.ez_mia import _error_zone_pos_neg_sum_ratio, run, validate
from privaudit.data.member_split import MemberSplit


def test_error_zone_ratio_hand_computed():
    # index 0 is dropped by ignore_bos regardless of mask/correctness.
    mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    target_ids = torch.tensor([0, 2, 3, 4, 5])
    top1_indices_tgt = torch.tensor([0, 9, 9, 4, 9])  # wrong at idx 1, 2, 4; correct at idx 3
    target_correct_lp = torch.tensor([-3.0, -1.0, -2.0, -0.1, -4.0])
    ref_correct_lp = torch.tensor([-2.0, -2.0, -1.0, -0.2, -1.0])

    # error-zone deltas (target - ref) at idx 1, 2, 4: +1.0, -1.0, -3.0
    # positive_sum = 1.0, negative_sum = 4.0 -> ratio = 0.25
    ratio = _error_zone_pos_neg_sum_ratio(
        target_correct_lp, ref_correct_lp, top1_indices_tgt, target_ids, mask,
        ignore_bos=True, min_tokens=2,
    )
    assert abs(ratio - 0.25) < 1e-6


def test_error_zone_ratio_no_error_zone_tokens_returns_zero():
    mask = torch.tensor([1.0, 1.0, 1.0])
    target_ids = torch.tensor([0, 2, 3])
    top1_indices_tgt = torch.tensor([0, 2, 3])  # target always correct -> empty error zone
    correct_lp = torch.tensor([-1.0, -1.0, -1.0])

    ratio = _error_zone_pos_neg_sum_ratio(
        correct_lp, correct_lp, top1_indices_tgt, target_ids, mask, ignore_bos=True, min_tokens=2,
    )
    assert ratio == 0.0


def test_error_zone_ratio_all_positive_deltas_returns_zero_not_inf():
    # negative_sum == 0 must fall back to 0.0 rather than dividing by (near) zero.
    mask = torch.tensor([1.0, 1.0, 1.0])
    target_ids = torch.tensor([0, 2, 3])
    top1_indices_tgt = torch.tensor([0, 9, 9])
    target_correct_lp = torch.tensor([-1.0, -0.5, -0.5])
    ref_correct_lp = torch.tensor([-1.0, -2.0, -2.0])  # target beats ref everywhere -> all positive deltas

    ratio = _error_zone_pos_neg_sum_ratio(
        target_correct_lp, ref_correct_lp, top1_indices_tgt, target_ids, mask, ignore_bos=True, min_tokens=2,
    )
    assert ratio == 0.0


def test_validate_requires_reference_model():
    job_cfg = {
        "model": {"access": "white_box"},
        "data": {"role": "member_nonmember_split"},
    }
    with pytest.raises(ValueError, match="reference_model"):
        validate(job_cfg, {"family": "EZ_MIA", "variant": "default"})


def test_validate_requires_reference_model_white_box():
    job_cfg = {
        "model": {"access": "white_box"},
        "data": {"role": "member_nonmember_split"},
        "reference_model": {"access": "black_box"},
    }
    with pytest.raises(ValueError, match="reference_model must also"):
        validate(job_cfg, {"family": "EZ_MIA", "variant": "default"})


def test_validate_passes_with_full_config():
    job_cfg = {
        "model": {"access": "white_box"},
        "data": {"role": "member_nonmember_split"},
        "reference_model": {"access": "white_box"},
    }
    validate(job_cfg, {"family": "EZ_MIA", "variant": "default"})  # should not raise


def test_run_requires_a_reference_model():
    member_split = MemberSplit(member_texts=["m"], nonmember_texts=["n"])
    with pytest.raises(ValueError, match="reference_model"):
        run(object(), member_split, {"family": "EZ_MIA", "variant": "default"}, output_cfg={}, reference_model=None)


def test_run_num_members_and_num_nonmembers_cap_the_sample_count(monkeypatch):
    calls = []

    def _fake_compute_ez_scores(target, reference, texts, batch_size, sequence_length):
        calls.append(len(texts))
        return [0.0] * len(texts)

    monkeypatch.setattr(ez_mia, "_compute_ez_scores", _fake_compute_ez_scores)

    member_split = MemberSplit(
        member_texts=[f"MEMBER_{i}" for i in range(50)],
        nonmember_texts=[f"NONMEMBER_{i}" for i in range(50)],
    )
    attack_cfg = {"family": "EZ_MIA", "variant": "default", "params": {"num_members": 7, "num_nonmembers": 20}}
    output = run(object(), member_split, attack_cfg, output_cfg={}, reference_model=object())

    assert calls == [7, 20]
    assert output.family == "EZ_MIA"


def test_run_uses_all_available_sequences_when_no_cap_given(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ez_mia, "_compute_ez_scores",
        lambda target, reference, texts, batch_size, sequence_length: calls.append(len(texts)) or [0.0] * len(texts),
    )
    member_split = MemberSplit(member_texts=["m1", "m2", "m3"], nonmember_texts=["n1", "n2"])
    run(object(), member_split, {"family": "EZ_MIA", "variant": "default"}, output_cfg={}, reference_model=object())
    assert calls == [3, 2]


# ---------------------------------------------------------------------------
# _compute_ez_scores: sequence_length clamped to the models' actual context window
#
# Long ("many chars") text tokenizes to many tokens; target and reference are compared
# token-by-token at the same positions, so both must be truncated to the *same* effective
# length. Clamping independently per model (one of the first things tried here) would let a
# shorter-context reference model produce shorter tensors than the target, breaking
# _error_zone_pos_neg_sum_ratio's shape assumptions.
# ---------------------------------------------------------------------------

class _FakeLoadedModel:
    def __init__(self, context_limit=None):
        self._context_limit = context_limit

    def max_context_length(self):
        return self._context_limit


def _stub_extract_stats(monkeypatch, calls):
    def _fake_extract_stats(model, texts, batch_size, sequence_length):
        calls.append(sequence_length)
        return [
            {
                "correct": torch.zeros(1), "top1_idx": torch.zeros(1, dtype=torch.long),
                "target_ids": torch.zeros(1, dtype=torch.long), "mask": torch.ones(1),
            }
            for _ in texts
        ]
    monkeypatch.setattr(ez_mia, "_extract_stats", _fake_extract_stats)


def test_compute_ez_scores_clamps_to_the_smaller_of_the_two_models_context_windows(monkeypatch):
    calls = []
    _stub_extract_stats(monkeypatch, calls)

    ez_mia._compute_ez_scores(
        _FakeLoadedModel(context_limit=64), _FakeLoadedModel(context_limit=16),
        ["text a", "text b"], batch_size=2, sequence_length=128,
    )

    # The smaller of the two (16), applied identically to both the target and reference call.
    assert calls == [16, 16]


def test_compute_ez_scores_uses_the_requested_length_when_neither_model_reports_a_limit(monkeypatch):
    calls = []
    _stub_extract_stats(monkeypatch, calls)

    ez_mia._compute_ez_scores(
        _FakeLoadedModel(context_limit=None), _FakeLoadedModel(context_limit=None),
        ["text a"], batch_size=1, sequence_length=128,
    )

    assert calls == [128, 128]
