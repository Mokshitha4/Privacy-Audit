"""Metric correctness tests, including an exact reproduction of a published result.

EZ-MIA/results.csv (from the paper's own experiment logs) reports, for icd_mimic on three
different target models (llama3-instruct, qwen3-4b, qwen3-0.6b), the identical triple
AUC=0.500000, TPR@1%FPR=0.010, TPR@0.1%FPR=0.001. That is only possible if every scored
example received the exact same EZ score (the `negative_sum == 0` fallback in
error_zone_pos_neg_sum_ratio returns 0.0 for every sequence whenever no token in the
error zone favors the target model) -- in that degenerate case, `roc_curve` collapses to
the diagonal and `tpr_at_fpr(target_fpr) == target_fpr` and `auc == 0.5` exactly,
independent of sample count. `test_tpr_at_fpr_reproduces_paper_icd_mimic_row` reproduces
that published row exactly using the same `metrics.tpr_at_fpr` / `roc_auc_score` call path
that `attacks/ez_mia.py` uses in production.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from privaudit.metrics import bleu, has_consecutive_ngram_match, memorization_events, rouge_l, tpr_at_fpr


def test_tpr_at_fpr_reproduces_paper_icd_mimic_row():
    # EZ-MIA/results.csv, icd_mimic rows (llama3-instruct / qwen3-4b / qwen3-0.6b, full_loss, seed=42):
    # AUC=0.500000, TPR@1%FPR=0.010, TPR@0.1%FPR=0.001
    n = 500  # arbitrary balanced size; the degenerate (constant-score) result is size-independent
    y = np.array([1] * n + [0] * n)
    scores = np.zeros(2 * n)  # every example fell back to the 0.0 score (no error-zone advantage)

    auc = roc_auc_score(y, scores)
    tpr_at_1pct = tpr_at_fpr(y, scores, 0.01)
    tpr_at_0_1pct = tpr_at_fpr(y, scores, 0.001)

    assert auc == pytest.approx(0.500000, abs=1e-6)
    assert tpr_at_1pct == pytest.approx(0.010, abs=1e-6)
    assert tpr_at_0_1pct == pytest.approx(0.001, abs=1e-6)


def test_tpr_at_fpr_perfect_separation():
    y = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert tpr_at_fpr(y, scores, 0.0) == pytest.approx(1.0)


def test_has_consecutive_ngram_match():
    gt = "the patient was prescribed metformin twice daily".split()
    gen_hit = "we noted the patient was prescribed metformin twice daily for glucose control".split()
    gen_miss = "the patient received a completely different medication regimen entirely".split()
    assert has_consecutive_ngram_match(gt, gen_hit, n=7) is True
    assert has_consecutive_ngram_match(gt, gen_miss, n=7) is False


def test_memorization_events_multiple_ns():
    gt = list(range(100))  # tokens as ints, function only needs equality + hashability
    gt = [str(x) for x in gt]
    gen = gt[:30] + ["X"] * 70  # verbatim match on the first 30 tokens only
    events = memorization_events(gt, gen, ns=(10, 20, 30, 50))
    assert events == {10: 1, 20: 1, 30: 1, 50: 0}


def test_rouge_l_identical_sequences_is_one():
    tokens = "the patient was prescribed metformin".split()
    assert rouge_l(tokens, tokens) == pytest.approx(1.0)


def test_rouge_l_disjoint_sequences_is_zero():
    assert rouge_l("a b c".split(), "d e f".split()) == 0.0


def test_bleu_identical_sequences_is_one():
    tokens = "the patient was prescribed metformin twice daily".split()
    assert bleu(tokens, tokens) == pytest.approx(1.0, abs=1e-6)


def test_bleu_disjoint_sequences_is_zero():
    ref = "the patient was prescribed metformin".split()
    cand = "totally unrelated words appearing here".split()
    assert bleu(ref, cand) == 0.0


def test_bleu_brevity_penalty_shortens_score():
    ref = "the patient was prescribed metformin twice daily for glucose control".split()
    short_cand = "the patient was prescribed metformin".split()  # exact prefix, but much shorter
    full_cand = ref
    assert bleu(ref, short_cand) < bleu(ref, full_cand)
