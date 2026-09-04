"""Dependency-light metric implementations shared across attack families.

ROUGE-L and BLEU are implemented from scratch (no `rouge_score` / `sacrebleu` /
`nltk` dependency) so the package's dependency surface stays small and the
unit tests are hermetic (no network access, no extra downloads). Both match
the standard definitions used by the HF `evaluate` metrics of the same name.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_curve


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Interpolated TPR at a given FPR on the ROC curve."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    fpr_values, tpr_values, _ = roc_curve(labels.astype(int), scores.astype(float), pos_label=1)
    if fpr_values.size == 0:
        return float("nan")
    target_fpr = float(np.clip(target_fpr, 0.0, 1.0))
    return float(np.interp(target_fpr, np.clip(fpr_values, 0.0, 1.0), np.clip(tpr_values, 0.0, 1.0)))


def has_consecutive_ngram_match(gt_tokens: Sequence[str], gen_tokens: Sequence[str], n: int) -> bool:
    """Whether `gen_tokens` contains a contiguous n-gram span found in `gt_tokens` (verbatim overlap)."""
    if len(gt_tokens) < n or len(gen_tokens) < n:
        return False
    gen_ngrams = {tuple(gen_tokens[j:j + n]) for j in range(len(gen_tokens) - n + 1)}
    for i in range(len(gt_tokens) - n + 1):
        if tuple(gt_tokens[i:i + n]) in gen_ngrams:
            return True
    return False


def memorization_events(gt_tokens: Sequence[str], gen_tokens: Sequence[str], ns: Iterable[int] = (10, 20, 30, 50)) -> dict:
    """Memorization-event indicator (0/1) for each n in `ns`."""
    return {n: int(has_consecutive_ngram_match(gt_tokens, gen_tokens, n)) for n in ns}


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = curr[j - 1] if curr[j - 1] >= prev[j] else prev[j]
        prev = curr
    return prev[m]


def rouge_l(reference_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> float:
    """ROUGE-L F1 (harmonic mean of LCS-based precision/recall), matching HF `evaluate.load("rouge")`."""
    if not reference_tokens or not candidate_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, candidate_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(reference_tokens: Sequence[str], candidate_tokens: Sequence[str], max_n: int = 4) -> float:
    """Sentence-level smoothed BLEU (Chen & Cherry 2014, method 1: add-one to zero n-gram counts).

    Weights are uniform across n in [1, max_n], matching the "smoothed BLEU up to 4-grams" spec.
    """
    if not candidate_tokens:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = _ngram_counts(candidate_tokens, n)
        ref_ngrams = _ngram_counts(reference_tokens, n)
        overlap = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in cand_ngrams.items())
        total = max(sum(cand_ngrams.values()), 0)
        if total == 0:
            precisions.append(0.0)
            continue
        if overlap == 0:
            if n == 1:
                # No shared unigrams at all: a genuinely zero-overlap sentence, not a
                # smoothing artifact -- BLEU is 0 without invoking smoothing.
                return 0.0
            # Add-one smoothing (Chen & Cherry 2014, method 1) so a single missing
            # higher-order n-gram doesn't zero out an otherwise-overlapping sentence.
            precisions.append(1.0 / (2 * total))
        else:
            precisions.append(overlap / total)
    if any(p <= 0 for p in precisions):
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    geo_mean = math.exp(log_avg)

    cand_len = len(candidate_tokens)
    ref_len = len(reference_tokens)
    if cand_len > ref_len:
        brevity_penalty = 1.0
    elif cand_len == 0:
        brevity_penalty = 0.0
    else:
        brevity_penalty = math.exp(1 - ref_len / cand_len)
    return brevity_penalty * geo_mean
