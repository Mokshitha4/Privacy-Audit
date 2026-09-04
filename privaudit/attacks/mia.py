"""Membership Inference Attack (MIA): Random Forest over per-sequence loss/confidence features.

Member set: sequences from the fine-tuning training split. Non-member set: a disjoint,
never-trained-on partition of the same source corpus (data.role == "member_nonmember_split").

Features per sequence: negative log-likelihood, perplexity, mean ground-truth token
confidence, and Min-k%-Prob. Classifier: RandomForestClassifier(100 trees, max_depth=10),
evaluated with 5-fold cross-validation on a balanced member/non-member set. Defaults match
the paper's methodology (k=20%, 5-fold CV); every number below is a default, not a fixed
constant -- override any of them per job via `attacks[].params`.

TPR@5%FPR is the canonical metric here, not AUC -- AUC is a poor primary MIA metric because
it averages performance across all FPR thresholds, including operating points no real
adversary would use; see Carlini et al., "Membership Inference Attacks From First
Principles" (2022), which argues attacks should be compared at low, fixed FPR instead.

Configurable via attack_cfg["params"]:
    num_members      int|null  cap on how many member sequences to draw (default: all available)
    num_nonmembers    int|null  cap on how many non-member sequences to draw (default: all available)
    max_length        int       tokenizer truncation length per sequence (default: 512)
    k_percent         float     Min-k%-Prob's k (default: 20)
    n_folds           int       cross-validation folds (default: 5)
    batch_size        int       inference batch size (default: 8)
    seed              int       RNG seed for subsampling, CV splitting, and the classifier (default: 42)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ..data.member_split import MemberSplit
from ..metrics import tpr_at_fpr
from ..models.loader import LoadedModel
from .base import AttackOutput

FAMILY = "MIA"
_DEFAULT_K_PERCENT = 20
_DEFAULT_N_FOLDS = 5
_DEFAULT_MAX_LENGTH = 512


def validate(job_cfg: dict, attack_cfg: dict) -> None:
    if job_cfg["data"].get("role") != "member_nonmember_split":
        raise ValueError("MIA requires data.role == 'member_nonmember_split'.")
    if job_cfg["model"].get("access") != "white_box":
        raise ValueError("MIA requires model.access == 'white_box' (needs per-token logits for loss/confidence/Min-k%).")


def _balanced_features(
    model: LoadedModel,
    member_split: MemberSplit,
    *,
    batch_size: int,
    seed: int,
    n_folds: int,
    max_length: int,
    k_percent: float,
    num_members: Optional[int],
    num_nonmembers: Optional[int],
):
    available_members = len(member_split.member_texts)
    available_nonmembers = len(member_split.nonmember_texts)
    requested_members = num_members if num_members is not None else available_members
    requested_nonmembers = num_nonmembers if num_nonmembers is not None else available_nonmembers
    n = min(requested_members, requested_nonmembers, available_members, available_nonmembers)
    if n < n_folds:
        raise ValueError(
            f"MIA needs at least {n_folds} member and {n_folds} non-member sequences for {n_folds}-fold CV; "
            f"got {n} of each after applying num_members/num_nonmembers caps and balancing "
            f"(available: {available_members} members, {available_nonmembers} non-members)."
        )
    rng = np.random.RandomState(seed)
    member_idx = rng.permutation(available_members)[:n]
    nonmember_idx = rng.permutation(available_nonmembers)[:n]
    member_texts = [member_split.member_texts[i] for i in member_idx]
    nonmember_texts = [member_split.nonmember_texts[i] for i in nonmember_idx]

    member_stats = model.sequence_stats(member_texts, batch_size=batch_size, max_length=max_length, k_percent=k_percent)
    nonmember_stats = model.sequence_stats(nonmember_texts, batch_size=batch_size, max_length=max_length, k_percent=k_percent)

    def _matrix(stats):
        return np.column_stack([stats["loss"], stats["perplexity"], stats["confidence"], stats["min_k_prob"]])

    X = np.vstack([_matrix(member_stats), _matrix(nonmember_stats)])
    y = np.array([1] * n + [0] * n)
    return X, y


def run(
    model: LoadedModel,
    member_split: MemberSplit,
    attack_cfg: dict,
    output_cfg: dict,
    reference_model: Optional[LoadedModel] = None,
) -> AttackOutput:
    params = attack_cfg.get("params", {})
    batch_size = int(params.get("batch_size", 8))
    seed = int(params.get("seed", 42))
    n_folds = int(params.get("n_folds", _DEFAULT_N_FOLDS))
    max_length = int(params.get("max_length", _DEFAULT_MAX_LENGTH))
    k_percent = float(params.get("k_percent", _DEFAULT_K_PERCENT))
    num_members = params.get("num_members")
    num_nonmembers = params.get("num_nonmembers")

    X, y = _balanced_features(
        model, member_split,
        batch_size=batch_size, seed=seed, n_folds=n_folds, max_length=max_length, k_percent=k_percent,
        num_members=num_members, num_nonmembers=num_nonmembers,
    )

    classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    proba = cross_val_predict(classifier, X, y, cv=cv, method="predict_proba")[:, 1]

    roc_auc = float(roc_auc_score(y, proba))
    accuracy = float(accuracy_score(y, proba > 0.5))
    tpr_1 = tpr_at_fpr(y, proba, 0.01)
    tpr_5 = tpr_at_fpr(y, proba, 0.05)

    metrics = {
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "tpr_at_1pct_fpr": tpr_1,
        "tpr_at_5pct_fpr": tpr_5,
    }

    return AttackOutput(family=FAMILY, metrics=metrics, canonical_score=tpr_5)
