"""EZ-MIA: a lighter-weight membership inference variant (arXiv:2601.12104).

Kept as a distinct attack from the Random Forest MIA (mia.py) rather than merged into it --
the two behave differently in practice: EZ-MIA correlates strongly with Exact Memorization on
full-loss checkpoints and drops to near-chance on masked-loss checkpoints, while the Random
Forest MIA is closer to independent of EM. Collapsing them into one variant would hide that
disagreement, which is the whole point of running multiple attack families side by side.

Score: for each sequence, compare the target model's and a reference model's log-probability
on the ground-truth next token, restricted to "error-zone" tokens where the target model's
top-1 prediction is wrong. The score is the ratio of the summed positive deltas (reference
better than target) to the summed negative deltas (target better than reference) in that
zone. Needs a reference model -- pass one via the job config's top-level `reference_model`
block (same shape as `model`).

Configurable via attack_cfg["params"]:
    num_members       int|null  cap on how many member sequences to score (default: all available)
    num_nonmembers    int|null  cap on how many non-member sequences to score (default: all available)
    sequence_length    int       tokenizer padding/truncation length per sequence (default: 128)
    batch_size         int       inference batch size (default: 8)
    seed               int       RNG seed for subsampling when caps are applied (default: 42)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ..data.member_split import MemberSplit
from ..metrics import tpr_at_fpr
from ..models.loader import LoadedModel
from .base import AttackOutput

FAMILY = "EZ_MIA"
_DEFAULT_SEQUENCE_LENGTH = 128


def validate(job_cfg: dict, attack_cfg: dict) -> None:
    if job_cfg["data"].get("role") != "member_nonmember_split":
        raise ValueError("EZ_MIA requires data.role == 'member_nonmember_split'.")
    if job_cfg["model"].get("access") != "white_box":
        raise ValueError("EZ_MIA requires model.access == 'white_box' (needs per-token logits).")
    reference_model_cfg = job_cfg.get("reference_model")
    if not reference_model_cfg:
        raise ValueError("EZ_MIA requires a top-level 'reference_model' block (same shape as 'model') in the job config.")
    if reference_model_cfg.get("access") != "white_box":
        raise ValueError("EZ_MIA's reference_model must also use access == 'white_box'.")


def _apply_mask_1d(values: torch.Tensor, mask: torch.Tensor, ignore_bos: bool):
    values_flat = values.reshape(-1) if values.ndim >= 2 else values
    mask_flat = mask.reshape(-1).to(torch.float32) if mask.ndim == 2 else mask.to(torch.float32)
    if ignore_bos and mask_flat.numel() > 0:
        mask_flat = mask_flat.clone()
        mask_flat[0] = 0.0
    return values_flat, mask_flat


def _error_zone_pos_neg_sum_ratio(
    tgt_correct_log_probs: torch.Tensor,
    ref_correct_log_probs: torch.Tensor,
    top1_indices_tgt: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    ignore_bos: bool = True,
    min_tokens: int = 2,
) -> float:
    """Positive-vs-negative log-prob delta ratio, restricted to tokens the target got wrong."""
    target_correct_lp, mask_flat = _apply_mask_1d(tgt_correct_log_probs, mask, ignore_bos)
    ref_correct_lp, _ = _apply_mask_1d(ref_correct_log_probs, mask, ignore_bos)
    top1_indices_flat = top1_indices_tgt.reshape(-1)
    targets_flat = target_ids.reshape(-1)
    base_mask = mask_flat > 0.5
    error_mask = (top1_indices_flat != targets_flat) & base_mask
    if error_mask.sum().item() < min_tokens:
        return 0.0
    deltas = (target_correct_lp - ref_correct_lp)[error_mask]
    positive_sum = deltas[deltas > 0].sum()
    negative_sum = deltas[deltas < 0].abs().sum()
    if negative_sum.item() == 0:
        return 0.0
    value = float((positive_sum / (negative_sum + 1e-12)).item())
    if value != value or value < 0:  # NaN guard
        return 0.0
    return value


def _extract_stats(model: LoadedModel, texts: List[str], batch_size: int, sequence_length: int) -> List[dict]:
    stats_out = []
    model.model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = model.tokenizer(
                batch_texts, padding="max_length", truncation=True, max_length=sequence_length, return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(model.device)
            attention_mask = enc["attention_mask"].to(model.device)

            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            pred_logits = logits[:, :-1, :]
            target_ids = input_ids[:, 1:]
            mask = attention_mask[:, 1:]

            log_probs = F.log_softmax(pred_logits, dim=-1)
            correct = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            _, top1_idx = torch.topk(log_probs, k=1, dim=-1)
            top1_idx = top1_idx.squeeze(-1)

            for b in range(input_ids.shape[0]):
                stats_out.append({
                    "correct": correct[b].cpu(),
                    "top1_idx": top1_idx[b].cpu(),
                    "target_ids": target_ids[b].cpu(),
                    "mask": mask[b].cpu(),
                })
    return stats_out


def _compute_ez_scores(target: LoadedModel, reference: LoadedModel, texts: List[str], batch_size: int, sequence_length: int) -> List[float]:
    # Target and reference are compared token-by-token at the same positions, so both must be
    # padded/truncated to the *same* effective length -- clamp to the smaller of the two
    # models' context windows (if either is known), not independently per model, or long
    # ("many chars") text could tokenize the two models' batches to different shapes.
    for loaded in (target, reference):
        context_limit = loaded.max_context_length()
        if context_limit is not None:
            sequence_length = min(sequence_length, context_limit)

    t_stats = _extract_stats(target, texts, batch_size, sequence_length)
    r_stats = _extract_stats(reference, texts, batch_size, sequence_length)
    scores = []
    for t_s, r_s in zip(t_stats, r_stats):
        score = _error_zone_pos_neg_sum_ratio(
            t_s["correct"], r_s["correct"], t_s["top1_idx"], t_s["target_ids"], t_s["mask"],
            ignore_bos=True, min_tokens=2,
        )
        scores.append(score)
    return scores


def run(
    model: LoadedModel,
    member_split: MemberSplit,
    attack_cfg: dict,
    output_cfg: dict,
    reference_model: Optional[LoadedModel] = None,
) -> AttackOutput:
    if reference_model is None:
        raise ValueError("EZ_MIA.run requires a reference_model (loaded from job config's top-level 'reference_model').")

    params = attack_cfg.get("params", {})
    batch_size = int(params.get("batch_size", 8))
    sequence_length = int(params.get("sequence_length", _DEFAULT_SEQUENCE_LENGTH))
    seed = int(params.get("seed", 42))
    num_members = params.get("num_members")
    num_nonmembers = params.get("num_nonmembers")

    rng = np.random.RandomState(seed)
    member_texts = member_split.member_texts
    nonmember_texts = member_split.nonmember_texts
    if num_members is not None and num_members < len(member_texts):
        member_texts = [member_texts[i] for i in rng.permutation(len(member_texts))[:num_members]]
    if num_nonmembers is not None and num_nonmembers < len(nonmember_texts):
        nonmember_texts = [nonmember_texts[i] for i in rng.permutation(len(nonmember_texts))[:num_nonmembers]]

    scores_m = _compute_ez_scores(model, reference_model, member_texts, batch_size, sequence_length)
    scores_nm = _compute_ez_scores(model, reference_model, nonmember_texts, batch_size, sequence_length)

    y = np.array([1] * len(scores_m) + [0] * len(scores_nm), dtype=np.int64)
    scores = np.nan_to_num(np.array(scores_m + scores_nm, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    auc = float(roc_auc_score(y, scores))
    tpr_1 = tpr_at_fpr(y, scores, 0.01)
    tpr_point1 = tpr_at_fpr(y, scores, 0.001)

    metrics = {
        "score": auc,
        "tpr_at_1pct_fpr": tpr_1,
        "tpr_at_0_1pct_fpr": tpr_point1,
    }

    return AttackOutput(family=FAMILY, metrics=metrics, canonical_score=auc)
