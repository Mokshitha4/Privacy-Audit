"""Plain-language explanation of a report -- rule-based, entirely local, no LLM/network call.

Written for someone with no privacy-attack background: given the report `run_job()` produces,
explain what was measured, what the numbers mean, and how confident to be in that reading.
Reads only `report["metrics"]` -- never `report.get("raw_generations")` -- so it is safe to call
on a report that has raw generations attached without that data ever entering this module.
"""
from __future__ import annotations

from typing import Dict, List, Optional

# Ordered worse-to-better so verdicts can be compared/aggregated.
_VERDICT_ORDER = ["none", "weak", "strong"]
_VERDICT_LABEL = {"none": "No signal", "weak": "Weak signal", "strong": "Strong signal"}
_VERDICT_BADGE = {"none": "⚪", "weak": "🟡", "strong": "🔴"}

_DISCLAIMER = (
    "This summary is generated locally from the numeric scores above using fixed rules. No data "
    "or scores were sent anywhere to produce it. It is not a legal, clinical, or regulatory "
    "determination, and does not replace a full privacy review. Different attacks measure "
    "different kinds of leakage and can legitimately disagree; that disagreement is itself "
    "informative, not a flaw in the tool."
)


def _band(value: Optional[float], weak_at: float, strong_at: float) -> str:
    """value below weak_at -> 'none', between weak_at and strong_at -> 'weak', above -> 'strong'."""
    if value is None:
        return "none"
    if value >= strong_at:
        return "strong"
    if value >= weak_at:
        return "weak"
    return "none"


def _explain_em(metrics: Dict) -> dict:
    ngram_keys = sorted(
        (k for k in metrics if k.startswith("mem_at_")),
        key=lambda k: int(k.split("_")[-1]),
    )
    top_key = ngram_keys[-1] if ngram_keys else None
    top_n = top_key.split("_")[-1] if top_key else None
    top_rate = metrics.get(top_key, 0.0) if top_key else 0.0

    band = _band(top_rate, weak_at=0.001, strong_at=0.05)

    detail: List[str] = []
    if top_key is not None:
        if top_rate > 0:
            detail.append(
                f"{top_rate:.0%} of tested prompts caused the model to output a verbatim "
                f"{top_n}-token match with the real training continuation. A coincidence this "
                "exact is effectively impossible by chance."
            )
        else:
            detail.append(
                f"None of the tested prompts produced a verbatim {top_n}-token match with the "
                "real training continuation."
            )
    if "rouge_l" in metrics:
        detail.append(f"Average text overlap with the real continuation (ROUGE-L): {metrics['rouge_l']:.2f} of 1.00.")
    if "bertscore_f1" in metrics:
        detail.append(f"Average meaning-level similarity to the real continuation (BERTScore-F1): {metrics['bertscore_f1']:.2f} of 1.00.")

    headline = {
        "none": "No exact memorization detected in the tested samples.",
        "weak": "The model rarely reproduced training text word-for-word.",
        "strong": "The model frequently reproduced training text word-for-word. This is a clear memorization signal.",
    }[band]

    return {
        "family": "EM",
        "name": "Exact Memorization",
        "what_it_checks": "Whether prompting the model with the start of a real training example causes it to output the rest, word for word.",
        "verdict": band,
        "verdict_label": _VERDICT_LABEL[band],
        "badge": _VERDICT_BADGE[band],
        "headline": headline,
        "detail": detail,
    }


def _explain_mia(metrics: Dict) -> dict:
    tpr5 = metrics.get("tpr_at_5pct_fpr")
    tpr1 = metrics.get("tpr_at_1pct_fpr")
    auc = metrics.get("roc_auc")

    # TPR@5%FPR's chance baseline is exactly 5% by definition; band on how far above chance it is.
    excess = (tpr5 - 0.05) if tpr5 is not None else None
    band = _band(excess, weak_at=0.02, strong_at=0.10)

    detail: List[str] = []
    if tpr5 is not None:
        detail.append(
            f"At a 5% false-alarm rate, {tpr5:.0%} of real training examples were correctly flagged as "
            "'in the training set' (chance alone would give about 5%)."
        )
    if tpr1 is not None:
        detail.append(f"At a stricter 1% false-alarm rate: {tpr1:.0%} correctly flagged (chance alone would give about 1%).")
    if auc is not None:
        detail.append(
            f"ROC-AUC: {auc:.2f} (0.50 = indistinguishable from guessing, 1.00 = perfectly distinguishable; "
            "shown for reference only, the 5%-false-alarm figure above is the one to trust)."
        )

    headline = {
        "none": "No stronger than random guessing at telling training data apart from held-out data.",
        "weak": "A modest number of training examples could be identified as such by an attacker.",
        "strong": "Training examples can be reliably picked out from held-out data. This is a clear membership-inference risk.",
    }[band]

    return {
        "family": "MIA",
        "name": "Membership Inference",
        "what_it_checks": "Whether an attacker who can query the model can tell which examples were in its training set versus never seen.",
        "verdict": band,
        "verdict_label": _VERDICT_LABEL[band],
        "badge": _VERDICT_BADGE[band],
        "headline": headline,
        "detail": detail,
    }


def _explain_ez_mia(metrics: Dict) -> dict:
    score = metrics.get("score")  # AUC-based; chance baseline is 0.5
    tpr1 = metrics.get("tpr_at_1pct_fpr")

    distance_from_chance = abs(score - 0.5) if score is not None else None
    band = _band(distance_from_chance, weak_at=0.03, strong_at=0.15)

    detail: List[str] = []
    if score is not None:
        detail.append(f"Separation score: {score:.2f} (0.50 = indistinguishable from a reference model, 1.00 = fully distinguishable).")
    if tpr1 is not None:
        detail.append(f"At a strict 1% false-alarm rate: {tpr1:.0%} of training examples correctly flagged as such.")

    headline = {
        "none": "No meaningful difference from the reference model. No signal detected by this check.",
        "weak": "A modest difference from the reference model was detected.",
        "strong": "A clear difference from the reference model was detected. This is a membership-inference risk.",
    }[band]

    return {
        "family": "EZ_MIA",
        "name": "Lightweight Membership Inference",
        "what_it_checks": "A faster membership-inference check that compares the model's confidence against a reference model's, on the same text.",
        "verdict": band,
        "verdict_label": _VERDICT_LABEL[band],
        "badge": _VERDICT_BADGE[band],
        "headline": headline,
        "detail": detail,
    }


_EXPLAINERS = {"EM": _explain_em, "MIA": _explain_mia, "EZ_MIA": _explain_ez_mia}


def _agreement_note(families: List[dict]) -> Optional[str]:
    if len(families) < 2:
        return None
    verdicts = {f["family"]: f["verdict"] for f in families}
    levels = set(verdicts.values())
    if len(levels) == 1:
        level = next(iter(levels))
        if level == "none":
            return "Every check run here found no signal, a consistent though not conclusive picture of low leakage risk."
        return f"Every check run here agrees: {_VERDICT_LABEL[level].lower()} of leakage."

    strongest = max(verdicts.items(), key=lambda kv: _VERDICT_ORDER.index(kv[1]))
    weakest = min(verdicts.items(), key=lambda kv: _VERDICT_ORDER.index(kv[1]))
    return (
        f"These checks disagree: {strongest[0]} shows {_VERDICT_LABEL[strongest[1]].lower()} while "
        f"{weakest[0]} shows {_VERDICT_LABEL[weakest[1]].lower()}. Don't rely on either one alone. "
        "Running multiple attack families and reading them together, as you just did, is exactly the point."
    )


def summarize(report: dict) -> dict:
    """Build a plain-language explanation from a report's metrics. Pure function: same report in,
    same explanation out; touches nothing outside `report["metrics"]`."""
    metrics = report.get("metrics", {})
    families = [_EXPLAINERS[family](metrics[family]) for family in _EXPLAINERS if family in metrics]
    return {
        "families": families,
        "agreement_note": _agreement_note(families),
        "disclaimer": _DISCLAIMER,
    }


def render_markdown(explanation: dict) -> str:
    """Render an explanation (from `summarize()`) as Markdown for display in the UI or CLI."""
    lines: List[str] = ["## What this run found"]
    for family in explanation["families"]:
        lines.append(f"\n### {family['badge']} {family['name']}: {family['verdict_label']}")
        lines.append(f"*{family['what_it_checks']}*")
        lines.append("")
        lines.append(f"**{family['headline']}**")
        for point in family["detail"]:
            lines.append(f"- {point}")

    if explanation.get("agreement_note"):
        lines.append("\n---")
        lines.append(f"**Reading these together:** {explanation['agreement_note']}")

    lines.append(f"\n---\n*{explanation['disclaimer']}*")
    return "\n".join(lines)
