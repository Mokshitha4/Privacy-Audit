"""Optional LLM narration layer on top of the rule-based explanation (privaudit.explain).

Opt-in only, and additive: this never replaces `explain.render_markdown()`'s rule-based
summary, it narrates it in friendlier prose on top. The only thing sent anywhere is the output
of `explain.summarize()` -- verdict labels, rounded metric values, and template sentences --
never `raw_generations`, never the raw report, never your data. Requires your own API key
(Anthropic or an OpenAI-compatible endpoint); this package holds no key of its own and makes no
network call unless you explicitly ask for one.

No SDK dependency: both providers are called directly over HTTPS with `urllib`, the same
minimal-dependency approach `models/loader.py` uses for its `openai` model source.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

from .explain import summarize

_SYSTEM_PROMPT = (
    "You are explaining the results of a local, automated privacy audit on a fine-tuned "
    "medical language model to someone with no background in privacy attacks or machine "
    "learning. You are given a JSON object describing what several independent attack checks "
    "found -- each has already been scored and labeled by fixed rules, so your job is not to "
    "re-score anything, only to narrate it clearly. Write a short, warm, jargon-free "
    "explanation: what was checked, what was found, how concerned they should be, and what a "
    "sensible next step is. Do not invent numbers, verdicts, or checks beyond what's given in "
    "the JSON. Do not claim certainty the data doesn't support. Keep it to 3-5 short "
    "paragraphs of plain text -- no markdown headers, no bullet lists."
)

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
}

_ENV_VAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


@dataclass
class LLMExplainConfig:
    provider: str  # "anthropic" | "openai"
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None


def _call_anthropic(api_key: str, model: str, payload_text: str, base_url: Optional[str] = None) -> str:
    base = (base_url or "https://api.anthropic.com/v1").rstrip("/")
    body = json.dumps({
        "model": model,
        "max_tokens": 700,
        "system": _SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": payload_text}],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/messages", data=body,
        headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - user-provided key, user's own choice of provider
        result = json.loads(resp.read().decode("utf-8"))
    try:
        return "".join(block.get("text", "") for block in result["content"]).strip()
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"Unexpected Anthropic API response shape: {result}") from e


def _call_openai(api_key: str, model: str, payload_text: str, base_url: Optional[str] = None) -> str:
    base = (base_url or "https://api.openai.com/v1").rstrip("/")
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": payload_text},
        ],
        "max_tokens": 700,
        "temperature": 0.3,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - user-provided key, user's own choice of provider
        result = json.loads(resp.read().decode("utf-8"))
    try:
        return result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected OpenAI API response shape: {result}") from e


_CALLERS = {"anthropic": _call_anthropic, "openai": _call_openai}


def explain_with_llm(report: dict, config: LLMExplainConfig) -> str:
    """Narrate `explain.summarize(report)` with an LLM. Raises ValueError for a missing
    key/unknown provider, or whatever the underlying HTTP call raises on failure -- callers
    should catch broadly and fall back to `explain.render_markdown()` alone, since this step is
    additive and optional by design."""
    if config.provider not in _CALLERS:
        raise ValueError(f"Unknown LLM provider {config.provider!r}; expected one of {sorted(_CALLERS)}.")

    api_key = config.api_key or os.environ.get(_ENV_VAR[config.provider])
    if not api_key:
        raise ValueError(
            f"No API key given for provider={config.provider!r}; set it explicitly or via the "
            f"{_ENV_VAR[config.provider]} environment variable."
        )

    model = config.model or _DEFAULT_MODELS[config.provider]
    payload_text = json.dumps(summarize(report))  # never includes raw_generations; see explain.py
    return _CALLERS[config.provider](api_key, model, payload_text, config.base_url)
