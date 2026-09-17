"""Model loading: local checkpoint / HF hub, white-box (weights) or black-box (API) access.

Every attack talks to a `LoadedModel`, never to `transformers`/HTTP directly. That keeps
attacks/em.py, attacks/mia.py, attacks/ez_mia.py agnostic to where the model actually came from.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


@dataclass
class ModelConfig:
    source: str  # "huggingface" | "local_checkpoint" | "api_endpoint" | "openai"
    identifier: str
    access: str  # "white_box" | "black_box"
    finetuning: Optional[dict] = None
    base_model: Optional[str] = None  # local_checkpoint + LoRA/PEFT adapter only; see load_model()
    api_key: Optional[str] = None  # source == "openai" only
    base_url: Optional[str] = None  # source == "openai" only; overrides the default OpenAI API base URL

    @classmethod
    def from_job_config(cls, model_cfg: dict) -> "ModelConfig":
        return cls(
            source=model_cfg["source"],
            identifier=model_cfg["identifier"],
            access=model_cfg.get("access", "white_box"),
            finetuning=model_cfg.get("finetuning"),
            base_model=model_cfg.get("base_model"),
            api_key=model_cfg.get("api_key"),
            base_url=model_cfg.get("base_url"),
        )


class LoadedModel:
    """Wraps either a local `transformers` model or a black-box API endpoint."""

    def __init__(self, cfg: ModelConfig, *, model=None, tokenizer=None, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @property
    def is_white_box(self) -> bool:
        return self.cfg.access == "white_box" and self.model is not None

    def require_white_box(self, attack_name: str) -> None:
        if not self.is_white_box:
            raise ValueError(
                f"{attack_name} requires white_box access to model weights (loss/logit features); "
                f"model '{self.cfg.identifier}' was loaded as {self.cfg.access}."
            )

    def generate_greedy(self, prompt: str, max_new_tokens: int) -> str:
        if self.is_white_box:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # greedy decoding == temperature 0, top-p 1
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = output_ids[0][inputs["input_ids"].shape[1]:].tolist()
            return self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        if self.cfg.source == "openai":
            return self._openai_generate(prompt, max_new_tokens)
        return self._api_generate(prompt, max_new_tokens)

    def _api_generate(self, prompt: str, max_new_tokens: int) -> str:
        import urllib.request

        payload = json.dumps({
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.cfg.identifier, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310 - user-provided, trusted endpoint
            body = json.loads(resp.read().decode("utf-8"))
        if "text" not in body:
            raise ValueError(f"API endpoint response missing 'text' field: {body}")
        return body["text"]

    def _openai_generate(self, prompt: str, max_new_tokens: int) -> str:
        """Chat Completions call. No `openai` package dependency -- a plain HTTPS POST, same
        as `_api_generate`, since that's all EM's generation needs."""
        import urllib.error
        import urllib.request

        base_url = (self.cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        payload = json.dumps({
            "model": self.cfg.identifier,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0,
            "top_p": 1,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.cfg.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise ValueError(f"OpenAI API request failed ({e.code}): {detail}") from e
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected OpenAI API response: {body}") from e

    def sequence_stats(self, texts: List[str], batch_size: int = 8, max_length: int = 512, k_percent: int = 20) -> dict:
        """Per-sequence loss, perplexity, mean ground-truth-token confidence, and Min-k%-Prob.

        Requires white-box access (needs per-token logits).
        """
        self.require_white_box("token-level feature extraction (MIA/EZ-MIA)")
        losses, ppls, confidences, min_k_probs = [], [], [], []
        self.model.eval()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                outputs = self.model(**enc, labels=enc["input_ids"])
                logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = enc["input_ids"][..., 1:].contiguous()
            # Mask on attention_mask, not pad-token identity: when pad_token is aliased to
            # eos_token (common when a tokenizer has no dedicated pad token), a real eos in
            # the sequence would otherwise be masked out as if it were padding.
            mask = enc["attention_mask"][..., 1:].contiguous().float()
            denom = torch.clamp(mask.sum(dim=1), min=1.0)

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss_per_token = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
            ).view(shift_labels.size())
            loss_per_sequence = (loss_per_token * mask).sum(dim=1) / denom
            ppl = torch.exp(loss_per_sequence)

            probs = torch.softmax(logits, dim=-1)
            shift_probs = probs[..., :-1, :].contiguous()
            token_probs = torch.gather(shift_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            confidence_per_sequence = (token_probs * mask).sum(dim=1) / denom

            losses.extend(loss_per_sequence.cpu().tolist())
            ppls.extend(ppl.cpu().tolist())
            confidences.extend(confidence_per_sequence.cpu().tolist())

            for seq_idx in range(token_probs.size(0)):
                valid = token_probs[seq_idx][mask[seq_idx] > 0.5]
                if valid.numel() == 0:
                    min_k_probs.append(0.0)
                    continue
                k_count = max(1, int(valid.numel() * k_percent / 100))
                sorted_probs, _ = torch.sort(valid)
                min_k_probs.append(float(sorted_probs[:k_count].mean().item()))

        return {
            "loss": np.array(losses),
            "perplexity": np.array(ppls),
            "confidence": np.array(confidences),
            "min_k_prob": np.array(min_k_probs),
        }


def _try_load_peft_config(identifier: str):
    """Detect a LoRA/PEFT adapter by identifier alone -- works for a local directory *or* an HF
    hub repo id, since PeftConfig.from_pretrained resolves both the same way AutoModel does.
    Returns None (not raises) when `identifier` isn't a PEFT adapter, so callers can treat this
    as a plain yes/no check.
    """
    from peft import PeftConfig

    try:
        return PeftConfig.from_pretrained(identifier)
    except Exception:
        return None


def load_model(model_cfg: dict) -> LoadedModel:
    cfg = ModelConfig.from_job_config(model_cfg)

    if cfg.source == "api_endpoint":
        if cfg.access != "black_box":
            raise ValueError("model.source == 'api_endpoint' must use access == 'black_box'.")
        return LoadedModel(cfg)

    if cfg.source == "openai":
        if cfg.access != "black_box":
            raise ValueError("model.source == 'openai' must use access == 'black_box'.")
        cfg.api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY")
        if not cfg.api_key:
            raise ValueError(
                "model.source == 'openai' requires an API key: set model.api_key in the job config, "
                "or the OPENAI_API_KEY environment variable."
            )
        return LoadedModel(cfg)

    if cfg.source not in ("huggingface", "local_checkpoint"):
        raise ValueError(f"Unknown model.source: {cfg.source!r}")

    if cfg.source == "local_checkpoint" and not Path(cfg.identifier).exists():
        raise FileNotFoundError(f"local_checkpoint path does not exist: {cfg.identifier}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A model.identifier -- whether a local checkpoint directory or an HF hub repo id -- may be
    # either a self-contained model or a LoRA/PEFT adapter (only adapter weights, needing a base
    # model loaded underneath it). Detect that from the identifier alone, uniformly for both.
    peft_config = _try_load_peft_config(cfg.identifier)
    is_adapter = peft_config is not None

    if cfg.base_model and not is_adapter:
        raise ValueError(
            f"model.base_model was set ({cfg.base_model!r}) but {cfg.identifier!r} has no "
            "adapter_config.json, so it doesn't look like a LoRA/PEFT checkpoint. "
            "Drop model.base_model for a self-contained model (it's already complete), "
            "or point model.identifier at the adapter itself."
        )

    if is_adapter:
        # LoRA/PEFT checkpoints only contain adapter weights: load the base model first, then
        # apply the adapter on top of it -- same two-step pattern as the original research
        # scripts (--model / --model_checkpoint_path), now handled for both a local checkpoint
        # dir and an adapter published on the HF hub.
        base_model_name = cfg.base_model or peft_config.base_model_name_or_path
        if not base_model_name:
            raise ValueError(
                f"{cfg.identifier!r} is a LoRA/PEFT adapter but its adapter_config.json has no "
                "'base_model_name_or_path' and model.base_model wasn't set. Set model.base_model "
                "to the base model's HF hub id or local path."
            )
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, cfg.identifier)
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.identifier, padding_side="left")
        except Exception:
            # Adapter-only repos/directories often ship no tokenizer files at all, or an
            # incomplete tokenizer definition (e.g. missing an optional backend dependency
            # needed to convert a slow tokenizer) -- fall back to the base model's tokenizer,
            # which is what the adapter was almost always trained against anyway.
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.identifier, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(cfg.identifier)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    return LoadedModel(cfg, model=model, tokenizer=tokenizer, device=device)
