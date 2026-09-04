# privacy audit

A local-first privacy-leakage audit package for fine-tuned clinical LLMs. It wraps three
attack families behind one consistent interface: point it at a model and a dataset, describe
the attack in a JSON config, get back a JSON report with the metric scores.

No web platform, no hosted execution, no accounts. The report is the only thing that ever
needs to leave your machine, and raw generations are opt-in and off by default.

Built to accompany "Don't Call It Privacy Until You Pick a Metric" — the point of running
several attack families side by side is that they disagree with each other, so no single
metric should be trusted alone.

## Attack families

| Family     | What it measures | Needs |
|------------|-------------------|-------|
| `EM`       | Exact Memorization — prompts the model with a prefix from a training sequence and checks whether it regurgitates the held-out continuation (n-gram match, ROUGE-L, BLEU, BERTScore-F1). | Just the training (member) split. Works on any Q&A / instruction-formatted text. |
| `MIA`      | Membership Inference — a Random Forest over per-sequence loss/perplexity/confidence/Min-k%-Prob features, evaluated with 5-fold CV. Canonical metric is TPR@5%FPR, not AUC (AUC averages over FPR thresholds no real adversary would use). | A member/non-member split, white-box model access. |
| `EZ_MIA`   | A lighter-weight MIA variant (arXiv:2601.12104) that compares a target and a reference model's confidence in "error-zone" tokens. Kept separate from `MIA` because the two attacks behave differently in practice (EZ-MIA tracks EM closely on full-loss checkpoints, near-chance on masked-loss; the Random Forest MIA doesn't). | A member/non-member split, white-box access to **both** a target and a reference model. |

**Out of scope for this build:** PA-style attacks (undefined in the source research code this
package was built from), and RAG-specific attacks.

## Install

```bash
pip install -e .
# BERTScore-F1 (used by EM) needs an extra, network-downloaded embedding model:
pip install -e ".[bertscore]"
# the optional local UI (see below):
pip install -e ".[ui]"
# for running the test suite:
pip install -e ".[dev]"
```

Requires Python >= 3.10.

## Quick start

```bash
privaudit validate --config job.json     # schema-check a config without running anything
privaudit run --config job.json --output report.json
```

## UI (optional)

```bash
pip install -e ".[ui]"
privaudit ui                 # opens on http://127.0.0.1:7860 by default
privaudit ui --port 8080
```

A local Gradio page wrapping the exact same JSON-in/JSON-out contract as the CLI: a text box
for the job config (pre-filled with an example), a **Validate** button, and a **Run** button
that writes the report JSON back into the page. It adds no capability beyond `privaudit
run`/`validate` — it's a friendlier way to edit and run a config without touching files.

Runs on `127.0.0.1` only (`share=False`, always) — nothing here is hosted or tunneled out;
the page and everything it does stay on your machine, same as the CLI.

## Job config

```json
{
  "model": {
    "source": "local_checkpoint",
    "identifier": "/local/path/to/your/lora_checkpoint",
    "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "access": "white_box",
    "finetuning": { "regime": "qlora", "loss": "masked", "epsilon": null }
  },
  "reference_model": {
    "source": "huggingface",
    "identifier": "meta-llama/Meta-Llama-3-8B-Instruct",
    "access": "white_box"
  },
  "data": {
    "role": "member_nonmember_split",
    "known_dataset": "custom",
    "schema": {
      "format": "csv",
      "member_file": "train.csv",
      "nonmember_file": "test.csv",
      "text_field": "note"
    },
    "path": "/local/path/to/your/data"
  },
  "attacks": [
    { "family": "EM", "variant": "default", "params": { "continuation_len": 300 } },
    { "family": "MIA", "variant": "default", "params": { "num_members": 500, "num_nonmembers": 500 } }
  ],
  "output": { "return_raw_generations": false }
}
```

- **`model`** — `source` is `huggingface` (hub identifier), `local_checkpoint` (a local
  directory), or `api_endpoint` (`access` must be `black_box`; only `EM` can run against a
  black-box model, since `MIA`/`EZ_MIA` need per-token logits).
  - A **full fine-tune / DP-SGD checkpoint** — local or on the hub — is self-contained and
    loads directly from `identifier`.
  - A **LoRA/PEFT adapter** only contains adapter weights, so the base model is loaded first
    and the adapter applied on top of it. This is detected from `identifier` alone and works
    identically whether `identifier` is a local checkpoint directory or an adapter repo
    published on the HF hub — no separate config needed to say which. The base model is
    normally auto-detected from the adapter's own `adapter_config.json`
    (`base_model_name_or_path`, written automatically when the adapter was saved); set
    `model.base_model` (an HF hub id or local path) to override that, or to supply it
    explicitly if the adapter config doesn't have it (e.g. a hand-edited or partially-copied
    checkpoint). If the adapter itself ships no tokenizer (common for adapter-only repos),
    the base model's tokenizer is used instead. Setting `base_model` when `identifier` isn't
    actually a PEFT adapter is a fail-fast config error.
- **`reference_model`** — same shape as `model`. Only required when `EZ_MIA` is one of the
  requested attacks.
- **`data.path`** is a local directory. Nothing under it is ever uploaded.
- **`data.known_dataset`** — one of `MedQA`, `ICD`, `mortality`, `readmission` (file names and
  field mappings are built in, matching the original research scripts), or `custom` with a
  `data.schema` block (see below). Every attack here uses `data.role: "member_nonmember_split"`.
- Each attack `family` (`EM`, `MIA`, `EZ_MIA`) may appear **at most once** per job.
- `privaudit validate` / `privaudit run` fail fast with a specific error message — e.g. `MIA`
  without `model.access: "white_box"`, or `EZ_MIA` without a top-level `reference_model`.

### Custom datasets (`data.schema`)

```json
{
  "format": "csv",
  "member_file": "train.csv",
  "nonmember_file": "test.csv",
  "text_field": "note"
}
```

Use `text_template` instead of `text_field` to combine several columns into one training-format
string, e.g. `"text_template": "Q: {question}\nA: {answer}"`. `format` is `csv` or `jsonl`.

### Attack params

Every attack accepts an optional `params` object; all fields below have defaults, so an empty
`params` (or an omitted one) reproduces the original paper's methodology.

**`EM`**

| param | default | meaning |
|---|---|---|
| `prefix_len` | 50 | prefix length in tokens |
| `continuation_len` | 500 | tokens to generate, and to hold out as groundtruth |
| `max_samples` | all eligible | cap on how many prefix/continuation pairs to build |
| `ngram_ns` | `[10, 20, 30, 50]` | n-gram window sizes for the memorization-event indicator |

**`MIA`**

| param | default | meaning |
|---|---|---|
| `num_members` / `num_nonmembers` | all available | cap on how many sequences of each class to draw |
| `max_length` | 512 | tokenizer truncation length per sequence |
| `k_percent` | 20 | Min-k%-Prob's k |
| `n_folds` | 5 | cross-validation folds |
| `batch_size` | 8 | inference batch size |
| `seed` | 42 | RNG seed for subsampling, CV splitting, and the classifier |

**`EZ_MIA`**

| param | default | meaning |
|---|---|---|
| `num_members` / `num_nonmembers` | all available | cap on how many sequences of each class to score |
| `sequence_length` | 128 | tokenizer padding/truncation length per sequence |
| `batch_size` | 8 | inference batch size |
| `seed` | 42 | RNG seed for subsampling when caps are applied |

## Report

```json
{
  "job_id": "…",
  "model_id": "your-org/your-finetuned-model",
  "attack_code_version": "0.1.0",
  "generated_at": "2026-09-04T12:00:00+00:00",
  "metrics": {
    "EM": { "mem_at_10": 0.0, "mem_at_20": 0.0, "mem_at_30": 0.0, "mem_at_50": 0.0, "rouge_l": 0.0, "bleu": 0.0, "bertscore_f1": 0.0 },
    "MIA": { "roc_auc": 0.0, "accuracy": 0.0, "tpr_at_1pct_fpr": 0.0, "tpr_at_5pct_fpr": 0.0 }
  },
  "canonical_scores": { "EM": 0.0, "MIA": 0.0 }
}
```

`canonical_scores` picks one number per family for at-a-glance comparison: `mem_at_{max(ngram_ns)}`
for EM, `tpr_at_5pct_fpr` for MIA, AUC for EZ_MIA. `raw_generations` is only present when
`output.return_raw_generations` is `true`.

## Package layout

```
privaudit/
  attacks/em.py, mia.py, ez_mia.py   # one module per attack family, common run()/validate() contract
  models/loader.py                   # HF hub / local checkpoint / API endpoint, white- or black-box
  data/schema.py, member_split.py    # known + custom dataset schemas, member/non-member loading
  metrics.py                         # dependency-light ROUGE-L, BLEU, TPR@FPR, n-gram matching
  report.py                          # JSON Schema validation (job config + report) and report building
  runner.py                          # orchestrates one job end to end
  cli.py                             # `privaudit run` / `privaudit validate` / `privaudit ui`
  ui.py                              # optional local Gradio front end (extra: "ui")
```

## Testing

```bash
pytest tests/ -q
```

Tests run against fake models (no GPU, no network, no real checkpoints) and are structured as:
- exact algorithmic correctness (ROUGE-L, BLEU, TPR@FPR, the EZ-MIA scoring kernel)
- one exact reproduction of a published result: `EZ-MIA/results.csv`'s `icd_mimic` row
  (AUC=0.5, TPR@1%FPR=0.010, TPR@0.1%FPR=0.001) is a degenerate, analytically-reproducible case
- per-attack pipeline sanity checks against fake models with synthetic, clearly-separable data
  (`test_em.py`, `test_mia.py`, `test_ez_mia.py`)
- a full end-to-end run (`test_integration.py`): one job config with all three attack families,
  real temp-directory CSV data, and a mocked model, driven through the actual
  `runner.run_job()` orchestration and the real `privaudit run`/`validate` CLI entry points —
  not just each attack module called in isolation
- job-config schema validation and fail-fast error messages

Full reproduction of the paper's non-degenerate numbers needs the actual fine-tuned checkpoints
and a GPU, neither of which are available in this repo.
