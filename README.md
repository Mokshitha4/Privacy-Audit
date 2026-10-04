# Privacy audit

A local-first privacy-leakage audit package for fine-tuned clinical LLMs, usable as a command-line tool, a local web UI, or a plain Python package, all three run the exact same code underneath. Point it at a model and a dataset, describe which attacks to run, and get back a report with the scores.

It wraps three attack families behind one consistent interface: Exact Memorization, Membership Inference, and EZ-MIA. Running them side by side is the point, these metrics do not correlate with each other depending on task and fine-tuning setup, so no single one should be trusted alone. That finding comes from "Joint Auditing of Fine-Tuning and Privacy Metrics in Clinical LLMs," the paper this package was built to accompany.

No web platform, no hosted execution, no accounts. The report is the only thing that ever leaves your machine, and even that only if you choose to share it, raw model generations are opt-in and off by default.

## Attack families

| Family     | What it measures | Needs |
|------------|-------------------|-------|
| `EM`       | Exact Memorization: prompts the model with a prefix from a training sequence and checks whether it regurgitates the held-out continuation (n-gram match, ROUGE-L, BLEU, BERTScore-F1). | Just the training (member) split. Works on any Q&A / instruction-formatted text. |
| `MIA`      | Membership Inference: a Random Forest over per-sequence loss/perplexity/confidence/Min-k%-Prob features, evaluated with 5-fold CV.  | A member/non-member split, white-box model access. |
| `EZ_MIA`   | A lighter-weight MIA variant (arXiv:2601.12104) that compares a target and a reference model's confidence in "error-zone" tokens. Kept separate from `MIA` because the two attacks behave differently in practice. | A member/non-member split, white-box access to **both** a target and a reference model. |


## Install

```bash
pip install -e .
# BERTScore-F1 (used by EM) needs an extra, network-downloaded embedding model:
pip install -e ".[bertscore]"
# the local web UI (see below):
pip install -e ".[webui]"
# the legacy Gradio UI:
pip install -e ".[ui]"
# for running the test suite:
pip install -e ".[dev]"
```

Requires Python >= 3.10.

## Quick start (CLI)

Everything in this package runs end to end from the command line; the UI below is an optional
form wrapped around these exact same commands, not a separate code path. The whole workflow is:
write a job config once (see [Job config](#job-config) below), then run it.

```bash
# 1. Write your job config to job.json (model + data + which attacks to run; see Job config below)

# 2. Schema-check it before touching a model or dataset -- catches a typo'd field name,
#    a missing reference_model, etc. in milliseconds instead of after a model finishes loading.
privaudit validate --config job.json

# 3. Run it. Prints progress (loading the model, loading the dataset, each attack) as it goes,
#    then writes the full JSON report.
privaudit run --config job.json --output report.json

# 4. Turn the numbers into a plain-language summary -- rule-based, local, no network call.
privaudit explain --report report.json
```

## Use as a library

A third option alongside the CLI and the UI: call the same functions directly from Python,
e.g. from a notebook or your own pipeline. These are exactly what `cli.py` itself calls, not a
separate wrapper API, so there's nothing the CLI can do that the library can't.

```python
from privaudit.report import validate_job_config
from privaudit.runner import run_job
from privaudit.explain import summarize, render_markdown

job_cfg = {
    "model": {"source": "huggingface", "identifier": "your-org/your-finetuned-model", "access": "white_box"},
    "data": {
        "role": "member_nonmember_split",
        "schema": {"format": "csv", "member_file": "train.csv", "nonmember_file": "test.csv", "text_field": "note"},
        "path": "/local/path/to/your/data",
    },
    "attacks": [{"family": "EM", "variant": "default"}],
}

validate_job_config(job_cfg)  # raises ValueError with a specific message on a bad config

report = run_job(job_cfg, on_progress=print)  # on_progress is optional; called with a short
                                               # status string at each stage (loading the model,
                                               # loading the dataset, each attack)

print(render_markdown(summarize(report)))  # the same plain-language summary `explain` prints
```

Optional LLM narration : `privaudit.llm_explain.explain_with_llm(report,
LLMExplainConfig(provider="anthropic", ...))`, same opt-in, bring-your-own-key behavior as
`--llm` below.

## UI 

```bash
pip install -e ".[webui]"
privaudit ui                 # opens http://127.0.0.1:8765 in your browser
privaudit ui --port 8080
privaudit ui --no-browser    # just start the server, don't open a tab
```

A local web UI wrapping the exact same JSON-in/JSON-out contract as the CLI: a small FastAPI
backend (`privaudit/webapp/`) serving a hand-built, dependency-free HTML/CSS/JS frontend, no
Node toolchain and nothing fetched from a CDN at runtime (even the serif typeface is bundled
under `webapp/static/fonts/`, so the page works exactly the same offline). It's a **form**, not
a raw JSON box: dropdowns for the enum fields (`source`, `access`, `role`, `format`, ...), text
inputs for the strings, and every attack parameter shown pre-filled with its default
(`prefix_len` 50, `continuation_len` 500, MIA `n_folds` 5, ...) so you edit only what you need.
Each attack's parameters appear only when that attack is enabled; the `api_key`/`base_url`
fields appear only when `source` is `openai`. The reference-model fields live inside the EZ_MIA
section (not a separate step) and appear only when EZ_MIA is enabled, since it's the only attack
that ever needs one.

- **Preview config** shows the assembled JSON (copy it straight into `privaudit run --config`).
- **Validate** runs the same schema check as `privaudit validate`.
- **Run audit** streams live progress (loading the model, loading the dataset, each attack) into
  a status log over a WebSocket, then shows a plain-language explanation of the results front
  and center, with each attack family's verdict rendered as a stamped label (clear / weak /
  leak detected) rather than a raw score. The full report JSON stays available in a collapsed
  "Technical details" section.

It adds no capability beyond `privaudit run` / `validate` / `explain`; the server only ever
binds to `127.0.0.1`, never tunneled or hosted elsewhere. A "local run" stamp in the header is a
standing reminder of that, not just copy: nothing about a run leaves your machine unless you
explicitly opt into AI narration below. Light and dark themes are both supported (toggle in the
header); the design intentionally reads as a case file rather than a dashboard.

The previous Gradio-based UI is still available for anyone who prefers it:

```bash
pip install -e ".[ui]"
privaudit ui-gradio
```

## Explaining a report

Reports are numbers (`mem_at_50: 0.02`, `tpr_at_5pct_fpr: 0.05`, ...), useful for a reviewer,
opaque to anyone else. `privaudit explain` turns a report into plain language:

```bash
privaudit explain --report report.json
```

```
## What this run found

### ⚪ Exact Memorization: No signal
*Whether prompting the model with the start of a real training example causes it to output the rest, word for word.*

**No exact memorization detected in the tested samples.**
- None of the tested prompts produced a verbatim 50-token match with the real training continuation.
...

**Reading these together:** Every check run here found no signal, a consistent though not conclusive picture of low leakage risk.
```

This is entirely **rule-based and local** (`privaudit/explain.py`): fixed thresholds turn each
attack's numbers into a "no / weak / strong evidence" verdict with a one-line reason, and a
closing note when attacks disagree with each other (the paper's central point: no single metric
should be trusted alone). No network call, nothing sent anywhere, works offline. It reads only
`report["metrics"]`, never `raw_generations`, even if the report has that opt-in field.

### Optional: narrate it with an LLM

`explain`'s rule-based summary can optionally be handed to an LLM to turn into a warmer,
free-form narrative, for someone who wants prose, not a scored checklist. This is **off by
default**, additive (it's appended after the rule-based summary, never replaces it), and:

- sends **only** the rule-based summary above (verdict labels and rounded scores), **never**
  `raw_generations`, never the raw report, never anything else on your machine;
- needs **your own API key** for whichever provider you pick, the same bring-your-own-key model
  the `openai` model source already uses; `privaudit` holds no key of its own;
- falls back cleanly to the rule-based summary alone if the call fails for any reason.

Four providers, picked with `--provider`: `anthropic`, `openai`, `huggingface` (via its
[Inference Providers router](https://huggingface.co/docs/inference-providers), an OpenAI-compatible
endpoint that can reach many hosted open models with one API key), and `openrouter` (a proxy in
front of many providers' models, also OpenAI-compatible). Each falls back to its own environment
variable when `--api-key` is omitted: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`,
`OPENROUTER_API_KEY`.

```bash
privaudit explain --report report.json --llm --provider anthropic --api-key sk-ant-...
# or set ANTHROPIC_API_KEY / OPENAI_API_KEY / HF_TOKEN / OPENROUTER_API_KEY and omit --api-key
privaudit explain --report report.json --llm --provider openai --llm-model gpt-4o-mini
privaudit explain --report report.json --llm --provider huggingface --llm-model meta-llama/Meta-Llama-3-8B-Instruct
privaudit explain --report report.json --llm --provider openrouter --llm-model anthropic/claude-3.5-sonnet
```

`--llm-base-url` overrides the provider's default endpoint (a self-hosted router, a proxy, ...).

In the UI, the same toggle lives under "5. AI narration".

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

- **`model`**: `source` is `huggingface` (hub identifier), `local_checkpoint` (a local
  directory), `api_endpoint` (a generic HTTP endpoint you host yourself), or `openai` (the
  OpenAI API). `api_endpoint` and `openai` both require `access: "black_box"`, and only `EM`
  can run against a black-box model, since `MIA`/`EZ_MIA` need per-token logits.
  - A **full fine-tune / DP-SGD checkpoint**, local or on the hub, is self-contained and
    loads directly from `identifier`.
  - A **LoRA/PEFT adapter** only contains adapter weights, so the base model is loaded first
    and the adapter applied on top of it. This is detected from `identifier` alone and works
    identically whether `identifier` is a local checkpoint directory or an adapter repo
    published on the HF hub, no separate config needed to say which. The base model is
    normally auto-detected from the adapter's own `adapter_config.json`
    (`base_model_name_or_path`, written automatically when the adapter was saved); set
    `model.base_model` (an HF hub id or local path) to override that, or to supply it
    explicitly if the adapter config doesn't have it (e.g. a hand-edited or partially-copied
    checkpoint). If the adapter itself ships no tokenizer (common for adapter-only repos),
    the base model's tokenizer is used instead. Setting `base_model` when `identifier` isn't
    actually a PEFT adapter is a fail-fast config error.
  - **`api_endpoint`**: `identifier` is your endpoint's URL. `generate_greedy` POSTs
    `{"prompt", "max_new_tokens", "temperature": 0, "top_p": 1}` as JSON and expects
    `{"text": "..."}` back.
  - **`openai`**: `identifier` is the OpenAI model name (e.g. `"gpt-4o-mini"`). Needs an API
    key: set `model.api_key`, or leave it out and set the `OPENAI_API_KEY` environment variable
    (preferred, so the key doesn't end up in a config file); `privaudit` fails fast with a
    clear error if neither is set. `model.base_url` optionally points at an OpenAI-compatible
    endpoint instead of `https://api.openai.com/v1`. Talks to the real Chat Completions API
    (`POST {base_url}/chat/completions`) via a plain HTTPS call, no `openai` package dependency.
- **`reference_model`**: same shape as `model`. Only required when `EZ_MIA` is one of the
  requested attacks.
- **`data.path`** is a local directory. Nothing under it is ever uploaded.
- **`data.schema`**: required, describes your dataset; there's no dataset-specific
  special-casing (see below). Every attack here uses `data.role: "member_nonmember_split"`.
- Each attack `family` (`EM`, `MIA`, `EZ_MIA`) may appear **at most once** per job.
- `privaudit validate` / `privaudit run` fail fast with a specific error message, e.g. `MIA`
  without `model.access: "white_box"`, `EZ_MIA` without a top-level `reference_model`, or
  `openai` without an API key anywhere.

### Dataset schema (`data.schema`)

```json
{
  "format": "csv",
  "member_file": "train.csv",
  "nonmember_file": "test.csv",
  "text_field": "note"
}
```

`format` is `csv` or `jsonl`. Use `text_template` instead of `text_field` to combine several
columns into one training-format string: it's a Python `str.format` template applied to each
row, and supports nested access for structured data, e.g. a chat-formatted JSONL row like
`{"dialog": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` can
use `"text_template": "Question: {dialog[0][content]}\nAnswer: {dialog[1][content]}"`.

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
  "job_id": "...",
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
  models/loader.py                   # HF hub / local checkpoint / API endpoint / openai, white- or black-box
  data/schema.py, member_split.py    # dataset schema, member/non-member loading
  metrics.py                         # dependency-light ROUGE-L, BLEU, TPR@FPR, n-gram matching
  report.py                          # JSON Schema validation (job config + report) and report building
  runner.py                          # orchestrates one job end to end; on_progress callback for live status
  explain.py                         # rule-based, local, no-network plain-language report summary
  llm_explain.py                     # optional: narrate explain.py's summary with an LLM (opt-in, BYO key)
  cli.py                             # `privaudit run` / `validate` / `explain` / `ui` / `ui-gradio`
  webapp/server.py                   # local web UI: FastAPI backend (extra: "webui")
  webapp/static/                     # the web UI's HTML/CSS/JS and self-hosted font files
  ui.py                              # legacy local Gradio front end (extra: "ui")
```

## Testing

```bash
pytest tests/ -q
```

Tests run against fake models (no GPU, no network, no real checkpoints) and are structured as:
- exact algorithmic correctness (ROUGE-L, BLEU, TPR@FPR, the EZ-MIA scoring kernel)
- per-attack pipeline sanity checks against fake models with synthetic, clearly-separable data
  (`test_em.py`, `test_mia.py`, `test_ez_mia.py`)
- a full end-to-end run (`test_integration.py`): one job config with all three attack families,
  real temp-directory CSV data, and a mocked model, driven through the actual
  `runner.run_job()` orchestration and the real `privaudit run`/`validate`/`explain` CLI entry
  points, not just each attack module called in isolation
- job-config schema validation and fail-fast error messages
- `explain.py` (rule-based summary bands, agreement/disagreement notes, and a check that it
  never reads `raw_generations`) and `llm_explain.py` (request/response shape for both
  providers, missing-key handling, and the same never-touches-`raw_generations` guarantee),
  the latter with `urllib.request.urlopen` monkeypatched, so no real network call is made
- the web UI's backend contract (`test_webapp.py`): `/api/validate` and a full WebSocket
  round-trip through `/ws/run`, with `run_job`/`explain_with_llm` mocked out
