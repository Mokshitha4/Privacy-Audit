"""Optional local UI: a Gradio front end wrapping the same JSON-in/JSON-out contract as the CLI.

The form fields below assemble a job config dict, which is then handed to the exact same
`report.validate_job_config()` / `runner.run_job()` calls the CLI uses -- the UI adds no new
capability. Every field is pre-filled with its default (from the attack modules / schema); the
user edits what they need. The assembled JSON is shown in "Technical details" so it can be
copied straight into `privaudit run --config`.

Run streams live progress (model loading, each attack) into a status log via a background
thread, and once done renders a plain-language explanation of the results (privaudit.explain,
rule-based, no network call) front and center -- the raw report JSON stays available but tucked
into a collapsed "Technical details" section, since the primary audience for this page is
someone auditing a model who may have no privacy-attack background.

Requires the `ui` extra (`pip install -e ".[ui]"`); the core package has no UI dependency.

Local-first, not hosted: `launch()` always passes `share=False`, so the app only ever binds to
localhost. Gradio's `share=True` tunnels through Gradio's own public servers -- exactly the
hosted-execution model this package avoids -- so it is never used here.
"""
from __future__ import annotations

import json
import queue
import threading
import traceback
from typing import Optional

from .explain import render_markdown, summarize
from .llm_explain import LLMExplainConfig, explain_with_llm
from .report import validate_job_config
from .runner import run_job

_INITIAL_SUMMARY = "*Run a check below to see a plain-language explanation of the results here.*"


# ---------------------------------------------------------------------------
# JSON-string entry points (unchanged contract; used by the form layer below)
# ---------------------------------------------------------------------------

def _validate(config_text: str) -> str:
    try:
        job_cfg = json.loads(config_text)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    try:
        validate_job_config(job_cfg)
    except ValueError as e:
        return str(e)
    return "Config is valid."


def _run(config_text: str) -> str:
    try:
        job_cfg = json.loads(config_text)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"}, indent=2)
    try:
        report = run_job(job_cfg)
    except ValueError as e:
        return json.dumps({"error": str(e)}, indent=2)
    except Exception as e:
        # Model/data loading failures, OOM, etc. -- surface the traceback in the UI rather than
        # crashing the Gradio process, since a failed run shouldn't take the server down.
        return json.dumps({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}, indent=2)
    return json.dumps(report, indent=2)


# ---------------------------------------------------------------------------
# Form -> job config dict
# ---------------------------------------------------------------------------

# Order matters: build_app() passes the form components to the click handlers in exactly this
# order, and the handlers zip them back into a dict against these names.
_FIELD_ORDER = [
    "model_source", "model_identifier", "model_access", "model_base_model",
    "model_api_key", "model_base_url",
    "ft_enabled", "ft_regime", "ft_loss", "ft_epsilon",
    "use_reference", "ref_source", "ref_identifier", "ref_access", "ref_base_model",
    "ref_api_key", "ref_base_url",
    "data_role", "data_path",
    "schema_format", "schema_member_file", "schema_nonmember_file", "schema_text_field", "schema_text_template",
    "em_enabled", "em_prefix_len", "em_continuation_len", "em_max_samples", "em_ngram_ns",
    "mia_enabled", "mia_num_members", "mia_num_nonmembers", "mia_max_length", "mia_k_percent",
    "mia_n_folds", "mia_batch_size", "mia_seed",
    "ezmia_enabled", "ezmia_num_members", "ezmia_num_nonmembers", "ezmia_sequence_length",
    "ezmia_batch_size", "ezmia_seed",
    "return_raw_generations",
]

# Separate from _FIELD_ORDER: these describe an optional post-processing step (narrate the
# already-computed rule-based summary with an LLM), not anything in the job config itself, so
# they're never passed through _assemble_config()/validate_job_config().
_LLM_FIELD_ORDER = ["llm_enabled", "llm_provider", "llm_api_key", "llm_model"]


def _num(value):
    """Form value -> int when integral, float otherwise, None when blank.

    Accepts both real numbers (from gr.Number) and strings (from gr.Textbox fields used for
    optional caps, where gr.Number(value=None) unhelpfully renders as a literal 0)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"expected a number, got {value!r}") from e
    return int(number) if number.is_integer() else number


def _parse_int_list(text):
    if text is None or not str(text).strip():
        return None
    try:
        return [int(part) for part in str(text).replace(",", " ").split()]
    except ValueError as e:
        raise ValueError(f"ngram_ns must be a comma- or space-separated list of integers, got {text!r}") from e


def _model_block(source, identifier, access, base_model, api_key, base_url) -> dict:
    block = {"source": source, "identifier": (identifier or "").strip(), "access": access}
    if (base_model or "").strip():
        block["base_model"] = base_model.strip()
    if source == "openai":
        if (api_key or "").strip():
            block["api_key"] = api_key.strip()
        if (base_url or "").strip():
            block["base_url"] = base_url.strip()
    return block


def _attack_params(f: dict, prefix: str, keys: list[str]) -> dict:
    params: dict = {}
    for key in keys:
        n = _num(f.get(f"{prefix}_{key}"))
        if n is not None:
            params[key] = n
    return params


def _assemble_config(f: dict) -> dict:
    """Build a job config dict from the flat {field_name: value} mapping. Raises ValueError on
    field-level problems (e.g. an unparseable ngram list); schema-level problems are left for
    validate_job_config()."""
    model = _model_block(
        f["model_source"], f["model_identifier"], f["model_access"],
        f.get("model_base_model"), f.get("model_api_key"), f.get("model_base_url"),
    )
    if f.get("ft_enabled"):
        finetuning = {}
        if f.get("ft_regime"):
            finetuning["regime"] = f["ft_regime"]
        if f.get("ft_loss"):
            finetuning["loss"] = f["ft_loss"]
        epsilon = _num(f.get("ft_epsilon"))
        if epsilon is not None:
            finetuning["epsilon"] = epsilon
        if finetuning:
            model["finetuning"] = finetuning

    config: dict = {"model": model}

    if f.get("use_reference"):
        config["reference_model"] = _model_block(
            f["ref_source"], f["ref_identifier"], f["ref_access"],
            f.get("ref_base_model"), f.get("ref_api_key"), f.get("ref_base_url"),
        )

    schema = {
        "format": f["schema_format"],
        "member_file": (f["schema_member_file"] or "").strip(),
        "nonmember_file": (f["schema_nonmember_file"] or "").strip(),
    }
    template = (f.get("schema_text_template") or "").strip()
    field = (f.get("schema_text_field") or "").strip()
    if template:
        schema["text_template"] = f["schema_text_template"]
    elif field:
        schema["text_field"] = field
    config["data"] = {"role": f["data_role"], "path": (f["data_path"] or "").strip(), "schema": schema}

    attacks = []
    if f.get("em_enabled"):
        params = _attack_params(f, "em", ["prefix_len", "continuation_len", "max_samples"])
        ngram_ns = _parse_int_list(f.get("em_ngram_ns"))
        if ngram_ns:
            params["ngram_ns"] = ngram_ns
        attacks.append({"family": "EM", "variant": "default", "params": params})
    if f.get("mia_enabled"):
        params = _attack_params(
            f, "mia",
            ["num_members", "num_nonmembers", "max_length", "k_percent", "n_folds", "batch_size", "seed"],
        )
        attacks.append({"family": "MIA", "variant": "default", "params": params})
    if f.get("ezmia_enabled"):
        params = _attack_params(
            f, "ezmia",
            ["num_members", "num_nonmembers", "sequence_length", "batch_size", "seed"],
        )
        attacks.append({"family": "EZ_MIA", "variant": "default", "params": params})
    config["attacks"] = attacks

    config["output"] = {"return_raw_generations": bool(f.get("return_raw_generations"))}
    return config


def _form_config_dict(*values) -> dict:
    return _assemble_config(dict(zip(_FIELD_ORDER, values)))


def _preview_form(*values) -> str:
    try:
        return json.dumps(_form_config_dict(*values), indent=2)
    except ValueError as e:
        return f"// {e}"


def _validate_form(*values):
    try:
        config = _form_config_dict(*values)
    except ValueError as e:
        return str(e), ""
    config_text = json.dumps(config, indent=2)
    return _validate(config_text), config_text


def _run_form_streaming(*values):
    """Generator: yields (status_log, summary_markdown, report_json, config_json) tuples as the
    job progresses, so the page shows live progress instead of appearing frozen while a model
    loads and runs. The actual work happens on a background thread; progress messages arrive
    through a queue so this generator can yield without blocking on run_job() itself.

    `values` is job-config fields (per _FIELD_ORDER) followed by LLM-narration fields (per
    _LLM_FIELD_ORDER); the latter never touch the job config or validate_job_config()."""
    job_values = values[:len(_FIELD_ORDER)]
    llm_fields = dict(zip(_LLM_FIELD_ORDER, values[len(_FIELD_ORDER):]))

    try:
        config = _form_config_dict(*job_values)
    except ValueError as e:
        yield f"Config error: {e}", _INITIAL_SUMMARY, "", ""
        return

    config_text = json.dumps(config, indent=2)
    log_lines = ["Starting..."]
    yield "\n".join(log_lines), _INITIAL_SUMMARY, "", config_text

    progress_queue: "queue.Queue[tuple[str, Optional[str]]]" = queue.Queue()
    result_box: dict = {}

    def on_progress(message: str) -> None:
        progress_queue.put(("progress", message))

    def worker() -> None:
        try:
            result_box["report"] = run_job(config, on_progress=on_progress)
        except ValueError as e:
            result_box["error"] = str(e)
        except Exception as e:
            result_box["error"] = f"{type(e).__name__}: {e}"
            result_box["traceback"] = traceback.format_exc()
        finally:
            progress_queue.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        kind, payload = progress_queue.get()
        if kind == "progress":
            log_lines.append(payload)
            yield "\n".join(log_lines), _INITIAL_SUMMARY, "", config_text
        else:
            break

    if "error" in result_box:
        log_lines.append(f"Failed: {result_box['error']}")
        error_payload = {"error": result_box["error"]}
        if "traceback" in result_box:
            error_payload["traceback"] = result_box["traceback"]
        yield "\n".join(log_lines), _INITIAL_SUMMARY, json.dumps(error_payload, indent=2), config_text
        return

    report = result_box["report"]
    summary_markdown = render_markdown(summarize(report))
    report_text = json.dumps(report, indent=2)

    if llm_fields.get("llm_enabled"):
        log_lines.append("Generating AI narration...")
        yield "\n".join(log_lines), summary_markdown, report_text, config_text
        try:
            llm_text = explain_with_llm(report, LLMExplainConfig(
                provider=llm_fields["llm_provider"],
                api_key=(llm_fields.get("llm_api_key") or "").strip() or None,
                model=(llm_fields.get("llm_model") or "").strip() or None,
            ))
            summary_markdown = f"{summary_markdown}\n\n---\n### 🤖 AI narration ({llm_fields['llm_provider']})\n\n{llm_text}"
        except Exception as e:
            # Narration is additive and optional -- a failed call must never hide the
            # rule-based summary that already succeeded.
            log_lines.append(f"AI narration failed ({type(e).__name__}: {e}); showing the rule-based summary only.")

    log_lines.append("Done.")
    yield "\n".join(log_lines), summary_markdown, report_text, config_text


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

_CSS = """
.privaudit-lede { font-size: 1.05em; opacity: 0.85; margin-bottom: 0.5em; }
#privaudit-status textarea { font-family: ui-monospace, "SF Mono", Consolas, monospace; font-size: 0.85em; }
#privaudit-summary { padding: 1em 1.25em; border-radius: 12px; background: var(--block-background-fill); border: 1px solid var(--border-color-primary); }
"""


def build_app():
    try:
        import gradio as gr
    except ImportError as e:
        raise ImportError(
            "The UI requires the 'gradio' package. Install it with: pip install -e '.[ui]'"
        ) from e

    def _visible(flag):
        return gr.update(visible=bool(flag))

    with gr.Blocks(title="privaudit") as app:
        gr.Markdown("# 🔒 Privacy Audit")
        gr.Markdown(
            "Check whether a fine-tuned clinical LLM leaks details from its training data. Fill in "
            "the model and dataset below, run the checks, and read the plain-language summary -- no "
            "privacy-attack background needed. Everything runs on this machine and stays here; "
            "nothing is uploaded anywhere by this page.",
            elem_classes=["privaudit-lede"],
        )

        model_sources = ["huggingface", "local_checkpoint", "api_endpoint", "openai"]

        def _base_model_visible(source):
            return gr.update(visible=source in ("huggingface", "local_checkpoint"))

        def _openai_visible(source):
            return gr.update(visible=source == "openai")

        c = {}
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Accordion("1. Model", open=True):
                    c["model_source"] = gr.Dropdown(model_sources, value="huggingface", label="source")
                    c["model_identifier"] = gr.Textbox(
                        label="identifier",
                        placeholder="HF hub id, local checkpoint path, API endpoint URL, or OpenAI model name (e.g. gpt-4o-mini)",
                    )
                    c["model_access"] = gr.Dropdown(["white_box", "black_box"], value="white_box", label="access")
                    with gr.Group(visible=True) as model_base_model_group:
                        c["model_base_model"] = gr.Textbox(
                            label="base_model (optional)",
                            placeholder="e.g. Qwen/Qwen3-0.6B",
                            info="Only for a LoRA/PEFT checkpoint whose base model isn't recorded in its adapter_config.json.",
                        )
                    with gr.Group(visible=False) as model_openai_group:
                        c["model_api_key"] = gr.Textbox(
                            label="api_key", type="password",
                            placeholder="sk-... (blank = use the OPENAI_API_KEY environment variable)",
                        )
                        c["model_base_url"] = gr.Textbox(
                            label="base_url (optional)", placeholder="https://api.openai.com/v1",
                            info="Override to point at an OpenAI-compatible endpoint instead.",
                        )
                    c["ft_enabled"] = gr.Checkbox(value=False, label="Record fine-tuning metadata (optional)")
                    with gr.Group(visible=False) as ft_group:
                        c["ft_regime"] = gr.Dropdown(["full_ft", "qlora", "dp_sgd"], value="full_ft", label="regime")
                        c["ft_loss"] = gr.Dropdown(["full", "masked"], value="full", label="loss")
                        c["ft_epsilon"] = gr.Textbox(value="", label="epsilon", placeholder="DP-SGD budget; blank otherwise")

                with gr.Accordion("2. Reference model (required for EZ-MIA)", open=False):
                    c["use_reference"] = gr.Checkbox(value=False, label="Use a reference model")
                    with gr.Group(visible=False) as ref_group:
                        c["ref_source"] = gr.Dropdown(model_sources, value="huggingface", label="source")
                        c["ref_identifier"] = gr.Textbox(label="identifier", placeholder="HF hub id or local path")
                        c["ref_access"] = gr.Dropdown(["white_box", "black_box"], value="white_box", label="access")
                        with gr.Group(visible=True) as ref_base_model_group:
                            c["ref_base_model"] = gr.Textbox(label="base_model (optional)")
                        with gr.Group(visible=False) as ref_openai_group:
                            c["ref_api_key"] = gr.Textbox(
                                label="api_key", type="password",
                                placeholder="sk-... (blank = use the OPENAI_API_KEY environment variable)",
                            )
                            c["ref_base_url"] = gr.Textbox(label="base_url (optional)", placeholder="https://api.openai.com/v1")

                with gr.Accordion("3. Data", open=True):
                    c["data_role"] = gr.Dropdown(
                        ["member_nonmember_split", "rag_corpus"], value="member_nonmember_split", label="role"
                    )
                    c["data_path"] = gr.Textbox(
                        label="path",
                        placeholder="/local/path/to/your/data",
                        info="Local directory holding the member/non-member files. Never uploaded anywhere.",
                    )
                    gr.Markdown("**Dataset schema**: how to read your files and build text from each row")
                    c["schema_format"] = gr.Dropdown(["csv", "jsonl"], value="csv", label="format")
                    c["schema_member_file"] = gr.Textbox(value="train.csv", label="member_file")
                    c["schema_nonmember_file"] = gr.Textbox(value="test.csv", label="nonmember_file")
                    c["schema_text_field"] = gr.Textbox(
                        value="note", label="text_field",
                        info="Column/key holding the text. Used when text_template is blank.",
                    )
                    c["schema_text_template"] = gr.Textbox(
                        value="", label="text_template (optional)",
                        placeholder="Question: {dialog[0][content]}\\nAnswer: {dialog[1][content]}",
                        info="Overrides text_field; Python str.format against each row (supports nested [idx][key]).",
                    )

                with gr.Accordion("4. Attacks", open=True):
                    c["em_enabled"] = gr.Checkbox(value=True, label="EM: Exact Memorization")
                    with gr.Group(visible=True) as em_group:
                        c["em_prefix_len"] = gr.Number(value=50, precision=0, label="prefix_len")
                        c["em_continuation_len"] = gr.Number(
                            value=500, precision=0, label="continuation_len",
                            info="tokens generated and held out as groundtruth",
                        )
                        c["em_max_samples"] = gr.Textbox(
                            value="", label="max_samples", placeholder="all", info="blank = all eligible sequences"
                        )
                        c["em_ngram_ns"] = gr.Textbox(
                            value="10, 20, 30, 50", label="ngram_ns", info="memorization-event n-gram window sizes"
                        )

                    c["mia_enabled"] = gr.Checkbox(value=False, label="MIA: Membership Inference (Random Forest)")
                    with gr.Group(visible=False) as mia_group:
                        c["mia_num_members"] = gr.Textbox(
                            value="", label="num_members", placeholder="all", info="blank = all available"
                        )
                        c["mia_num_nonmembers"] = gr.Textbox(
                            value="", label="num_nonmembers", placeholder="all", info="blank = all available"
                        )
                        c["mia_max_length"] = gr.Number(value=512, precision=0, label="max_length")
                        c["mia_k_percent"] = gr.Number(value=20, label="k_percent", info="Min-k%-Prob's k")
                        c["mia_n_folds"] = gr.Number(value=5, precision=0, label="n_folds")
                        c["mia_batch_size"] = gr.Number(value=8, precision=0, label="batch_size")
                        c["mia_seed"] = gr.Number(value=42, precision=0, label="seed")

                    c["ezmia_enabled"] = gr.Checkbox(
                        value=False, label="EZ_MIA: lightweight MIA (needs a reference model)"
                    )
                    with gr.Group(visible=False) as ezmia_group:
                        c["ezmia_num_members"] = gr.Textbox(
                            value="", label="num_members", placeholder="all", info="blank = all available"
                        )
                        c["ezmia_num_nonmembers"] = gr.Textbox(
                            value="", label="num_nonmembers", placeholder="all", info="blank = all available"
                        )
                        c["ezmia_sequence_length"] = gr.Number(value=128, precision=0, label="sequence_length")
                        c["ezmia_batch_size"] = gr.Number(value=8, precision=0, label="batch_size")
                        c["ezmia_seed"] = gr.Number(value=42, precision=0, label="seed")

                with gr.Accordion("5. Output", open=True):
                    c["return_raw_generations"] = gr.Checkbox(
                        value=False, label="return_raw_generations",
                        info="Include raw model generations in the report (off by default).",
                    )

                with gr.Accordion("6. AI narration (optional)", open=False):
                    c["llm_enabled"] = gr.Checkbox(
                        value=False, label="Narrate the results with an LLM",
                        info="Sends only the rule-based summary below -- verdicts and rounded scores, "
                             "never raw text or raw_generations -- to the provider you choose. Off by default.",
                    )
                    with gr.Group(visible=False) as llm_group:
                        c["llm_provider"] = gr.Dropdown(["anthropic", "openai"], value="anthropic", label="provider")
                        c["llm_api_key"] = gr.Textbox(
                            label="api_key", type="password",
                            placeholder="blank = use ANTHROPIC_API_KEY / OPENAI_API_KEY environment variable",
                        )
                        c["llm_model"] = gr.Textbox(
                            label="model (optional)",
                            placeholder="blank = provider's small/fast default (e.g. claude-sonnet-5, gpt-4o-mini)",
                        )

                with gr.Row():
                    preview_btn = gr.Button("Preview config")
                    validate_btn = gr.Button("Validate")
                    run_btn = gr.Button("Run", variant="primary")

            with gr.Column(scale=4):
                gr.Markdown("### Status")
                status_box = gr.Textbox(
                    show_label=False, interactive=False, lines=6, max_lines=12,
                    elem_id="privaudit-status", placeholder="Nothing running yet.",
                )
                gr.Markdown("### Results")
                summary_markdown = gr.Markdown(_INITIAL_SUMMARY, elem_id="privaudit-summary")
                with gr.Accordion("Technical details", open=False):
                    config_preview = gr.Code(label="Generated job config", language="json", interactive=False)
                    report_box = gr.Code(label="Report JSON", language="json", interactive=False)

        # Show/hide dependent groups.
        c["model_source"].change(_base_model_visible, c["model_source"], model_base_model_group)
        c["model_source"].change(_openai_visible, c["model_source"], model_openai_group)
        c["ref_source"].change(_base_model_visible, c["ref_source"], ref_base_model_group)
        c["ref_source"].change(_openai_visible, c["ref_source"], ref_openai_group)
        c["ft_enabled"].change(_visible, c["ft_enabled"], ft_group)
        c["use_reference"].change(_visible, c["use_reference"], ref_group)
        c["em_enabled"].change(_visible, c["em_enabled"], em_group)
        c["mia_enabled"].change(_visible, c["mia_enabled"], mia_group)
        c["ezmia_enabled"].change(_visible, c["ezmia_enabled"], ezmia_group)
        c["llm_enabled"].change(_visible, c["llm_enabled"], llm_group)

        job_inputs = [c[name] for name in _FIELD_ORDER]
        llm_inputs = [c[name] for name in _LLM_FIELD_ORDER]
        preview_btn.click(fn=_preview_form, inputs=job_inputs, outputs=config_preview)
        validate_btn.click(fn=_validate_form, inputs=job_inputs, outputs=[status_box, config_preview])
        run_btn.click(
            fn=_run_form_streaming, inputs=job_inputs + llm_inputs,
            outputs=[status_box, summary_markdown, report_box, config_preview],
        )

    app.queue()
    return app


def launch(server_name: str = "127.0.0.1", server_port: Optional[int] = None) -> None:
    import gradio as gr

    app = build_app()
    theme = gr.themes.Soft(primary_hue="teal", secondary_hue="slate")
    app.launch(server_name=server_name, server_port=server_port, share=False, theme=theme, css=_CSS)
