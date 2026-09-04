"""Optional local UI: a Gradio front end wrapping the same JSON-in/JSON-out contract as the CLI.

This adds no new capability over `privaudit run` / `privaudit validate` -- it's a friendlier
local front end for the same `report.validate_job_config()` / `runner.run_job()` calls.
Requires the `ui` extra (`pip install -e ".[ui]"`); the core package has no UI dependency.

Local-first, not hosted: `launch()` always passes `share=False`, so the app only ever binds to
localhost. Gradio's `share=True` tunnels through Gradio's own public servers -- exactly the
hosted-execution model this package avoids -- so it is never used here.
"""
from __future__ import annotations

import json
import traceback
from typing import Optional

from .report import validate_job_config
from .runner import run_job

_EXAMPLE_CONFIG = {
    "model": {
        "source": "huggingface",
        "identifier": "your-org/your-finetuned-model",
        "access": "white_box",
    },
    "data": {
        "role": "member_nonmember_split",
        "known_dataset": "custom",
        "schema": {
            "format": "csv",
            "member_file": "train.csv",
            "nonmember_file": "test.csv",
            "text_field": "note",
        },
        "path": "/local/path/to/your/data",
    },
    "attacks": [
        {"family": "EM", "variant": "default", "params": {"continuation_len": 300}},
    ],
    "output": {"return_raw_generations": False},
}


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


def build_app():
    try:
        import gradio as gr
    except ImportError as e:
        raise ImportError(
            "The UI requires the 'gradio' package. Install it with: pip install -e '.[ui]'"
        ) from e

    with gr.Blocks(title="privaudit") as app:
        gr.Markdown(
            "# Privacy Audit\n"
            "Local privacy-leakage audit for fine-tuned clinical LLMs. Paste a job config, "
            "validate it, then run it. Everything runs on this machine and stays here -- "
            "nothing is uploaded anywhere by this UI."
        )
        config_box = gr.Code(
            value=json.dumps(_EXAMPLE_CONFIG, indent=2),
            language="json",
            label="Job config",
            lines=24,
        )
        with gr.Row():
            validate_btn = gr.Button("Validate")
            run_btn = gr.Button("Run", variant="primary")
        status_box = gr.Textbox(label="Validation result", interactive=False)
        report_box = gr.Code(label="Report", language="json", lines=24, interactive=False)

        validate_btn.click(fn=_validate, inputs=config_box, outputs=status_box)
        run_btn.click(fn=_run, inputs=config_box, outputs=report_box)

    return app


def launch(server_name: str = "127.0.0.1", server_port: Optional[int] = None) -> None:
    app = build_app()
    app.launch(server_name=server_name, server_port=server_port, share=False)
