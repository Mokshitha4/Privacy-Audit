"""CLI entry point: `privaudit run --config job.json --output report.json`."""
from __future__ import annotations

import argparse
import json
import sys

from .explain import render_markdown, summarize
from .report import validate_job_config
from .runner import run_job


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_utf8_console() -> None:
    """Force stdout/stderr to UTF-8. Windows consoles often default to a legacy codepage (e.g.
    cp1252) that can't encode the emoji/dashes in `explain`'s output, crashing with
    UnicodeEncodeError on an otherwise-successful run. Safe to skip if the stream doesn't
    support reconfiguring (e.g. output already redirected to something that manages its own
    encoding)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def main(argv=None) -> None:
    _ensure_utf8_console()
    parser = argparse.ArgumentParser(prog="privaudit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the attacks in a job config and write a report.")
    run_parser.add_argument("--config", required=True, help="Path to the job config JSON.")
    run_parser.add_argument("--output", required=True, help="Path to write the report JSON.")

    validate_parser = subparsers.add_parser("validate", help="Validate a job config without running anything.")
    validate_parser.add_argument("--config", required=True, help="Path to the job config JSON.")

    explain_parser = subparsers.add_parser(
        "explain", help="Print a plain-language summary of a report (rule-based, no network call)."
    )
    explain_parser.add_argument("--report", required=True, help="Path to a report JSON produced by `privaudit run`.")
    explain_parser.add_argument("--output", help="Write the summary here instead of printing it.")
    explain_parser.add_argument(
        "--llm", action="store_true",
        help="Also narrate the summary with an LLM (opt-in; sends only the rule-based summary above -- "
             "verdicts and rounded scores, never raw data -- to the provider you choose). Requires --provider.",
    )
    explain_parser.add_argument(
        "--provider", choices=["anthropic", "openai", "huggingface", "openrouter"], help="LLM provider for --llm."
    )
    explain_parser.add_argument("--llm-model", help="Model name for --llm (default: the provider's small/fast model).")
    explain_parser.add_argument(
        "--api-key",
        help="API key for --llm (default: read from ANTHROPIC_API_KEY / OPENAI_API_KEY / HF_TOKEN / OPENROUTER_API_KEY).",
    )
    explain_parser.add_argument(
        "--llm-base-url", help="Override the provider's API base URL for --llm (e.g. a self-hosted or proxy endpoint)."
    )

    ui_parser = subparsers.add_parser(
        "ui", help="Launch the local web UI (requires the 'webui' extra: pip install -e '.[webui]')."
    )
    ui_parser.add_argument("--port", type=int, default=None, help="Port to serve on (default: 8765).")
    ui_parser.add_argument("--no-browser", action="store_true", help="Don't automatically open a browser tab.")

    gradio_ui_parser = subparsers.add_parser(
        "ui-gradio", help="Launch the legacy Gradio UI (requires the 'ui' extra: pip install -e '.[ui]')."
    )
    gradio_ui_parser.add_argument("--port", type=int, default=None, help="Port to serve on (default: let Gradio pick one).")

    args = parser.parse_args(argv)

    if args.command == "ui":
        from .webapp import launch
        try:
            launch(port=args.port, open_browser=not args.no_browser)
        except ImportError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.command == "ui-gradio":
        from .ui import launch
        try:
            launch(server_port=args.port)
        except ImportError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.command == "validate":
        job_cfg = _load_json(args.config)
        try:
            validate_job_config(job_cfg)
        except ValueError as e:
            print(f"Invalid config: {e}", file=sys.stderr)
            sys.exit(1)
        print("Config is valid.")
        return

    if args.command == "explain":
        report = _load_json(args.report)
        text = render_markdown(summarize(report))
        if args.llm:
            if not args.provider:
                print("Error: --llm requires --provider {anthropic,openai,huggingface,openrouter}.", file=sys.stderr)
                sys.exit(1)
            from .llm_explain import LLMExplainConfig, explain_with_llm
            try:
                llm_text = explain_with_llm(
                    report, LLMExplainConfig(
                        provider=args.provider, api_key=args.api_key, model=args.llm_model, base_url=args.llm_base_url
                    )
                )
                text = f"{text}\n\n---\n## AI narration ({args.provider})\n\n{llm_text}"
            except Exception as e:
                # Narration is additive and optional -- a failed call should never hide the
                # rule-based summary that already succeeded.
                print(f"Warning: LLM narration failed ({type(e).__name__}: {e}); showing the rule-based summary only.", file=sys.stderr)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Explanation written to {args.output}")
        else:
            print(text)
        return

    if args.command == "run":
        job_cfg = _load_json(args.config)
        try:
            report = run_job(job_cfg, on_progress=lambda message: print(f"[privaudit] {message}"))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
