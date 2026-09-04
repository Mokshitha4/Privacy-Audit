"""CLI entry point: `privaudit run --config job.json --output report.json`."""
from __future__ import annotations

import argparse
import json
import sys

from .report import validate_job_config
from .runner import run_job


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="privaudit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the attacks in a job config and write a report.")
    run_parser.add_argument("--config", required=True, help="Path to the job config JSON.")
    run_parser.add_argument("--output", required=True, help="Path to write the report JSON.")

    validate_parser = subparsers.add_parser("validate", help="Validate a job config without running anything.")
    validate_parser.add_argument("--config", required=True, help="Path to the job config JSON.")

    ui_parser = subparsers.add_parser("ui", help="Launch a local Gradio UI (requires the 'ui' extra: pip install -e '.[ui]').")
    ui_parser.add_argument("--port", type=int, default=None, help="Port to serve on (default: let Gradio pick one).")

    args = parser.parse_args(argv)

    if args.command == "ui":
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

    if args.command == "run":
        job_cfg = _load_json(args.config)
        try:
            report = run_job(job_cfg)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
