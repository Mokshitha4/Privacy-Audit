"""Local web UI: a FastAPI app serving a static frontend over the same JSON-in/JSON-out
contract as the CLI (`report.validate_job_config()` / `runner.run_job()`).

Requires the `webui` extra (`pip install -e ".[webui]"`); the core package has no UI dependency.
"""
from .server import launch

__all__ = ["launch"]
