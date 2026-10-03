"""FastAPI backend for the local web UI.

Two endpoints do the real work, both thin wrappers around the same functions the CLI uses:

- POST /api/validate  -> report.validate_job_config()
- WS   /ws/run         -> runner.run_job(), streaming progress messages as they happen, then
                           a single final message with the report, the rule-based explanation
                           (privaudit.explain, no network call), and -- only if requested -- an
                           LLM narration on top of it.

Everything else is static files. No telemetry, no external calls except the opt-in LLM
narration step, which only ever sends the rule-based summary (verdicts + rounded scores),
never raw text or raw_generations.

Local-first, not hosted: `launch()` always binds to localhost and never passes through a
public tunnel.
"""
from __future__ import annotations

import asyncio
import mimetypes
import queue
import threading
import traceback
import webbrowser
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..explain import summarize
from ..llm_explain import LLMExplainConfig, explain_with_llm
from ..report import validate_job_config
from ..runner import run_job

# Some Python installs' mimetypes database predates woff2; without this it's served as
# application/octet-stream, which some browsers refuse to load as a web font.
mimetypes.add_type("font/woff2", ".woff2")

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="privaudit")


class ValidateRequest(BaseModel):
    config: dict


@app.post("/api/validate")
def api_validate(req: ValidateRequest) -> dict:
    try:
        validate_job_config(req.config)
    except ValueError as e:
        return {"valid": False, "error": str(e)}
    return {"valid": True, "error": None}


@app.websocket("/ws/run")
async def ws_run(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
    except WebSocketDisconnect:
        return

    config = payload.get("config") or {}
    llm_cfg = payload.get("llm") or {}

    try:
        validate_job_config(config)
    except ValueError as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()
        return

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
            # Model/data loading failures, OOM, etc. -- surface the traceback in the UI
            # rather than taking the server down.
            result_box["error"] = f"{type(e).__name__}: {e}"
            result_box["traceback"] = traceback.format_exc()
        finally:
            progress_queue.put(("done", None))

    threading.Thread(target=worker, daemon=True).start()

    loop = asyncio.get_event_loop()
    try:
        while True:
            kind, message = await loop.run_in_executor(None, progress_queue.get)
            if kind == "progress":
                await websocket.send_json({"type": "progress", "message": message})
            else:
                break

        if "error" in result_box:
            await websocket.send_json({
                "type": "error",
                "message": result_box["error"],
                "traceback": result_box.get("traceback"),
            })
            return

        report = result_box["report"]
        explanation = summarize(report)

        narration = None
        narration_error = None
        if llm_cfg.get("enabled"):
            await websocket.send_json({"type": "progress", "message": "Generating AI narration..."})
            try:
                narration = explain_with_llm(report, LLMExplainConfig(
                    provider=llm_cfg["provider"],
                    api_key=(llm_cfg.get("api_key") or "").strip() or None,
                    model=(llm_cfg.get("model") or "").strip() or None,
                    base_url=(llm_cfg.get("base_url") or "").strip() or None,
                ))
            except Exception as e:
                # Narration is additive and optional -- a failed call must never hide the
                # rule-based summary that already succeeded.
                narration_error = f"{type(e).__name__}: {e}"

        await websocket.send_json({
            "type": "result",
            "report": report,
            "summary": explanation,
            "narration": narration,
            "narration_error": narration_error,
        })
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass  # already closed


# Mounted last so it never shadows the /api and /ws routes above (Starlette matches in order,
# and a root-mounted StaticFiles would otherwise swallow everything).
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")


def launch(host: str = "127.0.0.1", port: Optional[int] = None, open_browser: bool = True) -> None:
    import uvicorn

    port = port or 8765
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
