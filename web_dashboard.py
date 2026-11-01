from pathlib import Path
from threading import Thread, Lock
import json
import subprocess
import sys
import time
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
UI = ROOT / "ui"
SCRIPT = ROOT / "scripts" / "run_experiment.py"

if not UI.is_dir():
    raise RuntimeError(f"UI folder not found: {UI}")

if not (UI / "index.html").is_file():
    raise RuntimeError(f"Missing frontend file: {UI / 'index.html'}")

app = FastAPI(title="QAWS Dashboard")
app.mount("/static", StaticFiles(directory=str(UI)), name="static")

LOCK = Lock()

ALLOWED_MODELS = {
    "qwen_0_5b", "qwen_1_5b", "qwen_3b",
    "llama_1b", "llama_3b",
    "smollm_135m", "smollm_360m", "smollm_1_7b",
    "gemma_270m", "gemma_1b", "gemma_4b",
}

ALLOWED_PRECISIONS = {"FP16", "INT8", "INT4", "INT3", "INT2"}

EXCLUDED_DIRS = {
    ".venv", "venv", ".git", "__pycache__",
    "node_modules", ".idea",
}

job = {
    "running": False,
    "status": "idle",
    "log": "",
    "started_at": None,
    "finished_at": None,
    "return_code": None,
    "command": None,
}


class RunRequest(BaseModel):
    model: str = "qwen_0_5b"
    precision: str = "FP16"


@app.get("/")
def home():
    return FileResponse(
        UI / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "project": ROOT.name,
        "dashboard_version": "3",
    }


def file_mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0


def find_result_files():
    found = []

    for path in ROOT.rglob("*.json"):
        try:
            relative_parts = path.relative_to(ROOT).parts

            if any(part in EXCLUDED_DIRS for part in relative_parts):
                continue

            if path.name.lower() in {
                "package.json", "tsconfig.json"
            }:
                continue

            if path.is_file():
                found.append(path)

        except (OSError, ValueError):
            continue

    return sorted(
        found,
        key=file_mtime,
        reverse=True,
    )


def normalise_key(key):
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def walk_dicts(obj):
    if isinstance(obj, dict):
        yield obj

        for value in obj.values():
            yield from walk_dicts(value)

    elif isinstance(obj, list):
        for value in obj:
            yield from walk_dicts(value)


def extract_metric(data, *keys):
    wanted = {normalise_key(key) for key in keys}
    sources = []

    if isinstance(data.get("metrics"), dict):
        sources.append(data["metrics"])

    sources.extend(
        source for source in walk_dicts(data)
        if source is not data
        and source is not data.get("metrics")
    )

    sources.append(data)

    for source in sources:
        for key, value in source.items():
            if (
                normalise_key(key) in wanted
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                return value

    return None


def first_value(data, *keys, default=None):
    for source in walk_dicts(data):
        for key in keys:
            value = source.get(key)

            if (
                value is not None
                and value != ""
                and not isinstance(value, (dict, list))
            ):
                return value

    return default


@app.get("/api/results")
def results():
    items = []
    files = find_result_files()

    for path in files[:150]:
        try:
            data = json.loads(
                path.read_text(encoding="utf-8-sig")
            )

            if not isinstance(data, dict):
                continue

            model = first_value(
                data, "model", "model_name", "model_id",
                default="—",
            )

            precision = first_value(
                data, "precision", "quantization",
                "quantization_method", "bits",
                default="—",
            )

            latency = extract_metric(
                data,
                "latency_seconds",
                "latency",
                "total_latency_seconds",
                "latency_s",
                "inference_time_seconds",
                "inference_time",
                "total_time_seconds",
                "elapsed_seconds",
                "elapsed_time",
                "duration_seconds",
                "duration",
                "generation_time_seconds",
                "avg_latency_seconds",
                "average_latency",
                "time_taken_seconds",
                "seconds_per_run",
            )

            throughput = extract_metric(
                data,
                "tokens_per_second",
                "throughput_tokens_per_second",
                "throughput",
                "tokens_sec",
                "tok_per_s",
                "generation_tokens_per_second",
            )

            ram = extract_metric(
                data,
                "ram_gb",
                "system_ram_gb",
                "peak_ram_gb",
                "memory_gb",
                "peak_memory_gb",
                "ram_usage_gb",
                "system_memory_gb",
            )

            gpu_memory = extract_metric(
                data,
                "gpu_memory_gb",
                "peak_gpu_memory_gb",
                "gpu_allocated_gb",
                "gpu_memory_allocated_gb",
                "max_gpu_memory_gb",
                "vram_gb",
                "peak_vram_gb",
            )

            items.append({
                "file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "name": str(first_value(
                    data, "run_name", "experiment_name", "name",
                    default=path.stem,
                )),
                "model": str(model),
                "precision": str(precision).upper(),
                "latency": latency,
                "throughput": throughput,
                "ram": ram,
                "gpu_memory": gpu_memory,
                "modified": file_mtime(path),
            })

        except (
            OSError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            continue

    return {
        "count": len(files),
        "results": items,
    }


def run_worker(model, precision):
    command = [
        sys.executable,
        str(SCRIPT),
        "--model", model,
        "--precision", precision,
    ]

    try:
        with LOCK:
            job.update({
                "running": True,
                "status": "running",
                "log": "$ " + subprocess.list2cmdline(command) + "\n\n",
                "started_at": time.time(),
                "finished_at": None,
                "return_code": None,
                "command": command,
            })

        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        if process.stdout:
            for line in process.stdout:
                with LOCK:
                    job["log"] = (
                        job["log"] + line
                    )[-30000:]

        return_code = process.wait()

        with LOCK:
            job["return_code"] = return_code
            job["status"] = (
                "completed" if return_code == 0 else "failed"
            )

            if return_code != 0:
                job["log"] += (
                    f"\nExperiment exited with code {return_code}.\n"
                )

    except Exception as exc:
        with LOCK:
            job["status"] = "failed"
            job["log"] += (
                f"\nERROR: {type(exc).__name__}: {exc}\n"
            )

    finally:
        with LOCK:
            job["running"] = False
            job["finished_at"] = time.time()


@app.post("/api/run")
def run_experiment(request: RunRequest):
    if not SCRIPT.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Experiment script not found: {SCRIPT}",
        )

    model = request.model.strip().lower()
    precision = request.precision.strip().upper()

    if model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {model}",
        )

    if precision not in ALLOWED_PRECISIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported precision: {precision}",
        )

    command = [
        sys.executable,
        str(SCRIPT),
        "--model", model,
        "--precision", precision,
    ]

    with LOCK:
        if job["running"]:
            raise HTTPException(
                status_code=409,
                detail="An experiment is already running.",
            )

        # Reserve the job before launching the worker.
        job.update({
            "running": True,
            "status": "starting",
            "log": "Preparing experiment...\n",
            "started_at": time.time(),
            "finished_at": None,
            "return_code": None,
            "command": command,
        })

    Thread(
        target=run_worker,
        args=(model, precision),
        daemon=True,
    ).start()

    return {
        "message": f"Started {model} with {precision}",
        "model": model,
        "precision": precision,
    }


@app.get("/api/status")
def status():
    with LOCK:
        return dict(job)