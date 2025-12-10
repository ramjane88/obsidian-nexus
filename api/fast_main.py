# api/fast_main.py - FastAPI serverless entrypoint for Obsidian Nexus
import os, json, uuid, time, asyncio
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from typing import Any, Dict

# import sync fusion functions from api.fusion
from api.fusion import fusion_loop_sync_with_selection, load_providers_config, load_free_provider, load_provider_costs

app = FastAPI(title="Obsidian Nexus API (FastAPI)")

PROVIDERS = load_providers_config()
FREE_PROVIDER = load_free_provider()
COSTS = load_provider_costs()

# simple in-memory job store (sufficient for PoC)
JOBS: Dict[str, Dict[str, Any]] = {}

@app.get("/api/health")
async def health():
    return {"status": "ok", "ts": int(time.time())}

@app.get("/api/providers")
async def providers_list():
    out = []
    for p in PROVIDERS:
        name = p.get("name")
        out.append({
            "name": name,
            "model": p.get("model"),
            "docs": p.get("docs", ""),
            "estimated_cost_per_token": COSTS.get(name, None),
            "notes": p.get("notes", "")
        })
    return {"providers": out, "free_provider": FREE_PROVIDER}

# run fusion in a thread (because fusion_loop_sync_with_selection is sync)
async def run_fusion_job(job_id: str, prompt: str, provider_choice, critic, rounds: int, max_tokens: int, budget, token_limit, fallback_to_free: bool):
    JOBS[job_id]["status"] = "running"
    try:
        res = await asyncio.to_thread(
            fusion_loop_sync_with_selection,
            prompt,
            provider_choice,
            critic,
            rounds,
            max_tokens,
            budget,
            token_limit,
            fallback_to_free
        )
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = res
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)

@app.post("/api/run")
async def run_endpoint(payload: Dict, background_tasks: BackgroundTasks):
    prompt = payload.get("prompt", "")
    rounds = int(payload.get("rounds", 1))
    provider_choice = payload.get("provider_choice", None)
    budget = payload.get("budget", None)
    token_limit = payload.get("token_limit", None)
    fallback_to_free = bool(payload.get("fallback_to_free", True))
    max_tokens = int(payload.get("max_tokens", 512))
    critic = payload.get("critic", None) or {"name":"openai","env":"OPENAI_API_KEY"}

    # attempt to parse stringified provider_choice
    if isinstance(provider_choice, str):
        try:
            provider_choice = json.loads(provider_choice)
        except:
            pass

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status":"queued", "created": int(time.time())}
    background_tasks.add_task(run_fusion_job, job_id, prompt, provider_choice, critic, rounds, max_tokens, budget, token_limit, fallback_to_free)
    return {"job_id": job_id}

@app.get("/api/job/{job_id}")
async def job_status(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job not found")
    return j

# friendly root for quick checks
@app.get("/")
async def root():
    return JSONResponse({"name":"Obsidian Nexus API","endpoints":["/api/health","/api/providers","/api/run","/api/job/{id}"]})
