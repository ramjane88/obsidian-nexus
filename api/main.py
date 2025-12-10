# api/main.py - Flask local backend with expanded provider metadata
from flask import Flask, request, jsonify
import os, uuid, threading, time, json
from .fusion import fusion_loop_sync_with_selection, load_providers_config, load_free_provider, load_provider_costs

app = Flask(__name__)

PROVIDERS = load_providers_config()
FREE_PROVIDER = load_free_provider()
COSTS = load_provider_costs()

JOBS = {}

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","ts": int(time.time())})

@app.route("/api/providers", methods=["GET"])
def providers_list():
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
    return jsonify({"providers": out, "free_provider": FREE_PROVIDER})

@app.route("/api/run", methods=["POST"])
def run():
    data = request.get_json(force=True)
    prompt = data.get("prompt","")
    rounds = int(data.get("rounds",1))
    provider_choice = data.get("provider_choice", None)
    budget = data.get("budget", None)
    token_limit = data.get("token_limit", None)
    fallback_to_free = bool(data.get("fallback_to_free", True))
    max_tokens = int(data.get("max_tokens", 512))
    critic = data.get("critic", None) or {"name":"openai","env":"OPENAI_API_KEY"}

    if isinstance(provider_choice, str):
        try:
            provider_choice = json.loads(provider_choice)
        except:
            pass

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status":"queued", "created": int(time.time())}

    def job_fn():
        JOBS[job_id]["status"] = "running"
        try:
            out = fusion_loop_sync_with_selection(prompt, provider_choice, critic, rounds=rounds, max_tokens=max_tokens, budget=budget, token_limit=token_limit, fallback_to_free=fallback_to_free)
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["result"] = out
        except Exception as e:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)

    t = threading.Thread(target=job_fn, daemon=True)
    t.start()
    return jsonify({"job_id": job_id})

@app.route("/api/job/<job_id>", methods=["GET"])
def job_status(job_id):
    j = JOBS.get(job_id)
    if not j:
        return jsonify({"error":"job not found"}), 404
    return jsonify(j)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
