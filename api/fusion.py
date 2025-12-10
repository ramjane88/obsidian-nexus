# api/fusion.py - Multi-provider fusion with 20+ provider stubs
# Safe local synchronous implementation for Termux testing.
# Add real keys/endpoints in env vars and implement provider calls where noted.

import os, json, time, requests
from typing import List, Dict, Any, Optional

# ------------------------------
# Utility helpers
# ------------------------------
def safe_json_load(s):
    try:
        return json.loads(s)
    except:
        return None

def estimate_tokens(prompt: str) -> int:
    return max(1, int(len(prompt) / 4))

def load_providers_config() -> List[Dict[str,Any]]:
    cfg = safe_json_load(os.getenv("ONX_PROVIDERS_JSON","[]"))
    return cfg if isinstance(cfg, list) else []

def load_provider_costs() -> Dict[str,float]:
    costs = safe_json_load(os.getenv("ONX_PROVIDER_COSTS_JSON","{}"))
    if isinstance(costs, dict): return costs
    return {
        "openai": 0.00006, "anthropic": 0.00005, "gemini": 0.00005,
        "groq": 0.000008, "deepseek": 0.000005, "mistral":0.000004,
        "together":0.000002, "qwen":0.000003, "aws_titan":0.00005,
        "cohere":0.000002, "ibm":0.00001, "xai":0.00002, "voyage":0.000001,
        "elevenlabs":0.000001, "replicate":0.000001, "hf":0.000001,
        "perplexity":0.00002, "nvidia":0.000003
    }

def load_free_provider() -> Optional[str]:
    v = os.getenv("ONX_FREE_PROVIDER","")
    return v if v else None

# ------------------------------
# Low-level provider call wrappers (synchronous)
# ------------------------------
def call_openai_sync(api_key: str, prompt: str, model="gpt-4o-mini", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[openai-missing-key] {prompt[:200]}"}
    url = "https://api.openai.com/v1/chat/completions"
    payload = {"model": model, "messages":[{"role":"user","content":prompt}], "max_tokens": max_tokens}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    return r.status_code, r.text, r.json() if r.ok else None

def call_anthropic_sync(api_key: str, prompt: str, model="claude-3-available", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[anthropic-missing-key] {prompt[:200]}"}
    url = "https://api.anthropic.com/v1/complete"
    payload = {"model": model, "prompt": prompt, "max_tokens_to_sample": max_tokens}
    headers = {"x-api-key": api_key, "Content-Type":"application/json"}
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    return r.status_code, r.text, r.json() if r.ok else None

def call_gemini_sync(api_key: str, prompt: str, model="gemini-2.0", max_tokens=512):
    # GEMINI: placeholder. Implement real HTTP call if you have private endpoint/key.
    if not api_key:
        return 401, None, {"generated_text": f"[gemini-missing-key] {prompt[:200]}"}
    return 200, None, {"generated_text": f"[gemini-stub] {prompt[:400]}"}

def call_groq_sync(api_key: str, prompt: str, model="groq-llama-70b", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[groq-missing-key] {prompt[:200]}"}
    return 200, None, {"generated_text": f"[groq-stub] {prompt[:400]}"}

def call_deepseek_sync(api_key: str, prompt: str, model="deepseek-r1", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[deepseek-missing-key] {prompt[:200]}"}
    return 200, None, {"generated_text": f"[deepseek-stub] {prompt[:400]}"}

def call_mistral_sync(api_key: str, prompt: str, model="mistral-large-2", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[mistral-missing-key] {prompt[:200]}"}
    return 200, None, {"generated_text": f"[mistral-stub] {prompt[:400]}"}

def call_together_sync(api_key: str, prompt: str, model="together-model", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[together-missing-key] {prompt[:200]}"}
    return 200, None, {"generated_text": f"[together-stub] {prompt[:400]}"}

def call_qwen_sync(api_key: str, prompt: str, model="qwen-72b", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[qwen-missing-key] {prompt[:200]}"}
    return 200, None, {"generated_text": f"[qwen-stub] {prompt[:400]}"}

def call_aws_titan_sync(api_key: str, prompt: str, model="titan-3", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[aws-titan-missing-key] {prompt[:200]}"}
    return 200, None, {"generated_text": f"[aws-titan-stub] {prompt[:400]}"}

def call_cohere_sync(api_key: str, prompt: str, model="command-xlarge", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[cohere-missing-key] {prompt[:200]}"}
    url = "https://api.cohere.ai/generate"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        return r.status_code, r.text, r.json() if r.ok else None
    except Exception:
        return 500, None, {"generated_text": f"[cohere-call-failed] {prompt[:200]}"}

def call_ibm_sync(api_key: str, prompt: str, model="ibm-granite", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[ibm-missing-key] {prompt[:200]}"}
    return 200, None, {"generated_text": f"[ibm-stub] {prompt[:400]}"}

def call_xai_sync(api_key: str, prompt: str, model="grok-2", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[xai-missing-key] {prompt[:200]}"}
    return 200, None, {"generated_text": f"[xai-stub] {prompt[:400]}"}

def call_voyage_sync(api_key: str, prompt: str, model="voyage-embed", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[voyage-missing-key] {prompt[:200]}"}
    return 200, None, {"generated_text": f"[voyage-stub] {prompt[:400]}"}

def call_eleven_sync(api_key: str, prompt: str, model="eleven-voice", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[eleven-missing-key] {prompt[:200]}"}
    return 200, None, {"generated_text": f"[eleven-stub] {prompt[:400]}"}

def call_replicate_sync(api_key: str, prompt: str, model="replicate-model", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[replicate-missing-key] {prompt[:200]}"}
    return 200, None, {"generated_text": f"[replicate-stub] {prompt[:400]}"}

def call_hf_sync(api_key: str, prompt: str, model="google/flan-t5-large", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[hf-missing-key] {prompt[:200]}"}
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        return r.status_code, r.text, r.json() if r.ok else None
    except Exception:
        return 500, None, {"generated_text": f"[hf-call-failed] {prompt[:200]}"}

def call_perplexity_sync(api_key: str, prompt: str, model="perplexity-r1", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[perplexity-missing-key] {prompt[:200]}"}
    return 200, None, {"generated_text": f"[perplexity-stub] {prompt[:400]}"}

def call_nvidia_sync(api_key: str, prompt: str, model="nvidia-llm", max_tokens=512):
    if not api_key:
        return 401, None, {"generated_text": f"[nvidia-missing-key] {prompt[:200]}"}
    return 200, None, {"generated_text": f"[nvidia-stub] {prompt[:400]}"}

# ------------------------------
# Provider dispatcher
# ------------------------------
def call_provider_sync_by_name(name: str, api_key: str, prompt: str, model_hint: str=None, max_tokens: int=512):
    n = name.lower()
    if n == "openai":
        return call_openai_sync(api_key, prompt, model_hint or os.getenv("OPENAI_MODEL","gpt-4o-mini"), max_tokens=max_tokens)
    if n == "anthropic":
        return call_anthropic_sync(api_key, prompt, model_hint or os.getenv("ANTHROPIC_MODEL","claude-3"), max_tokens=max_tokens)
    if n == "gemini":
        return call_gemini_sync(api_key, prompt, model_hint or os.getenv("GEMINI_MODEL","gemini-2.0"), max_tokens=max_tokens)
    if n == "groq":
        return call_groq_sync(api_key, prompt, model_hint or "groq-llama-70b", max_tokens=max_tokens)
    if n == "deepseek":
        return call_deepseek_sync(api_key, prompt, model_hint or "deepseek-r1", max_tokens=max_tokens)
    if n == "mistral":
        return call_mistral_sync(api_key, prompt, model_hint or "mistral-large-2", max_tokens=max_tokens)
    if n == "together":
        return call_together_sync(api_key, prompt, model_hint or "together-default", max_tokens=max_tokens)
    if n == "qwen":
        return call_qwen_sync(api_key, prompt, model_hint or "qwen-72b", max_tokens=max_tokens)
    if n == "aws_titan" or n=="titan":
        return call_aws_titan_sync(api_key, prompt, model_hint or "titan-3", max_tokens=max_tokens)
    if n == "cohere":
        return call_cohere_sync(api_key, prompt, model_hint or "command-xlarge", max_tokens=max_tokens)
    if n == "ibm":
        return call_ibm_sync(api_key, prompt, model_hint or "ibm-granite", max_tokens=max_tokens)
    if n == "xai" or n=="grok":
        return call_xai_sync(api_key, prompt, model_hint or "grok-2", max_tokens=max_tokens)
    if n == "voyage":
        return call_voyage_sync(api_key, prompt, model_hint or "voyage-embed", max_tokens=max_tokens)
    if n == "elevenlabs" or n=="eleven":
        return call_eleven_sync(api_key, prompt, model_hint or "eleven-voice", max_tokens=max_tokens)
    if n == "replicate":
        return call_replicate_sync(api_key, prompt, model_hint or "replicate-model", max_tokens=max_tokens)
    if n == "hf" or n=="huggingface":
        return call_hf_sync(api_key, prompt, model_hint or os.getenv("HF_MODEL","google/flan-t5-large"), max_tokens=max_tokens)
    if n == "perplexity":
        return call_perplexity_sync(api_key, prompt, model_hint or "perplexity-r1", max_tokens=max_tokens)
    if n == "nvidia":
        return call_nvidia_sync(api_key, prompt, model_hint or "nvidia-llm", max_tokens=max_tokens)
    return 200, None, {"generated_text": f"[unknown-provider {name}] {prompt[:200]}"}

# ------------------------------
# Critic scoring and consensus
# ------------------------------
def simple_extract_text_from_response(res):
    if not res:
        return ""
    if isinstance(res, dict):
        if "choices" in res:
            try:
                return res["choices"][0]["message"]["content"]
            except:
                pass
        if "generated_text" in res:
            return res.get("generated_text","")
        if "text" in res:
            return res.get("text","")
        if "completion" in res:
            return res.get("completion","")
    return str(res)

def critic_score_sync(critic_provider: Dict[str,str], question: str, replies: List[Dict[str,Any]]):
    prov = critic_provider.get("name")
    env = critic_provider.get("env")
    key = os.getenv(env or "") if env else ""
    probe = "You are a critic. Evaluate these replies for accuracy, coherence, usefulness on scale 0-10 and return a JSON list of {id,score,reason}.\n\nQUESTION:\n" + question + "\n\nREPLIES:\n"
    for r in replies:
        probe += f"\nID:{r['id']}\n{r['text']}\n"
    probe += "\nReturn only JSON."
    status, raw, res = call_provider_sync_by_name(prov, key, probe, max_tokens=512)
    if not res:
        return [{"id": r["id"], "score": 5, "reason": "fallback"} for r in replies]
    text = simple_extract_text_from_response(res)
    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        return [{"id": r["id"], "score": 5, "reason": "parse_fallback"} for r in replies]

def consensus_aggregate(replies: List[Dict[str,Any]], scores: List[Dict[str,Any]]):
    score_map = {s["id"]: float(s.get("score",5)) for s in scores}
    sorted_replies = sorted(replies, key=lambda r: score_map.get(r["id"],5), reverse=True)
    top = sorted_replies[:3]
    fused = "\n\n".join([t["text"] for t in top])
    reasons = {s["id"]: s.get("reason","") for s in scores}
    meta = {"top_ids":[t["id"] for t in top], "scores": score_map, "reasons": reasons}
    return fused, meta

# ------------------------------
# Provider selection (budget / token limit / fallback)
# ------------------------------
def choose_provider_chain(prompt: str,
                          preferred: Optional[Any],
                          max_tokens: int,
                          budget: Optional[float],
                          token_limit: Optional[int],
                          fallback_to_free: bool = True) -> List[Dict[str,Any]]:
    configs = load_providers_config()
    costs = load_provider_costs()
    free_name = load_free_provider()
    prefs = []
    if preferred is None:
        prefs = [c.get("name") for c in configs]
    elif isinstance(preferred, list):
        prefs = preferred
    else:
        prefs = [preferred]
    est_tokens = estimate_tokens(prompt)
    if token_limit is not None and est_tokens > token_limit:
        if fallback_to_free and free_name:
            return [p for p in configs if p.get("name")==free_name] or []
        return []
    for name in prefs:
        conf = next((c for c in configs if c.get("name")==name), None)
        if not conf: continue
        per = float(costs.get(name, costs.get(conf.get("name"), 0.0)))
        est_cost = est_tokens * per
        if budget is None or est_cost <= float(budget):
            return [conf]
    if fallback_to_free and free_name:
        conf = next((c for c in configs if c.get("name")==free_name), None)
        if conf: return [conf]
    return configs

# ------------------------------
# Fusion loop using selected providers
# ------------------------------
def fusion_loop_sync_with_selection(question: str,
                                    preferred: Optional[Any],
                                    critic: Dict[str,str],
                                    rounds: int = 1,
                                    max_tokens: int = 512,
                                    budget: Optional[float] = None,
                                    token_limit: Optional[int] = None,
                                    fallback_to_free: bool = True):
    providers_chain = choose_provider_chain(question, preferred, max_tokens, budget, token_limit, fallback_to_free)
    if not providers_chain:
        return {"fused":"[no provider available]","meta":{"reason":"no_provider"},"replies":[],"scores":[]}
    replies = []
    for ridx in range(rounds):
        for idx,p in enumerate(providers_chain):
            key = os.getenv(p.get("env","") or "")
            model_hint = p.get("model")
            status, raw, res = call_provider_sync_by_name(p.get("name"), key, question, model_hint, max_tokens=max_tokens)
            text = simple_extract_text_from_response(res)
            if not text:
                text = raw or "[no response]"
            replies.append({"id": f"r{ridx}_m{idx}", "provider": p.get("name"), "text": text})
            time.sleep(0.08)
    scores = critic_score_sync(critic, question, replies)
    fused, meta = consensus_aggregate(replies, scores)
    meta["provider_chain"] = [p.get("name") for p in providers_chain]
    meta["estimated_tokens"] = estimate_tokens(question)
    return {"fused": fused, "meta": meta, "replies": replies, "scores": scores}
