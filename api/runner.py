# api/runner.py - simple synchronous workflow runner (leadgen demo)
import os, json, csv, time
from .fusion import fusion_loop_sync_with_selection

def leadgen_workflow_sync(seed_query: str, providers, critic, limit=8):
    q = f"Find {limit} target companies for: {seed_query}. Output JSON list of objects: { '{name,website,one_line}' }"
    r = fusion_loop_sync_with_selection(q, None, critic, rounds=1)
    fused = r.get("fused","")
    companies = []
    try:
        companies = json.loads(fused)
    except Exception:
        for i,line in enumerate(fused.splitlines()):
            line=line.strip()
            if not line: continue
            parts = [p.strip() for p in line.replace('–','-').split('-')]
            if len(parts)>=2:
                companies.append({"name": parts[0], "website": parts[1]})
            else:
                companies.append({"name": line, "website": ""})
    results=[]
    for c in companies:
        prompt = f"Summarize {c.get('name')} ({c.get('website')}) in two sentences and provide 3 personalized outreach lines."
        res = fusion_loop_sync_with_selection(prompt, None, critic, rounds=1)
        results.append({"company": c, "summary": res.get("fused",""), "meta": res.get("meta",{})})
        time.sleep(0.2)
    fname = f"leads_{int(time.time())}.csv"
    with open(fname,"w",newline="",encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["name","website","summary"])
        for r in results:
            writer.writerow([r["company"].get("name",""), r["company"].get("website",""), r["summary"].replace('\n',' | ')])
    return {"file": fname, "count": len(results), "results": results}
