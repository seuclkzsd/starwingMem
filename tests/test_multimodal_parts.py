"""Offline check of multimodal handling (no network: embed() is stubbed).

A user whose Add carried content parts gets ContentPart[] back, everyone else gets strings; auth; restart replay.
Run: python tests/test_multimodal_parts.py
"""
import hashlib, importlib.util, json, os, random, shutil, sys, tempfile
from pathlib import Path

svc = str(Path(__file__).resolve().parents[1] / "service")
work = tempfile.mkdtemp()
sys.path.insert(0, svc)
os.environ.update(AML_TOKEN="t0k", AML_EMB_BUDGET_YUAN="0", AML_SEARCH_LOG=os.path.join(work, "search_log.jsonl"))
import numpy as np
import dashscope

def fake_embed(texts, kind="document", workers=8):
    out = []
    for t in texts:
        v = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)).standard_normal(dashscope.EMB_DIM)
        out.append((v / np.linalg.norm(v)).astype(np.float32))
    return (np.stack(out) if out else np.zeros((0, dashscope.EMB_DIM), np.float32)), [{w: 1.0 for w in t.lower().split()[:20]} for t in texts]

def no_net(*a, **k):
    raise RuntimeError("network call in offline test")

dashscope.embed = fake_embed
dashscope.chat = dashscope.chat_cached = dashscope.rerank = no_net
from fastapi.testclient import TestClient

def load(path, name, db):
    os.environ["AML_DB"] = db
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return TestClient(m.app), m

rng = random.Random(1)
words = "coffee tea movie hiking python recipe guitar paris tokyo budget gym book sister birthday".split()
adds = []
for s in range(30):
    n = 1 if s % 3 == 0 else rng.randint(2, 25)
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": " ".join(rng.choice(words) for _ in range(rng.randint(5, 400)))}
            for i in range(n)]
    for i, m in enumerate(msgs):
        if (s + i) % 2: m["timestamp"] = 1700000000000 + s * 86400000 + i * 1000
    adds.append({"request_id": f"r{s}", "user_id": f"u{s % 3}", "session_id": f"s{s}", "messages": msgs})
adds.append({"request_id": "fb", "user_id": "u0", "session_id": "s99", "messages": [
    {"role": "user", "content": "Streaming online QA feedback: question: x; answer: y; score: 1"}]})
queries = ["What would you recommend for my sister's birthday?", "Now is 2023-12-01. What did I say about tokyo last week?",
           "How many books did I mention?", [{"type": "text", "text": "Which recipe did I like?"}], "coffee or tea?"]
H = {"Authorization": "Token t0k"}

def run(client, prefix=""):
    for a in adds:
        r = client.post(prefix + "/add", json=a, headers=H); assert r.status_code == 200, r.text
    assert client.post(prefix + "/add", json=adds[0], headers=H).json()["success"]          # idempotent
    out = []
    for u in ("u0", "u1", "u2", "nobody"):
        for q in queries:
            for k in (3, 100):
                body = {"query": q, "user_id": u, "top_k": k}
                if q == "coffee or tea?": body["options"] = ["coffee", "tea"]
                r = client.post(prefix + "/search", json=body, headers=H); assert r.status_code == 200, r.text
                out.append(r.json())
    return out

new_c, new_m = load(os.path.join(svc, "server.py"), "server_new", f"{work}/new/mem.sqlite")
got = run(new_c)
print("textual: %d searches ok" % len(got))

mm = {"request_id": "m1", "user_id": "mu", "session_id": "ms", "messages": [
    {"role": "user", "content": "Alex: switching to Starbucks."},
    {"role": "user", "content": [
        {"type": "text", "text": "Alex: this is the Ethos Water piece.\nImage Caption: a bottle pours a heart"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"}}]}]}
assert new_c.post("/add", json=mm, headers=H).status_code == 200
r = new_c.post("/search", json={"query": [{"type": "text", "text": "Which water brand?"}], "user_id": "mu", "top_k": 5}, headers=H).json()
c0 = r["data"][0]["content"]
assert isinstance(c0, list) and c0[0]["type"] == "text" and "Ethos" in c0[0]["text"], r      # ContentPart[] for mm users
assert all(isinstance(d["content"], str) for x in got for d in x["data"]), "textual users must get strings"
assert new_c.post("/add", json=adds[0]).status_code == 401
print("multimodal user gets ContentPart[]; textual users get strings; auth ok")
again_c, _ = load(os.path.join(svc, "server.py"), "server_new2", f"{work}/new/mem.sqlite")   # restart: replay
r2 = again_c.post("/search", json={"query": [{"type": "text", "text": "Which water brand?"}], "user_id": "mu", "top_k": 5}, headers=H).json()
assert r2 == r, "mm replay differs"
assert again_c.post("/search", json={"query": queries[0], "user_id": "u0", "top_k": 100}, headers=H).json() == got[1]
print("restart replay: identical")
