"""Offline check that Search ids are unique within every response (no network: embed() is stubbed).

Run: python tests/test_search_ids.py
"""
import hashlib, os, random, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
_tmp = tempfile.mkdtemp()
os.environ["AML_DB"] = os.path.join(_tmp, "mem.sqlite")
os.environ["AML_CACHE"] = os.path.join(_tmp, "emb.sqlite")
os.environ["AML_EMB_BUDGET_YUAN"] = "0"
import numpy as np
import dashscope

def fake_embed(texts, kind="document", workers=8):
    out = []
    for t in texts:
        seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).standard_normal(dashscope.EMB_DIM).astype(np.float32)
        out.append(v / np.linalg.norm(v))
    sp = [{w: 1.0 for w in t.lower().split()[:20]} for t in texts]
    return (np.stack(out) if out else np.zeros((0, dashscope.EMB_DIM), np.float32)), sp

def no_net(*a, **k):
    raise RuntimeError("network call in offline test")

dashscope.embed = fake_embed
dashscope.chat = dashscope.chat_cached = dashscope.rerank = no_net
import memsys, server

store = memsys.MemoryStore(server.CFG)
rng = random.Random(0)
words = "coffee tea movie hiking python recipe guitar paris tokyo budget gym book".split()
# Many one-message sessions: the shape that made the digest item collide with a packed window.
for s in range(40):
    n = 1 if s % 2 == 0 else rng.randint(2, 6)
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "timestamp": 1700000000000 + s * 86400000 + i * 1000,
             "content": " ".join(rng.choice(words) for _ in range(rng.randint(5, 40)))} for i in range(n)]
    store.add({"request_id": f"r{s}", "user_id": "u1", "session_id": f"s{s}", "messages": msgs})
for j, topic in enumerate(["scuba diving in Bali", "sourdough baking", "vintage synthesizers"]):
    store.add({"request_id": f"solo{j}", "user_id": "u1", "session_id": f"solo{j}", "messages": [
        {"role": "user", "timestamp": 1710000000000 + j * 1000, "content": f"I have been really into {topic} lately."}]})
queries = ["Can you recommend something related to scuba diving in Bali?",
           "Any suggestions for sourdough baking tools?", "What would you suggest about vintage synthesizers?",
           "What would you recommend for my next trip?", "Any suggestions for a new hobby?",
           "Can you recommend a recipe I'd like?", "How many books did I mention?", "What did I say about coffee?",
           "Which movie should I watch tonight? I prefer something like before."]
bad = 0
for q in queries:
    for top_k in (1, 5, 100):
        d = store.search({"query": q, "user_id": "u1", "top_k": top_k})["data"]
        ids = [x["id"] for x in d]
        if len(ids) != len(set(ids)):
            bad += 1
            print("DUPLICATE", q, top_k, ids[:5])
        assert len(d) <= top_k
print("route of first query:", memsys.qtype(queries[0]), "| first ids:", [x["id"] for x in store.search({"query": queries[0], "user_id": "u1", "top_k": 100})["data"]][:3])
print("responses with duplicate ids:", bad)
sys.exit(1 if bad else 0)
