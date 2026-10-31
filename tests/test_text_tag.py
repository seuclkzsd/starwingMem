"""Offline check of the v1.3 textual window tag (no network: embed() is stubbed).

Run: python tests/test_text_tag.py
"""
import hashlib, os, re, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "service"))
_tmp = tempfile.mkdtemp()
os.environ.update({"AML_DB": os.path.join(_tmp, "mem.sqlite"), "AML_CACHE": os.path.join(_tmp, "emb.sqlite"),
                   "AML_EMB_BUDGET_YUAN": "0", "AML_TOKEN": "", "DASHSCOPE_API_KEY": "offline"})
os.environ.pop("AML_TEXT_TAG", None)
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


dashscope.embed = fake_embed
import server
from fastapi.testclient import TestClient

assert server.TEXT_TAG, "the tag must be on by default"
c = TestClient(server.app)
msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i} about coffee and the trip to Kyoto"} for i in range(12)]
r = c.post("/add", json={"user_id": "u1", "session_id": "s1", "request_id": "r1", "messages": msgs})
assert r.status_code == 200, r.text
q = {"user_id": "u1", "query": "Where was the trip?", "top_k": 20}
tagged = c.post("/search", json=q).json()["data"]
server.TEXT_TAG = False
plain = c.post("/search", json=q).json()["data"]
assert tagged and len(tagged) == len(plain)
for k, (a, b) in enumerate(zip(tagged, plain), 1):
    assert isinstance(a["content"], str) and a["content"] == f'<past_conversation index="{k}">\n{b["content"]}\n</past_conversation>'
    assert {x: a[x] for x in ("id", "score", "created_at")} == {x: b[x] for x in ("id", "score", "created_at")}
assert len({d["id"] for d in tagged}) == len(tagged)
print("textual tag: verbatim text, same ids, scores and order")
