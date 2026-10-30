"""AML Add/Search HTTP service around memsys.MemoryStore.

  AML_TOKEN=... AML_DB=./data/mem.sqlite uvicorn server:app --host 0.0.0.0 --port 8000

Auth: Authorization: Bearer <token> | Token <token>, or X-Api-Key: <token>.
Add returns 200 only after the messages are persisted (SQLite) and embedded,
so they are immediately searchable. Search returns verbatim memory text only.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

from collections import defaultdict

from fastapi import Body, Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

# Embedding spend of the deployed service is tracked apart from local experiments (their cap is 30 yuan in
# exp/cache/emb_usage.log); of the 100-yuan budget the official Smoke/Full runs may use up to 65 by default.
os.environ.setdefault("AML_EMB_USAGE_LOG", str(Path(os.environ.get("AML_DB", Path(__file__).with_name("data") / "mem.sqlite")).with_name("emb_usage_server.log")))
os.environ.setdefault("AML_EMB_BUDGET_YUAN", "65")

import dashscope
from memsys import Config, MemoryStore, passages

TOKEN = os.environ.get("AML_TOKEN", "")
DB_PATH = Path(os.environ.get("AML_DB", Path(__file__).with_name("data") / "mem.sqlite"))
# Chosen by RESULTS.md (2026-10-03): rerank off at a 16k budget (it lowered coverage), ReFind-style session channel on.
# 2026-10-24: header_auto (natural dates when the query states no "now": LoCoMo +2.4, LME unchanged by construction).
# 2026-10-25: advice/preference questions get the top-20 hit digest first (BEAM +1.5 sig., LME +2.2, LoCoMo +0.2).
# 2026-10-25: platform "Streaming online QA feedback:" messages are kept in SQLite but not indexed (simulated
#   streaming, LoCoMo 500: storing them raw -3.8 sig., verified-answer examples only -2.8; LME raw -1.7).
DEFAULT_CFG = {"pack": "fill", "budget": 16000, "clean_query": True, "time_boost": 1.0, "future_penalty": 3.0,
               "time_annot": True, "w_session": 0.5, "header_auto": True, "route": {"advice": {"digest_k": 20}},
               "stream_fb": "drop"}
CFG = Config(**{**DEFAULT_CFG, **json.loads(os.environ.get("AML_CFG", "{}"))})

app = FastAPI(title="AML memory")
store = MemoryStore(CFG)
db_lock = threading.Lock()
user_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)  # serialise add/search per user_id


def open_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS adds (user_id TEXT, request_id TEXT, body TEXT, seq INTEGER PRIMARY KEY AUTOINCREMENT)")
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS adds_req ON adds(user_id, request_id)")
    return db


DB = open_db()
for (body,) in DB.execute("SELECT body FROM adds ORDER BY seq"):
    store.add(json.loads(body))


def auth(authorization: str | None = Header(None), x_api_key: str | None = Header(None)):
    if not TOKEN:
        return
    supplied = x_api_key
    if authorization:
        parts = authorization.split(None, 1)
        supplied = parts[1] if len(parts) == 2 and parts[0].lower() in ("bearer", "token") else authorization
    if supplied != TOKEN:
        raise HTTPException(401, "unauthorized")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/add", dependencies=[Depends(auth)])
def add(req: dict = Body(...)):
    if not isinstance(req, dict) or not req.get("user_id") or not isinstance(req.get("messages"), list):
        raise HTTPException(422, "user_id and messages are required")
    rid = req.get("request_id")
    # Embed first (fills the cache) so the following search does not pay for it.
    texts = []
    for m in req["messages"]:
        c = m.get("content", "")
        if isinstance(c, list):
            c = "\n".join(p.get("text", "") for p in c if p.get("type") == "text")
        texts += [f"{m.get('role', 'user')}: {p}" for p in passages(c)]
    if texts:
        dashscope.embed(texts, "document")
    with db_lock:
        if rid and DB.execute("SELECT 1 FROM adds WHERE user_id=? AND request_id=?", (req["user_id"], rid)).fetchone():
            return {"success": True, "request_id": rid, "user_id": req["user_id"], "session_id": req.get("session_id")}
        DB.execute("INSERT INTO adds (user_id, request_id, body) VALUES (?,?,?)",
                   (req["user_id"], rid, json.dumps(req, ensure_ascii=False)))
        DB.commit()
    with user_locks[req["user_id"]]:
        return store.add(req)


@app.post("/search", dependencies=[Depends(auth)])
def search(req: dict = Body(...)):
    if not isinstance(req, dict) or not req.get("user_id") or req.get("query") in (None, ""):
        raise HTTPException(422, "user_id and query are required")
    top_k = int(req.get("top_k") or 100)
    with user_locks[req["user_id"]]:
        res = store.search({**req, "top_k": top_k})
    return JSONResponse({"data": [{"id": d["id"], "content": d["content"], "score": d["score"],
                                   "created_at": d["created_at"]} for d in res["data"][:top_k]]})
