"""AML Add/Search HTTP service around memsys.MemoryStore.

  AML_TOKEN=... AML_DB=./data/mem.sqlite uvicorn server:app --host 0.0.0.0 --port 8000

Auth: Authorization: Bearer <token> | Token <token>, or X-Api-Key: <token>.
Add returns 200 only after the messages are persisted (SQLite) and embedded,
so they are immediately searchable. Search returns verbatim memory text only.

Each track has its own store, config and SQLite file, chosen by path prefix:
/add + /search (textual), /mm/add + /mm/search, /coding/add + /coding/search.
The platform sends opaque ids, so tracks are told apart by the endpoints bound
to each platform version, not by request content.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
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
# Diagnostics only (what was asked, how much came back); evaluation data, deleted with the rest after the cycle.
SEARCH_LOG = Path(os.environ.get("AML_SEARCH_LOG", DB_PATH.with_name("search_log.jsonl")))
log_lock = threading.Lock()
# Chosen by RESULTS.md (2026-10-03): rerank off at a 16k budget (it lowered coverage), ReFind-style session channel on.
# 2026-10-24: header_auto (natural dates when the query states no "now": LoCoMo +2.4, LME unchanged by construction).
# 2026-10-25: advice/preference questions get the top-20 hit digest first (BEAM +1.5 sig., LME +2.2, LoCoMo +0.2).
# 2026-10-25: platform "Streaming online QA feedback:" messages are kept in SQLite but not indexed (simulated
#   streaming, LoCoMo 500: storing them raw -3.8 sig., verified-answer examples only -2.8; LME raw -1.7).
DEFAULT_CFG = {"pack": "fill", "budget": 16000, "clean_query": True, "time_boost": 1.0, "future_penalty": 3.0,
               "time_annot": True, "w_session": 0.5, "header_auto": True, "route": {"advice": {"digest_k": 20}},
               "stream_fb": "drop"}
# Per-track overrides on top of DEFAULT_CFG; AML_CFG / AML_CFG_MM / AML_CFG_CODING override further.
TRACK_CFG = {"": {}, "mm": {}, "coding": {}}
TRACK_ENV = {"": "AML_CFG", "mm": "AML_CFG_MM", "coding": "AML_CFG_CODING"}


class Track:
    """One store + its SQLite log of Add bodies (replayed on start)."""

    def __init__(self, name: str):
        self.name = name
        self.cfg = Config(**{**DEFAULT_CFG, **TRACK_CFG[name], **json.loads(os.environ.get(TRACK_ENV[name], "{}"))})
        self.store = MemoryStore(self.cfg)
        self.db_lock = threading.Lock()
        self.user_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)  # serialise add/search per user_id
        self.db = open_db(DB_PATH if not name else DB_PATH.with_name(f"{DB_PATH.stem}_{name}{DB_PATH.suffix}"))
        for (body,) in self.db.execute("SELECT body FROM adds ORDER BY seq"):
            self.store.add(json.loads(body))


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS adds (user_id TEXT, request_id TEXT, body TEXT, seq INTEGER PRIMARY KEY AUTOINCREMENT)")
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS adds_req ON adds(user_id, request_id)")
    return db


app = FastAPI(title="AML memory")
TRACKS = {name: Track(name) for name in TRACK_CFG}
CFG, store = TRACKS[""].cfg, TRACKS[""].store       # textual track (kept for scripts that import these)


def auth(authorization: str | None = Header(None), x_api_key: str | None = Header(None)):
    if not TOKEN:
        return
    supplied = x_api_key
    if authorization:
        parts = authorization.split(None, 1)
        supplied = parts[1] if len(parts) == 2 and parts[0].lower() in ("bearer", "token") else authorization
    if supplied != TOKEN:
        raise HTTPException(401, "unauthorized")


def health():
    return {"status": "ok"}


def do_add(t: Track, req: dict):
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
    with t.db_lock:
        if rid and t.db.execute("SELECT 1 FROM adds WHERE user_id=? AND request_id=?", (req["user_id"], rid)).fetchone():
            return {"success": True, "request_id": rid, "user_id": req["user_id"], "session_id": req.get("session_id")}
        t.db.execute("INSERT INTO adds (user_id, request_id, body) VALUES (?,?,?)",
                     (req["user_id"], rid, json.dumps(req, ensure_ascii=False)))
        t.db.commit()
    with t.user_locks[req["user_id"]]:
        return t.store.add(req)


def do_search(t: Track, req: dict):
    if not isinstance(req, dict) or not req.get("user_id") or req.get("query") in (None, ""):
        raise HTTPException(422, "user_id and query are required")
    top_k = int(req.get("top_k") or 100)
    t0 = time.time()
    with t.user_locks[req["user_id"]]:
        res = t.store.search({**req, "top_k": top_k})
    q = req["query"]
    rec = {"ts": round(t0, 3), "track": t.name or "textual", "user_id": req["user_id"], "top_k": top_k,
           "query": (q if isinstance(q, str) else json.dumps(q, ensure_ascii=False))[:500],
           "query_images": 0 if isinstance(q, str) else sum(1 for p in q if isinstance(p, dict) and p.get("type") == "image_url"),
           "options": req.get("options"), "n": len(res["data"][:top_k]),
           "chars": sum(len(d["content"]) for d in res["data"][:top_k] if isinstance(d["content"], str)),
           "sec": round(time.time() - t0, 3)}
    with log_lock, open(SEARCH_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # The contract wants a non-empty string for Textual/Coding but an ordered ContentPart[] for Multimodal
    # (v1.0.1 returned strings there and its multimodal Smoke scored 0 on all 13 questions).
    wrap = (lambda c: c if isinstance(c, list) else [{"type": "text", "text": c}]) if t.name == "mm" else (lambda c: c)
    return JSONResponse({"data": [{"id": d["id"], "content": wrap(d["content"]), "score": d["score"],
                                   "created_at": d["created_at"]} for d in res["data"][:top_k]]})


def _routes(t: Track):
    prefix = f"/{t.name}" if t.name else ""

    def add(req: dict = Body(...)):
        return do_add(t, req)

    def search(req: dict = Body(...)):
        return do_search(t, req)

    app.get(prefix + "/health")(health)
    app.post(prefix + "/add", dependencies=[Depends(auth)])(add)
    app.post(prefix + "/search", dependencies=[Depends(auth)])(search)


for _t in TRACKS.values():
    _routes(_t)
