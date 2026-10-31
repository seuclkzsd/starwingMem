"""AML Add/Search HTTP service around memsys.MemoryStore.

  AML_TOKEN=... AML_DB=./data/mem.sqlite uvicorn server:app --host 0.0.0.0 --port 8000

Auth: Authorization: Bearer <token> | Token <token>, or X-Api-Key: <token>.
Add returns 200 only after the messages are persisted (SQLite) and embedded,
so they are immediately searchable. Search returns verbatim memory text only.

All three tracks share these endpoints (the platform fixes them per key) and send
opaque ids, so a user is treated as multimodal once one of its Adds carries
content parts; its Search results then come back as ContentPart[], which the
contract asks for on that track. Everyone else gets strings.
v1.1: a user one of whose Adds carried an image_url part is served by mmmem.MMStore
instead (images on disk, returned after the text; lazy load, LRU); see mmmem.py.
v1.1: a user whose first Add is an agent trajectory (codemem.is_trajectory_add) is served by codemem.CodeStore from
then on (BM25 cards, no embedding or generative model at Add or Search, its own SQLite); image users never are.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

from collections import defaultdict

from fastapi import Body, Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

# Embedding spend of the deployed service is tracked apart from local experiments (their cap is 30 yuan in
# exp/cache/emb_usage.log). Official Smoke and Full runs may use up to 150 yuan by default (Full estimate: 87-139).
os.environ.setdefault("AML_EMB_USAGE_LOG", str(Path(os.environ.get("AML_DB", Path(__file__).with_name("data") / "mem.sqlite")).with_name("emb_usage_server.log")))
os.environ.setdefault("AML_EMB_BUDGET_YUAN", "150")

import codemem
import dashscope
import mmmem
from memsys import Config, MemoryStore, feedback_view, passages

TOKEN = os.environ.get("AML_TOKEN", "")
DB_PATH = Path(os.environ.get("AML_DB", Path(__file__).with_name("data") / "mem.sqlite"))
# Diagnostics only (hashes, lengths, counts, timings, never the text of a question, an option or a message; query_stats);
# evaluation data, deleted with the rest after the cycle.
SEARCH_LOG = Path(os.environ.get("AML_SEARCH_LOG", DB_PATH.with_name("search_log.jsonl")))
# Chosen by RESULTS.md (2026-10-03): rerank off at a 16k budget (it lowered coverage), ReFind-style session channel on.
# 2026-10-24: header_auto (natural dates when the query states no "now": LoCoMo +2.4, LME unchanged by construction).
# 2026-10-25: advice/preference questions get the top-20 hit digest first (BEAM +1.5 sig., LME +2.2, LoCoMo +0.2).
# 2026-10-25: platform "Streaming online QA feedback:" messages are kept in SQLite but not indexed (simulated
#   streaming, LoCoMo 500: storing them raw -3.8 sig., verified-answer examples only -2.8; LME raw -1.7).
# 2026-10-31: 200k passages indexed at most (a textual Full is ~880k passages at 18-26 KB each, the server has 14 GB);
#   evicted users rebuild from the embedding cache on their next Search: same response, no API cost.
DEFAULT_CFG = {"pack": "fill", "budget": 16000, "clean_query": True, "time_boost": 1.0, "future_penalty": 3.0,
               "time_annot": True, "w_session": 0.5, "header_auto": True, "route": {"advice": {"digest_k": 20}},
               "stream_fb": "drop", "max_indexed": 200_000}
CFG = Config(**{**DEFAULT_CFG, **json.loads(os.environ.get("AML_CFG", "{}"))})
# v1.1 multimodal path (users whose Adds carried an image_url part; Textual/Coding contracts allow text parts only):
# clock time kept in window headers (MobileMem: 38% of temporal answers are minutes/hours), 16k of text, images after.
# 16k from the 300-question local run (MobileMem+MemLens, Qwen3-14B): 24k-16k -0.045 [-0.090,-0.001], 8k-16k +0.014 n.s.
# max_indexed off: MMStore has its own LRU (mmmem.MM_DEFAULTS max_passages).
# max_items 88: text and images share top_k=100, this leaves room for all 12 images (MobileMem: 9.5% of the questions
# reach 95+ windows; mean images 3.46 -> 3.89, questions with none 6.1% -> 4.3%, text tokens unchanged).
MM_CFG = Config(**{**DEFAULT_CFG, "header_auto": False, "budget": 16000, "max_indexed": 0, "max_items": 88,
                   **json.loads(os.environ.get("AML_CFG_MM", "{}"))})
MM_OPTS = json.loads(os.environ.get("AML_MM", "{}"))      # mmmem.MM_DEFAULTS overrides, e.g. {"img_max": 0}
BLOB_DIR = Path(os.environ.get("AML_BLOBS", DB_PATH.with_name("blobs")))
# Structure of multimodal Adds only (part-type layout, counts, bytes), never content; deleted with the rest.
ADD_LOG = Path(os.environ.get("AML_ADD_LOG", DB_PATH.with_name("add_log.jsonl")))
# v1.1 coding path. AML_CODE_DETECT=0: no new user is routed there (users already there stay, their data is only there).
CODE_DETECT = os.environ.get("AML_CODE_DETECT", "1") == "1"
CODE_CFG = codemem.CodeCfg(**json.loads(os.environ.get("AML_CFG_CODING", "{}")))
# Structure of coding Adds/Searches only (counts, hashes), never content; deleted with the rest. "" turns it off.
CODE_LOG = Path(os.environ.get("AML_CODE_LOG", DB_PATH.with_name("coding_shape.jsonl")))

app = FastAPI(title="AML memory")
store = MemoryStore(CFG)
db_lock = threading.Lock()
user_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)  # serialise add/search per user_id
log_lock = threading.Lock()
mm_users: set[str] = set()              # users whose Adds carried content parts (multimodal track)
route_lock = threading.Lock()           # one coding/not-coding decision per user, also for concurrent first Adds
code_db_lock = threading.Lock()


def note_track(req: dict) -> None:
    if any(isinstance(m.get("content"), list) for m in req.get("messages") or [] if isinstance(m, dict)):
        mm_users.add(req["user_id"])


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS adds (user_id TEXT, request_id TEXT, body TEXT, seq INTEGER PRIMARY KEY AUTOINCREMENT)")
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS adds_req ON adds(user_id, request_id)")
    return db


DB = open_db(DB_PATH)
DB.execute("CREATE TABLE IF NOT EXISTS img_users (user_id TEXT PRIMARY KEY)")
CODE_DB = open_db(DB_PATH.with_name(f"{DB_PATH.stem}_coding{DB_PATH.suffix}"))     # never replayed, read lazily
img_users: set[str] = {u for (u,) in DB.execute("SELECT user_id FROM img_users")}   # served by mmstore
text_users: set[str] = {u for (u,) in DB.execute("SELECT DISTINCT user_id FROM adds")}   # not coding, for good
code_users: set[str] = {u for (u,) in CODE_DB.execute("SELECT DISTINCT user_id FROM adds")} - img_users   # code_store
mmstore = mmmem.MMStore(MM_CFG, DB, db_lock, mmmem.Blobs(BLOB_DIR), MM_OPTS)
code_store = codemem.CodeStore(CODE_DB, code_db_lock, CODE_CFG)
for (uid, body) in DB.execute("SELECT user_id, body FROM adds ORDER BY seq"):
    if uid in img_users:          # loaded lazily by mmstore on its first Search
        continue
    req = json.loads(body)
    store.add(req)
    note_track(req)


def log_line(path: Path, rec: dict) -> None:
    with log_lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()[:12]


def query_stats(req: dict) -> dict:
    """All a diagnostic log keeps of a question: hash and length of its text, number of options."""
    text, opts = codemem.as_text(req.get("query")), req.get("options")
    return {"q_sha": sha12(text), "q_chars": len(text),
            "n_options": len(opts) if isinstance(opts, (list, dict)) else int(opts is not None)}


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
async def health():         # on the event loop: answers even with every worker thread in a 429 backoff
    return {"status": "ok"}


@app.post("/add", dependencies=[Depends(auth)])
def add(req: dict = Body(...)):
    if not isinstance(req, dict) or not req.get("user_id") or not isinstance(req.get("messages"), list):
        raise HTTPException(422, "user_id and messages are required")
    if req["user_id"] in img_users or mmmem.has_image(req):
        return mm_add(req)
    if routes_to_code(req):
        return code_add(req)
    rid = req.get("request_id")
    # Embed first (fills the cache) so the following search does not pay for it.
    texts = []
    for m in req["messages"]:
        c = m.get("content", "")
        if isinstance(c, list):
            c = "\n".join(p.get("text", "") for p in c if p.get("type") == "text")
        c = feedback_view(c, CFG.stream_fb)      # what store.add keeps: a message it drops is never read, so never billed
        if c is not None:
            texts += [f"{m.get('role', 'user')}: {p}" for p in passages(c)]
    if texts:
        dashscope.embed(texts, "document")
    with db_lock:
        if rid and DB.execute("SELECT 1 FROM adds WHERE user_id=? AND request_id=?", (req["user_id"], rid)).fetchone():
            return {"success": True, "request_id": rid, "user_id": req["user_id"], "session_id": req.get("session_id")}
        DB.execute("INSERT INTO adds (user_id, request_id, body) VALUES (?,?,?)",
                   (req["user_id"], rid, json.dumps(req, ensure_ascii=False)))
        DB.commit()
    note_track(req)
    with user_locks[req["user_id"]]:
        return store.add(req)


@app.post("/search", dependencies=[Depends(auth)])
def search(req: dict = Body(...)):
    if isinstance(req, dict) and is_code(req.get("user_id")):
        return code_search(req)
    if not isinstance(req, dict) or not req.get("user_id") or req.get("query") in (None, ""):
        raise HTTPException(422, "user_id and query are required")
    if req["user_id"] in img_users:
        return mm_search(req)
    top_k = int(req.get("top_k") or 100)
    t0 = time.time()
    with user_locks[req["user_id"]]:
        res = store.search({**req, "top_k": top_k})
    data = res["data"][:top_k]
    mm = req["user_id"] in mm_users
    q = req["query"]
    rec = {"ts": round(t0, 3), "user_id": req["user_id"], "mm": mm, "top_k": top_k, **query_stats(req),
           "query_images": 0 if isinstance(q, str) else sum(1 for p in q if isinstance(p, dict) and p.get("type") == "image_url"),
           "n": len(data), "chars": sum(len(d["content"]) for d in data), "sec": round(time.time() - t0, 3)}
    log_line(SEARCH_LOG, rec)
    return JSONResponse({"data": [{"id": d["id"], "content": [{"type": "text", "text": d["content"]}] if mm else d["content"],
                                   "score": d["score"], "created_at": d["created_at"]} for d in data]})


# ------------------------------------------------------------------------------------------ v1.1 multimodal path
def _shape(msgs: list) -> str:
    """Part-type layout of an Add, e.g. 'T|TI|I' (no content): for diagnosing the platform's caption placement."""
    out = []
    for m in msgs:
        c = m.get("content") if isinstance(m, dict) else None
        out.append("S" if isinstance(c, str) else "".join({"text": "T", "image_url": "I"}.get(p.get("type"), "?")
                                                          for p in c if isinstance(p, dict)) if isinstance(c, list) else "-")
    return "|".join(out)[:300]


def mm_add(req: dict):
    uid, rid = req["user_id"], req.get("request_id")
    try:
        msgs = [mmmem.strip_message(m, mmstore.blobs) for m in req["messages"]]   # images -> blobs, base64 never persisted
    except mmmem.BadPayload as e:
        raise HTTPException(422, str(e))
    texts = mmstore.doc_texts(msgs)
    if texts:
        dashscope.embed(texts, "document")         # same strings the index embeds: Search then hits the cache
    moved = code_store.rows(uid) if uid in code_users else []    # a first image after the route to code: its Adds follow
    with db_lock:
        DB.execute("INSERT OR IGNORE INTO img_users (user_id) VALUES (?)", (uid,))
        DB.executemany("INSERT OR IGNORE INTO adds (user_id, request_id, body) VALUES (?,?,?)", [(uid, r, b) for r, b in moved])
        if not (rid and DB.execute("SELECT 1 FROM adds WHERE user_id=? AND request_id=?", (uid, rid)).fetchone()):
            DB.execute("INSERT INTO adds (user_id, request_id, body) VALUES (?,?,?)",
                       (uid, rid, json.dumps({**req, "messages": msgs}, ensure_ascii=False)))
        DB.commit()
        img_users.add(uid)
    if uid in code_users:         # after the commit: a crash in between leaves stale rows, not a lost user
        code_users.discard(uid)
        code_store.forget(uid)
    store.users.pop(uid, None)    # text-path copy of Adds that came before its first image (SQLite has them all)
    log_line(ADD_LOG, {"ts": round(time.time(), 3), "user_id": uid, "shape": _shape(req["messages"]),
                       "ts_set": sum(1 for m in req["messages"] if isinstance(m, dict) and m.get("timestamp") is not None),
                       "img_bytes": sum(p["bytes"] for m in msgs if isinstance(m.get("content"), list)
                                        for p in m["content"] if p.get("type") == "image_ref")})
    return {"success": True, "request_id": rid, "user_id": uid, "session_id": req.get("session_id")}


def mm_search(req: dict):
    t0 = time.time()
    try:
        top_k = int(req.get("top_k") or 100)
    except (TypeError, ValueError):
        raise HTTPException(422, "top_k must be an integer")
    try:
        with user_locks[req["user_id"]]:
            data = mmstore.search_parts({**req, "top_k": top_k})[:top_k]
    except mmmem.BadPayload as e:
        raise HTTPException(422, str(e))
    store.users.pop(req["user_id"], None)
    q = req["query"]
    imgs = [p for d in data for p in d["content"] if p["type"] == "image_url"]
    log_line(SEARCH_LOG, {
        "ts": round(t0, 3), "user_id": req["user_id"], "mm": "img", "top_k": top_k, **query_stats(req),
        "query_images": 0 if isinstance(q, str) else sum(1 for p in q if p.get("type") == "image_url"),
        "n": len(data), "n_img": len(imgs),
        "img_b64": sum(len(p["image_url"]["url"]) for p in imgs),
        "chars": sum(len(p["text"]) for d in data for p in d["content"] if p["type"] == "text"),
        "resident": mmstore.resident(), "sec": round(time.time() - t0, 3)})
    return JSONResponse({"data": data})


# ---------------------------------------------------------------------------------------------- v1.1 coding path
def is_code(uid) -> bool:
    return isinstance(uid, str) and uid in code_users


def routes_to_code(req: dict) -> bool:
    """Decided once per user, on its first Add with messages; image users have already gone to mm_add."""
    uid = req["user_id"]
    if is_code(uid):
        return True
    if not CODE_DETECT or not isinstance(uid, str) or uid in text_users or not req["messages"]:
        return False
    hit = codemem.is_trajectory_add(req)
    with route_lock:
        if uid not in code_users and uid not in text_users:
            (code_users if hit else text_users).add(uid)
        return uid in code_users


def _uh(v) -> str | None:          # opaque platform id -> short hash (lets a Smoke count Adds/Searches per unit)
    return None if v is None else sha12(str(v))


def _name(v) -> str:               # a role or field name as sent; anything else may be free text: hash only
    s = str(v)
    return s if re.fullmatch(r"[A-Za-z_][\w.-]{0,31}", s, re.ASCII) else sha12(s)


def code_log(rec: dict) -> None:
    if str(CODE_LOG) not in ("", "."):
        try:
            log_line(CODE_LOG, rec)
        except (OSError, ValueError):   # diagnostics must not fail a request
            pass


def code_add(req: dict):
    t0 = time.time()
    out = code_store.add(req)
    msgs = req["messages"]
    roles: dict[str, int] = defaultdict(int)
    for m in msgs:
        roles[_name(m.get("role")) if isinstance(m, dict) else type(m).__name__] += 1
    lens = [len(codemem.as_text(m.get("content"))) for m in msgs if isinstance(m, dict)]
    code_log({"op": "add", "ts": round(t0, 3), "u": _uh(req["user_id"]), "sid": _uh(req.get("session_id")),
              "n_msgs": len(msgs), "roles": roles, "chars": sum(lens), "max_chars": max(lens, default=0),
              "list_content": sum(1 for m in msgs if isinstance(m, dict) and isinstance(m.get("content"), list)),
              "has_ts": sum(1 for m in msgs if isinstance(m, dict) and m.get("timestamp") is not None),
              "extra_keys": sorted(_name(k) for k in set(req) - {"user_id", "request_id", "session_id", "messages"})[:16],
              "sec": round(time.time() - t0, 3)})
    return out


def code_search(req: dict):
    """Any internal failure degrades to "no memory": an agent without memory beats a failed task."""
    t0 = time.time()
    err = None
    try:
        res = code_store.search(req)
        out = JSONResponse(res)
    except Exception as e:
        err, res = type(e).__name__, {"data": []}
        out = JSONResponse(res)
    q, top_k = req.get("query"), req.get("top_k")
    code_log({"op": "search", "ts": round(t0, 3), "u": _uh(req["user_id"]), "q_type": type(q).__name__,
              "top_k": top_k if isinstance(top_k, int) else type(top_k).__name__, **query_stats(req),
              "n": len(res["data"]), "chars": sum(len(d["content"]) for d in res["data"]),
              "extra_keys": sorted(_name(k) for k in set(req) - {"user_id", "query", "top_k", "options"})[:16],
              "error": err, "sec": round(time.time() - t0, 3)})
    return out
