"""Thin DashScope client: text-embedding-v4 (dense+sparse, disk-cached), rerank, chat."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

BASE = "https://dashscope.aliyuncs.com"
EMB_MODEL = "text-embedding-v4"
EMB_DIM = 1024
CACHE_PATH = Path(os.environ.get("AML_CACHE", Path(__file__).with_name("cache") / "emb.sqlite"))


def _key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        env = Path.home() / ".aml_env"
        for line in env.read_text().splitlines():
            if line.startswith("DASHSCOPE_API_KEY="):
                key = line.split("=", 1)[1].strip()
    return key


# Local vLLM endpoints (gpu01/gpu02 via ssh tunnels): {"model-name": "http://127.0.0.1:8001/v1", ...}
# A value may be a list of bases (round-robin). Keys may carry a role suffix: "qwen3-14b@ans" is served as "qwen3-14b".
LOCAL = json.loads(os.environ.get("AML_LOCAL_MODELS", "{}"))
_local = httpx.Client(timeout=900, trust_env=False, limits=httpx.Limits(max_connections=128))
_rr_lock = threading.Lock()
_rr_i: dict = {}


def _base(model: str) -> str:
    b = LOCAL[model]
    if isinstance(b, str):
        return b
    with _rr_lock:
        i = _rr_i.get(model, 0)
        _rr_i[model] = i + 1
    return b[i % len(b)]

# Local calls retry for ~37 min (backoff capped at 60 s): the ssh tunnel through mn01 has dropped for 1-15 min
# at a time (2026-10-24), and a run should pause through that rather than record errors. Engine deaths are caught
# by the vLLM-log monitor instead.
LOCAL_ATTEMPTS = 40


def any_base(model: str) -> str:
    """A replica for cheap side calls (/tokenize): picked without advancing the round-robin counter, otherwise a
    tokenize-then-generate pattern sends every generation to the same replica (seen 2026-10-24: b queued 40, a idle)."""
    b = LOCAL[model]
    return b if isinstance(b, str) else b[threading.get_ident() % len(b)]

_client = httpx.Client(timeout=120, trust_env=False, limits=httpx.Limits(max_connections=64), headers={"Authorization": f"Bearer {_key()}"})


def _post(url: str, payload: dict, retries: int = 12) -> dict:  # 12: survives minutes of 429 throttling
    for attempt in range(retries):
        try:
            r = _client.post(url, json=payload)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 60))
                continue
            raise RuntimeError(f"{r.status_code} {r.text[:300]}")
        except httpx.TransportError:
            time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f"giving up on {url}")


def _open_db(path: Path) -> sqlite3.Connection:
    """Cache DB shared by several concurrent bench processes: WAL lets readers proceed while one writes."""
    db = sqlite3.connect(path, check_same_thread=False, timeout=300)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    return db


class EmbeddingCache:
    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = _open_db(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, dense BLOB, sparse TEXT)")
        self.lock = threading.Lock()

    @staticmethod
    def key(text: str, kind: str) -> str:
        return hashlib.sha1(f"{EMB_MODEL}|{EMB_DIM}|{kind}|{text}".encode()).hexdigest()

    def get(self, keys: list[str]) -> dict:
        out = {}
        with self.lock:
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                q = f"SELECT k, dense, sparse FROM emb WHERE k IN ({','.join('?' * len(part))})"
                for k, d, s in self.db.execute(q, part):
                    out[k] = (np.frombuffer(d, dtype=np.float16).astype(np.float32), json.loads(s))
        return out

    def put(self, rows: list[tuple[str, np.ndarray, dict]]) -> None:
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?,?,?)",
                                [(k, d.astype(np.float16).tobytes(), json.dumps(s)) for k, d, s in rows])
            self.db.commit()


CACHE = EmbeddingCache()


# Spend control (2026-10-24: user topped up 100 yuan for everything until submission, official runs included).
# Every billed embedding call appends "<unix time> <tokens>" to USAGE_LOG; calls are refused once the logged
# spend reaches AML_EMB_BUDGET_YUAN.
PRICE_YUAN_PER_M = 0.5            # text-embedding-v4, Beijing region, 0.0005 yuan / 1K tokens
USAGE_LOG = Path(os.environ.get("AML_EMB_USAGE_LOG", CACHE_PATH.with_name("emb_usage.log")))
BUDGET_YUAN = float(os.environ.get("AML_EMB_BUDGET_YUAN", "30"))
_spent = {"t": 0.0, "tokens": 0}
_spent_lock = threading.Lock()


def spent_tokens() -> int:
    try:
        return sum(int(line.split()[1]) for line in USAGE_LOG.read_text().splitlines() if line.strip())
    except FileNotFoundError:
        return 0


def _check_budget() -> None:
    with _spent_lock:
        if time.time() - _spent["t"] > 10:
            _spent.update(t=time.time(), tokens=spent_tokens())
        yuan = _spent["tokens"] * PRICE_YUAN_PER_M / 1e6
    if yuan >= BUDGET_YUAN:
        raise RuntimeError(f"embedding budget exhausted: {yuan:.2f} >= {BUDGET_YUAN} yuan (see {USAGE_LOG})")


def _log_usage(tokens: int) -> None:
    with _spent_lock:
        with open(USAGE_LOG, "a") as f:
            f.write(f"{int(time.time())} {tokens}\n")
        _spent["tokens"] += tokens


def _embed_batch(texts: list[str], kind: str) -> list[tuple[np.ndarray, dict]]:
    _check_budget()
    params = {"dimension": EMB_DIM, "output_type": "dense&sparse"}
    if kind == "query":
        params["text_type"] = "query"
        params["instruct"] = "Given a question about a past conversation, retrieve the messages that help answer it"
    else:
        params["text_type"] = "document"
    data = _post(f"{BASE}/api/v1/services/embeddings/text-embedding/text-embedding",
                 {"model": EMB_MODEL, "input": {"texts": texts}, "parameters": params})
    _log_usage(int(data.get("usage", {}).get("total_tokens", 0)))
    embs = sorted(data["output"]["embeddings"], key=lambda e: e["text_index"])
    out = []
    for e in embs:
        v = np.asarray(e["embedding"], dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-9
        sp = {str(t["index"]): t["value"] for t in e.get("sparse_embedding", [])}
        out.append((v, sp))
    return out


def embed(texts: list[str], kind: str = "document", workers: int = 8) -> tuple[np.ndarray, list[dict]]:
    """Return (N x D dense matrix, list of sparse dicts). kind: document | query."""
    # v4 accepts up to 8192 tokens; clip very long inputs by characters as a guard.
    texts = [t[:24000] if t.strip() else "(empty)" for t in texts]
    keys = [CACHE.key(t, kind) for t in texts]
    have = CACHE.get(list(set(keys)))
    todo = sorted({k: t for k, t in zip(keys, texts) if k not in have}.items())
    batches = [todo[i:i + 10] for i in range(0, len(todo), 10)]

    def run(batch):
        res = _embed_batch([t for _, t in batch], kind)
        rows = [(k, d, s) for (k, _), (d, s) in zip(batch, res)]
        CACHE.put(rows)
        return rows

    if batches:
        with ThreadPoolExecutor(workers) as ex:
            for rows in ex.map(run, batches):
                for k, d, s in rows:
                    have[k] = (d, s)
    dense = np.stack([have[k][0] for k in keys]) if keys else np.zeros((0, EMB_DIM), np.float32)
    return dense, [have[k][1] for k in keys]


_LLM_CACHE = CACHE_PATH.with_name("llm.sqlite")
_ldb = _open_db(_LLM_CACHE)
_ldb.execute("CREATE TABLE IF NOT EXISTS llm (k TEXT PRIMARY KEY, v TEXT)")
_llock = threading.Lock()



RR_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
             "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
RR_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
RR_INSTRUCT = "Given a question about a past conversation, judge whether this excerpt helps answer it"


def _local_rerank(query: str, docs: list[str], model: str) -> list[float]:
    q = f"{RR_PREFIX}<Instruct>: {RR_INSTRUCT}\n<Query>: {query}\n"
    out = []
    for start in range(0, len(docs), 64):
        part = [f"<Document>: {d[:6000]}{RR_SUFFIX}" for d in docs[start:start + 64]]
        r = _local.post(f"{_base(model)}/score", json={"model": model.split("@")[0], "text_1": q, "text_2": part})
        r.raise_for_status()
        out += [x["score"] for x in sorted(r.json()["data"], key=lambda x: x["index"])]
    return out


def rerank(query: str, docs: list[str], model: str = "qwen3-rerank") -> list[float]:
    """Relevance score per doc (same order as docs). Cached per (model, query, doc set)."""
    ck = hashlib.sha1(json.dumps([model, query, docs]).encode()).hexdigest()
    with _llock:
        row = _ldb.execute("SELECT v FROM llm WHERE k=?", ("rr:" + ck,)).fetchone()
    if row:
        return json.loads(row[0])
    if model in LOCAL:
        scores = _local_rerank(query, docs, model)
        with _llock:
            _ldb.execute("INSERT OR REPLACE INTO llm VALUES (?,?)", ("rr:" + ck, json.dumps(scores)))
            _ldb.commit()
        return scores
    scores = [0.0] * len(docs)
    for start in range(0, len(docs), 100):
        part = [d[:12000] for d in docs[start:start + 100]]
        data = _post(f"{BASE}/api/v1/services/rerank/text-rerank/text-rerank",
                     {"model": model, "input": {"query": query, "documents": part},
                      "parameters": {"top_n": len(part), "return_documents": False}})
        for r in data["output"]["results"]:
            scores[start + r["index"]] = r["relevance_score"]
    with _llock:
        _ldb.execute("INSERT OR REPLACE INTO llm VALUES (?,?)", ("rr:" + ck, json.dumps(scores)))
        _ldb.commit()
    return scores


def chat(prompt: str, model: str, max_tokens: int = 512) -> str:
    if model in LOCAL:
        for attempt in range(LOCAL_ATTEMPTS):
            try:
                r = _local.post(f"{_base(model)}/chat/completions", json={
                    "model": model.split("@")[0], "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                    "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}})
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except (httpx.TransportError, httpx.HTTPStatusError):
                time.sleep(min(2 ** attempt, 60))
        raise RuntimeError(f"local model {model} failed")
    data = _post(f"{BASE}/compatible-mode/v1/chat/completions",
                 {"model": model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "max_tokens": max_tokens, "enable_thinking": False})
    return data["choices"][0]["message"]["content"].strip()



def chat_cached(prompt: str, model: str, max_tokens: int = 512) -> str:
    """chat() with a persistent cache so repeated experiments cost nothing."""
    k = hashlib.sha1(f"{model}|{max_tokens}|{prompt}".encode()).hexdigest()
    with _llock:
        row = _ldb.execute("SELECT v FROM llm WHERE k=?", (k,)).fetchone()
    if row:
        return row[0]
    v = chat(prompt, model, max_tokens)
    with _llock:
        _ldb.execute("INSERT OR REPLACE INTO llm VALUES (?,?)", (k, v))
        _ldb.commit()
    return v
