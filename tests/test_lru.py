"""Offline checks of the per-user index LRU in memsys (Config.max_indexed, on in DEFAULT_CFG). dashscope is a
stub (deterministic embed/rerank, chat raises), every non-local socket connect is refused and counted.

  * policy: least recently added/searched goes first, never a user in use or being added to
  * equiv: MemoryStore level, every Search response of v1.0.3 == cap off == small caps (frequent evictions), byte for
    byte, for DEFAULT_CFG and for a config with every option on
  * stream: threads taking turns over users with interleaved Adds and Searches, so users are evicted between their own
    Searches and rebuilt: each user's responses == its v1.0.3 responses
  * server: HTTP level, raw /search bytes of the v1.0.3 server == this server, at the default cap and at cap 1, after a
    restart and with concurrent clients
  * concurrency: same users searched concurrently while others are added, searched and evicted; negative control: the
    same without the in_use guard must fail
  * dashscope: the real dashscope.embed on all cache hits: no budget check, no usage-log line, also through evictions
    (child process: from-scratch environment, temp HOME and cache, import-time _key() cut out, so no key is ever read)

Run: python tests/test_lru.py [policy equiv stream server concurrency dashscope]   (default: all)
"""
from __future__ import annotations

import ast
import collections
import concurrent.futures as cf
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# a child process and a moved temp dir must find tiktoken's local copy (a miss would download)
TIKTOKEN = os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(tempfile.gettempdir()) / "data-gym-cache"))
import test_multimodal_v11 as mm           # offline guard, dashscope stub module

ROOT, SVC = mm.ROOT, mm.SVC
PY = sys.executable


# ------------------------------------------------------------------------------------------- stub dashscope
D = 1024
_WMAT = np.random.default_rng(12345).standard_normal((8192, D)).astype(np.float32)
STUB = {"embed_calls": 0, "embed_texts": 0, "query_delay": 0.0, "embed_sec": 0.0}
_stub_lock = threading.Lock()


def stub_embed(texts, kind="document", workers=8):
    """Bag-of-words dense (fp16-rounded like the cache) + up to 70 sparse terms; deterministic in (text, kind)."""
    t0 = time.perf_counter()
    if kind == "query" and STUB["query_delay"]:
        time.sleep(STUB["query_delay"])
    texts = [t[:24000] if t.strip() else "(empty)" for t in texts]
    dense = np.empty((len(texts), D), np.float32)
    sparse = []
    for i, t in enumerate(texts):
        ws = re.findall(r"\w+", t.lower()) or ["empty"]
        hs = [zlib.crc32(w.encode()) for w in ws]
        v = _WMAT[[h & 8191 for h in hs]].sum(0)
        v[zlib.crc32(f"{kind}|{t}".encode()) % D] += 0.5
        dense[i] = (v / (np.linalg.norm(v) + 1e-9)).astype(np.float16).astype(np.float32)
        tf: dict = {}
        for h in hs:
            tf[h] = tf.get(h, 0) + 1
        sparse.append({str(h % 250002): round(0.02 + ((h >> 9) % 1000) / 1000.0 * (1 + math.log(c)), 6)
                       for h, c in list(tf.items())[:70]})
    with _stub_lock:
        STUB["embed_calls"] += 1
        STUB["embed_texts"] += len(texts)
        STUB["embed_sec"] += time.perf_counter() - t0
    return dense, sparse


def stub_rerank(query, docs, model="qwen3-rerank"):
    return [((zlib.crc32(f"{query}|{d}".encode()) % 10007) / 10007.0) for d in docs]


mm.stub.EMB_DIM, mm.stub.embed, mm.stub.rerank = D, stub_embed, stub_rerank


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def real_dashscope(svc: Path) -> types.ModuleType:
    """service/dashscope.py as is, minus its import-time _key() call: no key file, no key variable is ever read."""
    src = (svc / "dashscope.py").read_text()
    assert src.count("{_key()}") == 1
    m = types.ModuleType("dashscope")
    m.__file__ = str(svc / "dashscope.py")
    sys.modules["dashscope"] = m
    exec(compile(src.replace("{_key()}", "offline-no-key"), m.__file__, "exec"), m.__dict__)
    return m


def load_memsys(svc: Path, name: str):
    if str(svc) not in sys.path:
        sys.path.insert(0, str(svc))          # timeannot (same file in both copies)
    mod = load_module(svc / "memsys.py", name)
    assert mod.dashscope is mm.stub, "memsys must see the stub dashscope"
    return mod


def default_cfg() -> dict:
    tree = ast.parse((SVC / "server.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "DEFAULT_CFG":
            return ast.literal_eval(node.value)
    raise KeyError("DEFAULT_CFG")


def v103_cfg(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k != "max_indexed"}      # no such field in v1.0.3


# ------------------------------------------------------------------------------------------- workload
SYL = ["ba", "ko", "ri", "tu", "me", "la", "shi", "no", "ve", "da", "pi", "zo", "ga", "mu", "te", "rho", "xi", "qua"]
TOPICS = ["coffee", "tokyo", "guitar", "marathon", "sourdough", "python", "museum", "budget", "sister", "garden",
          "violin", "kayak", "vinyl", "yoga", "chess", "paris", "bakery", "camera", "novel", "hiking"]


def vocab(n: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    out = set()
    while len(out) < n:
        out.add("".join(rng.choice(SYL) for _ in range(rng.randint(2, 4))))
    return sorted(out)


VOCAB = vocab(4000)


def sentence(rng: random.Random, n: int) -> str:
    ws = []
    for _ in range(n):
        r = rng.random()
        ws.append(rng.choice(TOPICS) if r < 0.06 else VOCAB[min(int(rng.paretovariate(1.1)) - 1, len(VOCAB) - 1)]
                  if r < 0.5 else rng.choice(VOCAB))
    return " ".join(ws).capitalize() + "."


def session_msgs(rng: random.Random, t0: int | None, n: int, special: bool) -> list[dict]:
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        r = rng.random()
        if r < 0.08:
            content = sentence(rng, rng.randint(380, 900))            # several passages
        elif special and role == "user" and r < 0.13:
            content = f"Please forget that I {rng.choice(['like', 'own', 'visited'])} the {rng.choice(TOPICS)} place."
        elif special and role == "user" and r < 0.18:
            content = f"Always remind me about {rng.choice(TOPICS)} whenever I ask about {rng.choice(TOPICS)}."
        elif special and r < 0.2:
            content = ("Streaming online QA feedback: " + json.dumps(
                {"question": f"What about {rng.choice(TOPICS)}?", "predicted_answer": sentence(rng, 6),
                 "score": rng.choice([0, 1])}))
        else:
            content = sentence(rng, rng.randint(4, 120))
        m = {"role": role, "content": content}
        if t0 is not None and rng.random() < 0.9:
            m["timestamp"] = t0 + i * 60_000
        msgs.append(m)
    if special and rng.random() < 0.3:
        msgs.insert(0, {"role": "system", "content": f"User profile: likes {rng.choice(TOPICS)} and {rng.choice(TOPICS)}."})
    return msgs


def queries(rng: random.Random) -> list:
    a, b = rng.sample(TOPICS, 2)
    return [f"What did I say about {a}?",
            f"Now is 2023-07-01. What did I do about {b} in the past two weeks?",
            f"How many times did I mention {a}?",
            f"Can you recommend something for my {b}?",
            f"Please summarize how my {a} plans evolved over time.",
            [{"type": "text", "text": f"Which {b} did I like?"}],
            f"Did I prefer {a} or {b}?"]


def workload(seed: int, n_users: int, sessions: tuple, special: bool = True) -> list[tuple]:
    """Ops: streaming-style interleaved Add/Search for the first half of the users, then batch Adds, then Searches
    of every user in shuffled order (twice)."""
    rng = random.Random(seed)
    users = [f"u{seed}_{k}" for k in range(n_users)]
    adds = {u: [] for u in users}
    for k, u in enumerate(users):
        t = 1_680_000_000_000 + k * 3_600_000
        for s in range(rng.randint(*sessions)):
            t0 = None if rng.random() < 0.1 else t + s * 86_400_000
            msgs = session_msgs(rng, t0, rng.randint(1, 18), special)
            adds[u].append({"request_id": f"{u}-r{s}", "user_id": u, "session_id": f"{u}-s{s}", "messages": msgs})
    qs = {u: queries(rng) for u in users}

    def search(u, q, k):
        body = {"user_id": u, "query": q, "top_k": k}
        if isinstance(q, str) and q.startswith("Did I prefer"):
            body["options"] = [w for w in TOPICS if w in q] + ["neither"]
        return ("search", body)

    ops = []
    stream = users[: n_users // 2]
    pending = {u: list(adds[u]) for u in stream}
    while any(pending.values()):
        u = rng.choice([x for x in stream if pending[x]])
        ops.append(("add", pending[u].pop(0)))
        if rng.random() < 0.6:
            ops.append(search(u, rng.choice(qs[u]), rng.choice([5, 100])))
        if rng.random() < 0.15:                                  # a repeated request id (idempotent Add)
            ops.append(("add", adds[u][0]))
    for u in users[n_users // 2:]:
        for a in adds[u]:
            ops.append(("add", a))
    ops.append(("search", {"user_id": "nobody", "query": "anything?", "top_k": 10}))
    for _ in range(2):
        order = [(u, q) for u in users for q in qs[u]]
        rng.shuffle(order)
        ops += [search(u, q, rng.choice([1, 7, 100])) for u, q in order]
    return ops


def dumps(res) -> bytes:
    return json.dumps(res, ensure_ascii=False, separators=(",", ":"),
                      default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode()


def run_store(store, ops) -> list[bytes]:
    out = []
    for kind, req in ops:
        if kind == "add":
            store.add(json.loads(json.dumps(req)))
        else:
            out.append(dumps(store.search(json.loads(json.dumps(req)))))
    return out


def indexed(store) -> int:
    return sum(len(u.p_owner) for u in store.users.values())


def per_user(ops) -> dict:
    """Each user's ops in order. Users are independent, so this order alone fixes its responses."""
    out: dict = {}
    for kind, req in ops:
        out.setdefault(req["user_id"], []).append((kind, req))
    return out


def by_user(ops, bodies: list) -> dict:
    """Responses of a sequential run (in op order) grouped per user."""
    it, out = iter(bodies), {}
    for kind, req in ops:
        if kind == "search":
            out.setdefault(req["user_id"], []).append(next(it))
    return out


def concurrent_users(ops, run_op, threads: int = 8, turn: int = 3) -> dict:
    """`threads` workers take turns over the users, `turn` ops at a time. One worker per user at a time and its ops stay
    in order (what server.user_locks and a sequential client give), while a user waiting for its next turn is idle, so
    it can be evicted between its own Searches. run_op(kind, req) returns the /search bytes or None."""
    users = per_user(ops)
    todo = collections.deque((uid, 0) for uid in users)
    lock = threading.Lock()
    got: dict = {uid: [] for uid in users}

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                uid, pos = todo.popleft()
            res = [run_op(kind, req) for kind, req in users[uid][pos:pos + turn]]
            got[uid] += [r for r in res if r is not None]
            if pos + turn < len(users[uid]):
                with lock:
                    todo.append((uid, pos + turn))

    with cf.ThreadPoolExecutor(threads) as ex:
        for f in [ex.submit(worker) for _ in range(threads)]:
            f.result()
    return got


# ------------------------------------------------------------------------------------------- counters on the LRU copy
def instrument(ms) -> dict:
    """Count rebuilds (and from-scratch ones) and drop_index calls; track the indexed total after each Search."""
    c = {"rebuild": 0, "scratch": 0, "drop": 0, "max_after_search": 0}
    rebuild, drop = ms.UserMem.rebuild, ms.UserMem.drop_index
    lock = threading.Lock()

    def rb(self):
        with lock:
            c["rebuild"] += 1
            c["scratch"] += self.n_indexed == 0
        return rebuild(self)

    def dr(self):
        with lock:
            c["drop"] += 1
        return drop(self)

    ms.UserMem.rebuild, ms.UserMem.drop_index = rb, dr
    return c


EVERY = {"rerank_top": 30, "pin_forget": True, "forget_suppress": 0.35, "pin_system": True, "pin_rules": True,
         "whole_auto": True, "whole_budget": 9000, "opt_recall": 0.5, "opt_floor": 3, "digest_k": 8,
         "trunc_long": 60, "header_ago": True, "natural_date": True, "time_annot": True, "time_annot_v": 2,
         "w_session": 0.5, "session_cap": 2, "recency": 0.3, "seed_k": 5, "win": 1, "stream_fb": "pos",
         "time_boost": 1.0, "future_penalty": 3.0, "clean_query": True, "budget": 6000,
         "route": {"advice": {"digest_k": 20}, "count": {"pack": "session"}}}


def test_equiv(base: Path) -> None:
    mb = load_memsys(base, "memsys_base")
    ml = load_memsys(SVC, "memsys_lru")
    cnt = instrument(ml)
    for name, cfg in (("DEFAULT_CFG", default_cfg()), ("every option", EVERY)):
        ops = workload(2, 24, (1, 5))
        ref = run_store(mb.MemoryStore(mb.Config(**v103_cfg(cfg))), ops)
        s0 = ml.MemoryStore(ml.Config(**{**cfg, "max_indexed": 0}))
        assert run_store(s0, ops) == ref, f"{name}: max_indexed=0 differs from v1.0.3"
        total = indexed(s0)
        assert not s0.recent and all(u.in_use == 0 for u in s0.users.values()), "LRU bookkeeping ran while off"
        line = [f"{name:12s} searches={len(ref):4d} passages={total:5d} | off: same"]
        for cap in (max(1, total // 3), 1):
            for k in cnt:
                cnt[k] = 0
            s = ml.MemoryStore(ml.Config(**{**cfg, "max_indexed": cap}))
            out, over = [], []
            for kind, req in ops:
                if kind == "add":
                    s.add(json.loads(json.dumps(req)))
                    continue
                out.append(dumps(s.search(json.loads(json.dumps(req)))))
                # sequential: past the cap only if one user alone is larger than it (then nobody else is indexed)
                resident = [len(u.p_owner) for u in s.users.values() if u.p_owner]
                if sum(resident) > cap and len(resident) > 1:
                    over.append((len(out), resident))
            bad = [i for i, (a, b) in enumerate(zip(ref, out)) if a != b]
            assert len(out) == len(ref) and not bad, f"{name} cap={cap}: {len(bad)} responses differ, first {bad[:3]}"
            assert not over, f"cap={cap}: indexed total over the cap with several users resident: {over[:2]}"
            assert cnt["drop"] > 0, cnt
            line.append(f"cap={cap:5d}: same, {cnt['drop']:4d} evictions, {cnt['scratch']:4d} scratch rebuilds")
        print(" | ".join(line))
    print("equiv: all Search responses byte-identical to v1.0.3")


def test_policy(base: Path) -> None:
    ml = load_memsys(SVC, "memsys_lru")
    cfg = default_cfg()
    assert cfg["max_indexed"] > 0, "the cap must be on by default"
    rng = random.Random(1)
    rid = [0]

    def add(s, uid, n=6):
        rid[0] += 1
        return s.add({"request_id": f"r{rid[0]}", "user_id": uid, "session_id": uid, "timestamp": 0,
                      "messages": [{"role": "user", "content": sentence(rng, 20)} for _ in range(n)]})

    def find(s, uid):
        return s.search({"user_id": uid, "query": "What about coffee?", "top_k": 5})

    def live(s):
        return sorted(k for k, u in s.users.items() if u.p_owner)

    s = ml.MemoryStore(ml.Config(**{**cfg, "max_indexed": 12}))         # room for two users of 6 passages
    for u in "abcd":
        add(s, u)
    find(s, "a"), find(s, "b")
    assert live(s) == ["a", "b"], live(s)                               # exactly at the cap: nothing goes
    a = s.users["a"]
    n_msgs, reqs = len(a.msgs), set(a.seen_requests)
    a.forgets, a.rules = [(0, "x")], [0]
    find(s, "c")
    assert live(s) == ["b", "c"], live(s)                               # the least recently used goes
    assert (a.p_msg, a.dense, a.sparse_inv, a.bm25, a.sessions, a.forget_vecs) == (None,) * 6
    assert (a.p_owner, a.p_toks, a.sparse_lists, a.forgets, a.rules, a.n_indexed, a.dirty) == ([], [], {}, [], [], 0, True)
    assert len(a.msgs) == n_msgs and a.seen_requests == reqs and n_msgs == 6     # the messages stay
    # an Add counts as use (even one that adds nothing)
    add(s, "b", 0)
    find(s, "d")
    assert live(s) == ["b", "d"], live(s)
    # so does a Search that needs no rebuild; recency order is now d, then b
    find(s, "b")
    # a user with a Search in flight is passed over for the next one
    s.users["d"].in_use = 1
    find(s, "a")
    assert live(s) == ["a", "d"], live(s)
    s.users["d"].in_use = 0
    find(s, "c")                                                         # d is older than a: d goes now
    assert live(s) == ["a", "c"], live(s)
    # a user held by an Add (or anyone else holding user.lock) is passed over too
    with s.users["a"].lock:
        find(s, "b")
    assert live(s) == ["a", "b"], live(s)
    # users the caller removed (server.py: a user turning multimodal) leave no trace
    s.users.pop("a")
    find(s, "c"), find(s, "d")
    assert "a" not in s.recent, list(s.recent)
    # one user larger than the cap: it stays, alone
    s2 = ml.MemoryStore(ml.Config(**{**cfg, "max_indexed": 3}))
    add(s2, "x", 2), add(s2, "big", 8)
    find(s2, "x"), find(s2, "big")
    assert live(s2) == ["big"], live(s2)
    # off: no order kept, nothing evicted
    s3 = ml.MemoryStore(ml.Config(**{**cfg, "max_indexed": 0}))
    assert s3.cfg.max_indexed == 0
    for u in "abc":
        add(s3, u), find(s3, u)
    assert live(s3) == ["a", "b", "c"] and not s3.recent
    print("policy: least recently used goes first, users in use or being added to stay, a too large user stays alone")


def test_stream(base: Path) -> None:
    mb = load_memsys(base, "memsys_base")
    ml = load_memsys(SVC, "memsys_lru")
    cnt = instrument(ml)
    ops = workload(21, 24, (2, 7)) + workload(22, 8, (1, 3))
    for name, cfg in (("DEFAULT_CFG", default_cfg()), ("every option", EVERY)):
        ref_store = mb.MemoryStore(mb.Config(**v103_cfg(cfg)))
        ref = by_user(ops, run_store(ref_store, ops))
        total = indexed(ref_store)
        for cap in (total // 4, 1):
            for k in cnt:
                cnt[k] = 0
            s = ml.MemoryStore(ml.Config(**{**cfg, "max_indexed": cap}))

            def op(kind, req):
                req = json.loads(json.dumps(req))
                if kind == "add":
                    s.add(req)
                    return None
                return dumps(s.search(req))

            got = concurrent_users(ops, op)
            bad = [u for u in ref if got.get(u) != ref[u]]
            print(f"stream {name:12s} users={len(ref)} searches={sum(map(len, ref.values()))} cap={cap:5d}: "
                  f"{len(bad)} users differ, evictions={cnt['drop']} scratch rebuilds={cnt['scratch']}")
            assert not bad, bad[:3]
            assert cnt["scratch"] > len(ref), cnt                 # users really were evicted and rebuilt mid-stream
            assert all(u.in_use == 0 for u in s.users.values())
    print("stream: every user's responses byte-identical to v1.0.3 under concurrent eviction")


def test_server(base: Path) -> None:
    from fastapi.testclient import TestClient
    work = Path(tempfile.mkdtemp(prefix="lru_srv_"))
    mb = load_memsys(base, "memsys_base")
    ml = load_memsys(SVC, "memsys_lru")
    cnt = instrument(ml)
    os.environ.update(AML_TOKEN="t0k")
    H = {"Authorization": "Token t0k"}

    def server(src: Path, mem, name: str, cfg: dict | None):
        os.environ["AML_DB"] = str(work / name / "mem.sqlite")
        for k in ("AML_SEARCH_LOG", "AML_ADD_LOG", "AML_BLOBS", "AML_CODE_LOG", "AML_CFG_MM", "AML_CFG_CODING"):
            os.environ.pop(k, None)
        os.environ["AML_SEARCH_LOG"] = str(work / name / "search_log.jsonl")
        os.environ["AML_CFG"] = json.dumps(cfg or {})
        sys.modules["memsys"] = mem
        m = load_module(src / "server.py", "server_" + name)
        assert m.MemoryStore is mem.MemoryStore and m.dashscope is mm.stub
        return TestClient(m.app), m

    ops = workload(5, 16, (2, 8))
    results = {}
    for name, src, mem, cfg in (("v103", base, mb, None), ("v11", SVC, ml, None), ("cap1", SVC, ml, {"max_indexed": 1})):
        for k in cnt:
            cnt[k] = 0
        client, mod = server(src, mem, name, cfg)
        if name == "v11":
            assert mod.store.cfg.max_indexed == default_cfg()["max_indexed"] > 0 == mod.MM_CFG.max_indexed, "default caps"
        out = []
        for kind, req in ops:
            r = client.post("/" + kind, json=req, headers=H)
            assert r.status_code == 200, (name, r.status_code, r.text[:200])
            if kind == "search":
                out.append(r.content)
        results[name] = (out, dict(cnt))
    ref = results["v103"][0]
    for name, (out, c) in results.items():
        print(f"server {name:5s}: {len(out)} /search bodies, byte-identical to v1.0.3: {out == ref}, "
              f"evictions={c['drop']}, scratch rebuilds={c['scratch']}")
        assert out == ref, name
    assert results["cap1"][1]["drop"] > 0
    # restart at cap 1: SQLite replay, nothing indexed, same bytes for the last Searches of the run above
    client, mod = server(SVC, ml, "cap1", {"max_indexed": 1})
    tail = [req for kind, req in ops if kind == "search"][-40:]
    assert [client.post("/search", json=q, headers=H).content for q in tail] == ref[-40:], "restart at cap 1 differs"
    ref_u = by_user(ops, ref)
    for k in cnt:
        cnt[k] = 0
    client, mod = server(SVC, ml, "par1", {"max_indexed": 1})     # concurrent clients, as a real evaluation

    def op(kind, req):
        r = client.post("/" + kind, json=req, headers=H)
        assert r.status_code == 200, (r.status_code, r.text[:200])
        return r.content if kind == "search" else None

    got = concurrent_users(ops, op)
    bad = [u for u in ref_u if got.get(u) != ref_u[u]]
    print(f"server par1 : concurrent clients, {len(bad)} users differ from v1.0.3, evictions={cnt['drop']}, "
          f"scratch rebuilds={cnt['scratch']}")
    assert not bad and cnt["scratch"] > len(ref_u), (bad[:3], cnt)
    shutil.rmtree(work, ignore_errors=True)
    print("server: /search bytes of v1.1 == v1.0.3 at the default cap and at cap 1, after a restart, with concurrent clients")


def _concurrency(base: Path, guard: bool, n_threads: int = 16, iters: int = 40) -> tuple[int, int, dict]:
    mb = load_memsys(base, "memsys_base")
    ml = load_memsys(SVC, "memsys_lru")
    cnt = instrument(ml)
    cfg = default_cfg()
    ops = workload(9, 20, (2, 8))
    adds = [r for k, r in ops if k == "add"]
    reqs = list({json.dumps(r, sort_keys=True): r for k, r in ops if k == "search"}.values())
    ref_store = mb.MemoryStore(mb.Config(**v103_cfg(cfg)))
    for a in adds:
        ref_store.add(json.loads(json.dumps(a)))
    ref = {json.dumps(r, sort_keys=True): dumps(ref_store.search(json.loads(json.dumps(r)))) for r in reqs}
    total = indexed(ref_store)

    store = ml.MemoryStore(ml.Config(**{**cfg, "max_indexed": total // 4}))
    if not guard:                                  # negative control: evict() without the in_use check
        def evict_unguarded(self=store):
            with self.lock:
                tot = sum(len(u.p_owner) for u in list(self.users.values()))
                for uid in list(self.recent):
                    if tot <= self.cfg.max_indexed:
                        break
                    u = self.users.get(uid)
                    if u is not None and u.p_owner and u.lock.acquire(blocking=False):
                        tot -= len(u.p_owner)
                        u.drop_index()
                        u.lock.release()
        store.evict = evict_unguarded
    for a in adds:
        store.add(json.loads(json.dumps(a)))
    noise = [f"noise{k}" for k in range(8)]
    # Add and Search of one user are serialised by server.user_locks; without that lock v1.0.3 itself fails (KeyError on
    # a session added after the rebuild), LRU or not. Searches of the static users stay unserialised on purpose.
    noise_lock = {u: threading.Lock() for u in noise}
    errors, mismatches = [], [0]
    STUB["query_delay"] = 0.002                    # widen the window between rebuild and index reads
    for k in cnt:
        cnt[k] = 0

    def worker(seed):
        rng = random.Random(seed)
        for i in range(iters):
            try:
                r = rng.random()
                if r < 0.7:
                    req = rng.choice(reqs)
                    got = dumps(store.search(json.loads(json.dumps(req))))
                    if got != ref[json.dumps(req, sort_keys=True)]:
                        mismatches[0] += 1
                elif r < 0.85:
                    u = rng.choice(noise)
                    with noise_lock[u]:
                        store.add({"request_id": f"{u}-{seed}-{i}", "user_id": u, "session_id": f"{u}-{seed}",
                                   "messages": session_msgs(rng, 1_690_000_000_000 + i, rng.randint(1, 30), True)})
                else:
                    u = rng.choice(noise)
                    with noise_lock[u]:
                        store.search({"user_id": u, "query": f"What about {rng.choice(TOPICS)}?", "top_k": 50})
            except Exception as e:                 # noqa: BLE001 - every failure is a finding here
                errors.append(f"{type(e).__name__}: {e}")

    t0 = time.time()
    ths = [threading.Thread(target=worker, args=(s,)) for s in range(n_threads)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    STUB["query_delay"] = 0.0
    sec = time.time() - t0
    for uid, u in store.users.items():             # invariants once quiet
        assert u.in_use == 0, uid
        if u.p_owner:
            assert u.dense.shape[0] == len(u.p_owner) == len(u.p_toks) == len(u.p_msg), uid
            assert u.n_indexed <= len(u.msgs) and (u.dirty or u.n_indexed == len(u.msgs)), uid
        elif not guard and u.dense is not None:
            continue                               # the unguarded evict left a half-built user: that is the point
        else:
            assert u.dirty and u.dense is None and u.n_indexed == 0, uid
    info = {**cnt, "sec": round(sec, 1), "first_errors": errors[:3]}
    return len(errors), mismatches[0], info


def test_concurrency(base: Path) -> None:
    n_err, n_bad, info = _concurrency(base, guard=True)
    print(f"concurrency (guarded): errors={n_err} mismatches={n_bad} evictions={info['drop']} "
          f"scratch rebuilds={info['scratch']} in {info['sec']} s")
    assert n_err == 0 and n_bad == 0, info
    assert info["drop"] > 20, info
    n_err, n_bad, info = _concurrency(base, guard=False)
    print(f"negative control (no in_use guard): errors={n_err} mismatches={n_bad} evictions={info['drop']}")
    assert n_err + n_bad > 0, "the race was not observed: this test would not catch a missing guard"
    print("concurrency: guarded eviction is invisible to Searches, unguarded eviction breaks them")


# ------------------------------------------------------------------------------------------- child process
def clean_env(tmp: Path, **extra) -> dict:
    """Nothing from the parent environment; temp HOME (no ~/.aml_env), temp cache/log, budget 0."""
    return {"PATH": "/usr/bin:/bin", "HOME": str(tmp), "LANG": "C.UTF-8", "OPENBLAS_NUM_THREADS": "4", "TMPDIR": str(tmp),
            "TIKTOKEN_CACHE_DIR": TIKTOKEN,
            "AML_CACHE": str(tmp / "cache" / "emb.sqlite"), "AML_EMB_USAGE_LOG": str(tmp / "emb_usage.log"),
            "AML_EMB_BUDGET_YUAN": "0", **extra}


def child_dashscope(svc: Path, tmp: Path) -> None:
    sys.path.insert(0, str(svc))
    ds = real_dashscope(svc)                        # the real module, key lookup cut out
    assert not mm.NET
    assert ds.BUDGET_YUAN == 0.0 and ds.USAGE_LOG == tmp / "emb_usage.log" and Path.home() == tmp
    calls = {"post": 0, "check": 0, "batch": 0}
    check, batch = ds._check_budget, ds._embed_batch

    def post(*a, **k):
        calls["post"] += 1
        raise RuntimeError("network")

    def chk():
        calls["check"] += 1
        return check()

    def bat(*a, **k):
        calls["batch"] += 1
        return batch(*a, **k)

    ds._post, ds._check_budget, ds._embed_batch = post, chk, bat
    texts = [f"user: message {i} about coffee" for i in range(37)] + ["assistant: " + "x " * 30000, "   "]
    norm = [t[:24000] if t.strip() else "(empty)" for t in texts]
    rng = np.random.default_rng(0)
    rows = []
    for t in norm:
        v = rng.standard_normal(ds.EMB_DIM).astype(np.float32)
        rows.append((ds.CACHE.key(t, "document"), v / np.linalg.norm(v), {"17": 0.25, "4096": 0.5}))
    ds.CACHE.put(rows)
    log = ds.USAGE_LOG
    assert not log.exists()
    # 1) all hits with the budget already exhausted (0 >= 0): no error, no budget check, no log line
    dense, sparse = ds.embed(texts, "document")
    assert dense.shape == (len(texts), ds.EMB_DIM) and sparse[0] == {"17": 0.25, "4096": 0.5}
    assert np.array_equal(dense[0], rows[0][1].astype(np.float16).astype(np.float32))
    assert calls == {"post": 0, "check": 0, "batch": 0} and not log.exists(), calls
    # 2) a spent log far over budget (50 yuan logged, cap 30): still fine, file untouched
    log.write_text("1700000000 100000000\n")
    ds.BUDGET_YUAN, before = 30.0, log.read_bytes()
    ds.embed(texts[:5], "document")
    assert log.read_bytes() == before and calls["check"] == 0
    # 3) one miss: the budget check runs and refuses before any request; nothing logged
    try:
        ds.embed(["user: not cached"], "document")
        raise AssertionError("a miss over budget must raise")
    except RuntimeError as e:
        assert "budget exhausted" in str(e), e
    assert calls == {"post": 0, "check": 1, "batch": 1} and log.read_bytes() == before, calls
    # 4) memsys on this embed(): one pass without a cap fills the cache like /add does (fake paid call), then a pass with
    #    evictions everywhere and the budget exhausted must get every vector (forget targets too) from the cache
    import memsys
    assert memsys.dashscope is ds
    paid = {"texts": 0}

    def fake_batch(ts, kind):
        paid["texts"] += len(ts)
        dn, sp = stub_embed(ts, kind)
        return [(dn[i], sp[i]) for i in range(len(ts))]

    ds._embed_batch = fake_batch
    ops = workload(3, 8, (1, 4))
    cfg = default_cfg()
    first = run_store(memsys.MemoryStore(memsys.Config(**cfg)), ops)
    assert paid["texts"] > 0
    ds._embed_batch, paid["texts"] = bat, 0
    calls.update(check=0, batch=0)
    store = memsys.MemoryStore(memsys.Config(**{**cfg, "max_indexed": 5}))
    drops = [0]
    drop = memsys.UserMem.drop_index

    def counted(self):
        drops[0] += 1
        return drop(self)

    memsys.UserMem.drop_index = counted
    second = run_store(store, ops)
    assert second == first and drops[0] > 20, drops
    assert calls == {"post": 0, "check": 0, "batch": 0} and log.read_bytes() == before, calls
    assert not mm.NET, mm.NET
    print("dashscope: all-hit embed() -> no _check_budget, no _embed_batch, no usage-log write (even over budget); "
          "a miss -> 'budget exhausted' before any request; no socket connect")
    print(f"dashscope: {len(second)} Searches with {drops[0]} evictions, budget exhausted: 0 embed batches, 0 budget "
          f"checks, usage log untouched, responses identical")


def test_dashscope(base: Path) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="lru_ds_"))
    r = subprocess.run([PY, "-I", __file__, "_child_dashscope", str(SVC), str(tmp)], env=clean_env(tmp),
                       capture_output=True, text=True, timeout=600)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr[-3000:])
    shutil.rmtree(tmp, ignore_errors=True)
    assert r.returncode == 0, "dashscope child failed"


TESTS = {"policy": test_policy, "equiv": test_equiv, "stream": test_stream, "server": test_server,
         "concurrency": test_concurrency, "dashscope": test_dashscope}
NEED_BASE = {"equiv", "stream", "server", "concurrency"}


def main() -> None:
    if sys.argv[1:2] == ["_child_dashscope"]:
        return child_dashscope(Path(sys.argv[2]), Path(sys.argv[3]))
    tmp = Path(tempfile.mkdtemp(prefix="lru_"))
    base = tmp / "v103"
    base.mkdir()
    try:
        for f in ("server.py", "memsys.py", "timeannot.py"):
            (base / f).write_bytes(subprocess.run(["git", "-C", str(ROOT), "show", f"v1.0.3:service/{f}"], capture_output=True,
                                                  check=True).stdout)
    except (OSError, subprocess.CalledProcessError):
        print("SKIP v1.0.3 comparisons: git tag v1.0.3 not available")
        base = None
    try:
        for name in sys.argv[1:] or TESTS:
            if base is None and name in NEED_BASE:
                continue
            t0 = time.time()
            TESTS[name](base)
            print(f"[{name}] passed in {time.time() - t0:.1f}s\n", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert not mm.NET, mm.NET
    print("no network calls; all LRU checks passed")


if __name__ == "__main__":
    main()
