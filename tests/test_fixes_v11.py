"""Offline regression tests of the fixes from the v1.1 review, at least one per fix. Sockets are refused and counted.
dashscope is the stub of test_multimodal_v11, except where the real embed() is the thing under test: there a child
process runs service/dashscope.py minus its import-time _key() call (no key file, no key variable is read) with a fake
_post, temp HOME and temp cache.

  * singleflight: the real embed(). Concurrent calls bill each document text once, with overlapping texts too; an owner's
    failure reaches its waiters, also for batches ex.map cancelled, with no hang and no billing in turn; a budget refusal
    likewise; the key published between first read and claim is not billed; queries stay on the v1.0.3 path; serial
    calls equal v1.0.3's embed() byte for byte, and the batches they send too
  * http: real embed(), real server: Search bytes of the v1.0.3 service == this service (test_v103.py replaces embed()
    and cannot see this), on a history Added concurrently by several users and on a workload sensitive to rounding
    (float32 first miss vs float16 from the cache); negative control: v1.0.3 bills the shared history several times
  * share: one string object per distinct content/role/session_id, after a restart too; any JSON value still accepted
  * rebuild: an embed failing inside rebuild() leaves the index as it was, so the retry does not append messages twice;
    negative control: v1.0.3 does
  * room: a Search filling every text window still returns 12 images within top_k; the text path keeps 100 windows
  * media: only JPEG/PNG/WebP up to 10 MiB get in, the bytes decide; the rest gets 422 and leaves nothing behind
  * health: /health answers with every worker thread busy (control: /search does not)
  * feedback: messages store.add drops are not embedded at Add

Run: python tests/test_fixes_v11.py [singleflight http share rebuild room media health feedback]   (default: all)
"""
from __future__ import annotations

import asyncio, base64, collections, hashlib, json, os, random, shutil, struct, subprocess, sys, tempfile, threading, time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_lru as tl                      # offline guard, stub embed, real_dashscope(), clean_env()

mm = tl.mm
ROOT, SVC, H, PY = mm.ROOT, mm.SVC, mm.H, tl.PY
FEEDBACK = 'Streaming online QA feedback: {"question": "which one?", "predicted_answer": "the blue one", "score": 1}'


# ------------------------------------------------------------------------------------------- the real embed(), child
class Api:
    """Stands in for dashscope._post: deterministic vectors per (text_type, text); counts what would be billed."""
    def __init__(self, ds, latency=0.02):
        self.ds, self.latency, self.lock = ds, latency, threading.Lock()
        self.calls, self.billed, self.posts = 0, collections.Counter(), []
        self.fail, self.fail_on, self.gate = None, 0, None

    def __call__(self, url, payload, retries=12):
        assert "embeddings" in url, url
        kind, texts = payload["parameters"]["text_type"], payload["input"]["texts"]
        with self.lock:
            self.calls += 1
            n = self.calls
            self.billed.update((kind, t) for t in texts)
            self.posts.append((kind, tuple(texts)))
        if self.gate:
            assert self.gate.wait(60)
        time.sleep(self.latency)
        if self.fail and (not self.fail_on or n == self.fail_on):
            raise self.fail
        return {"usage": {"total_tokens": sum(len(t.split()) + 1 for t in texts)},
                "output": {"embeddings": [self.emb(i, kind, t) for i, t in enumerate(texts)]}}

    def emb(self, i, kind, t):
        h = int(hashlib.sha1(f"{kind}|{t}".encode()).hexdigest()[:8], 16)
        r = np.random.default_rng(h)
        return {"text_index": i, "embedding": r.standard_normal(self.ds.EMB_DIM).tolist(),
                "sparse_embedding": [{"index": j, "value": 1.0 / (j + 1)} for j in r.integers(0, 3000, 12).tolist()]}

    def reset(self, **k):
        self.calls, self.billed, self.posts = 0, collections.Counter(), []
        self.fail, self.fail_on, self.gate = None, 0, None
        self.__dict__.update(k)


def together(fn, n):
    """fn(i) in n threads released at once; the result or exception of each. A hang is caught by the child's timeout."""
    gate = threading.Barrier(n)

    def one(i):
        gate.wait()
        try:
            return fn(i)
        except Exception as e:
            return e
    with ThreadPoolExecutor(n) as ex:
        return list(ex.map(one, range(n)))


def wait_until(cond, what):
    t0 = time.time()
    while not cond():
        assert time.time() - t0 < 60, what
        time.sleep(0.005)


def child_singleflight(svc: Path, base: Path | None, tmp: Path) -> None:
    sys.path.insert(0, str(svc))
    if base is not None:                        # v1.0.3 embed() for the serial comparison, its own cache
        os.environ["AML_CACHE"], cache = str(tmp / "cache103" / "emb.sqlite"), os.environ["AML_CACHE"]
        old = tl.real_dashscope(base)
        os.environ["AML_CACHE"] = cache
    ds = tl.real_dashscope(svc)
    api = Api(ds)
    ds._post, ds.BUDGET_YUAN = api, 1e9

    # concurrent calls, same texts: each billed once
    texts = [f"user: shared history line {i}" for i in range(40)]
    out = together(lambda i: ds.embed(texts + [f"user: private note {i}"]), 16)
    assert not [o for o in out if isinstance(o, Exception)], out
    assert len(api.billed) == 40 + 16 and set(api.billed.values()) == {1}, api.billed.most_common(3)
    assert not ds._inflight
    refs = {t: api.emb(0, "document", t) for t in texts}
    for dense, sparse in out:
        v = np.asarray(refs[texts[0]]["embedding"], np.float32)
        assert dense.shape == (41, ds.EMB_DIM) and np.abs(dense[0] - v / np.linalg.norm(v)).max() < 1e-3
        assert sparse[:40] == out[0][1][:40]
    print(f"singleflight: 16 concurrent calls of 40 shared texts: {api.calls} batches, each text billed once")

    # overlapping sets, several rounds: billed at most once, no call ever stuck
    api.reset(latency=0.001)
    rng = random.Random(3)
    pool = [f"user: pool text {i}" for i in range(300)]
    for _ in range(4):
        picks = [rng.sample(pool, 120) for _ in range(32)]
        out = together(lambda i: ds.embed(picks[i]), 32)
        assert not [o for o in out if isinstance(o, Exception)], out
        assert max(api.billed.values()) == 1 and not ds._inflight, api.billed.most_common(3)
    print(f"singleflight: 32 calls with overlapping texts, 4 rounds: {len(api.billed)} texts, each billed once")

    # the owner fails: all waiters fail with it at once, keys of the batches ex.map cancelled included
    waiting = set()

    class Counted(Future):
        def result(self, timeout=None):
            waiting.add(threading.get_ident())
            return super().result(timeout)

    ds.Future = Counted
    texts = [f"user: failing text {i}" for i in range(35)]        # 4 batches; workers=1 leaves 3 queued
    api.reset(latency=0.05, fail=RuntimeError("giving up on the API"), gate=threading.Event())
    res = []

    def call(ts, **k):
        try:
            res.append(ds.embed(ts, **k))
        except Exception as e:
            res.append(e)

    th = [threading.Thread(target=call, args=(texts,), kwargs={"workers": 1})]
    th[0].start()
    wait_until(lambda: api.calls == 1, "owner did not start")
    th += [threading.Thread(target=call, args=(texts[::-1],)) for _ in range(15)]
    for t in th[1:]:
        t.start()
    wait_until(lambda: len(waiting) == 15, "waiters did not arrive")
    api.gate.set()
    for t in th:
        t.join(60)
        assert not t.is_alive(), "a waiter is stuck"
    assert len(res) == 16 and all(isinstance(e, RuntimeError) and "giving up" in str(e) for e in res), res
    assert api.calls <= 4 and max(api.billed.values()) == 1 and not ds._inflight, (api.calls, ds._inflight)     # owner's only
    api.reset(latency=0)
    assert ds.embed(texts)[0].shape == (35, ds.EMB_DIM) and sum(api.billed.values()) == 35
    print("singleflight: owner failure with queued batches: 16 callers raise, no text tried twice; the retry works")

    # the second batch fails: waiters on the first get vectors, waiters on the second get the error
    texts = [f"user: half failing {i}" for i in range(20)]
    order = sorted(texts, key=lambda t: ds.CACHE.key(t, "document"))
    api.reset(latency=0, fail=RuntimeError("giving up on the API"), fail_on=2, gate=threading.Event())
    waiting.clear()
    res.clear()
    owner = threading.Thread(target=call, args=(texts,), kwargs={"workers": 1})
    owner.start()
    wait_until(lambda: api.calls == 1, "owner did not start")
    a, b = {}, {}

    def keep(d, ts):
        try:
            d["r"] = ds.embed(ts)
        except Exception as e:
            d["r"] = e

    th = [threading.Thread(target=keep, args=(a, order[:10])), threading.Thread(target=keep, args=(b, order[10:]))]
    for t in th:
        t.start()
    wait_until(lambda: len(waiting) == 2, "waiters did not arrive")
    api.gate.set()
    for t in [owner, *th]:
        t.join(60)
        assert not t.is_alive(), "stuck"
    assert isinstance(res[0], RuntimeError) and not isinstance(a["r"], Exception) and isinstance(b["r"], RuntimeError), (res, a, b)
    assert not ds._inflight
    print("singleflight: partial failure: finished keys reach their waiters, the failed keys raise")

    # a budget refusal reaches everybody
    ds.Future, ds.BUDGET_YUAN = Future, 0.0
    api.reset()
    texts = [f"user: over budget {i}" for i in range(25)]
    out = together(lambda i: ds.embed(texts), 16)
    assert all(isinstance(e, RuntimeError) and "budget exhausted" in str(e) for e in out), out
    assert api.calls == 0 and not ds._inflight
    ds.BUDGET_YUAN = 1e9
    print("singleflight: budget refusal: all 16 callers raise, no API call, nothing left claimed")

    # published between the first read and the claim: not billed
    api.reset()
    t = "user: published meanwhile"
    k, v = ds.CACHE.key(t, "document"), np.ones(ds.EMB_DIM, np.float32) / np.sqrt(ds.EMB_DIM)
    get = ds.CACHE.get

    def first_read(keys):
        got = get(keys)
        del ds.CACHE.get
        ds.CACHE.put([(k, v, {"1": 1.0})])
        return got

    ds.CACHE.get = first_read
    dense, sparse = ds.embed([t])
    assert api.calls == 0 and sparse == [{"1": 1.0}] and dense[0].tobytes() == v.astype(np.float16).astype(np.float32).tobytes()
    print("singleflight: a key published before the claim is read back, not billed")

    # queries stay as they were: each concurrent miss bills and uses its unrounded vector, then the cache
    api.reset(latency=0.5)
    q = "what did we talk about?"
    out = together(lambda i: ds.embed([q], "query"), 8)
    raw = np.asarray(api.emb(0, "query", q)["embedding"], np.float32)
    raw /= np.linalg.norm(raw) + 1e-9
    assert api.billed["query", q] == 8 and {o[0].tobytes() for o in out} == {raw.tobytes()}, api.billed
    assert ds.embed([q], "query")[0].tobytes() == raw.astype(np.float16).astype(np.float32).tobytes()
    assert not ds._inflight
    print("singleflight: 8 concurrent identical queries: 8 bills, 8 unrounded vectors, as v1.0.3")

    # serial calls: the same arrays and the same batches as v1.0.3's embed()
    if base is not None:
        seq = [([f"user: serial {i}" for i in range(25)], "document"), ([f"user: serial {i}" for i in range(25)], "document"),
               (["user: a", "user: a", "   ", "", "user: b " * 9000], "document"), (["q one", "q one", "q two"], "query"),
               (["q one"], "query"), ([f"user: serial {i}" for i in range(20, 35)], "document"), ([], "document"), ([], "query")]
        api_old = Api(old, 0)
        old._post, old.BUDGET_YUAN = api_old, 1e9
        api.reset(latency=0)
        for ts, kind in seq:
            a, b = ds.embed(ts, kind), old.embed(ts, kind)
            assert a[0].tobytes() == b[0].tobytes() and a[0].dtype == b[0].dtype and a[1] == b[1], (kind, ts[:2])
        assert sorted(api.posts) == sorted(api_old.posts) and api.billed == api_old.billed     # batches race within a call
        print(f"singleflight: {len(seq)} serial calls equal v1.0.3's embed(): arrays, sparse, {api.calls} batches")
    assert not mm.NET, mm.NET


def child_http(svc: Path, tmp: Path, out: Path) -> None:
    os.environ.update(AML_DB=str(tmp / "mem.sqlite"), AML_TOKEN="", AML_SEARCH_LOG=str(tmp / "search_log.jsonl"))
    sys.path.insert(0, str(svc))
    ds = tl.real_dashscope(svc)
    api = Api(ds, 0.01)
    ds._post, ds.BUDGET_YUAN = api, 1e9
    import httpx, server
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "timestamp": 1767261600000 + i * 86400000,
             "content": f"On day {i} we discussed topic {i % 7}: " + " ".join(f"token{i * j % 97}" for j in range(160))}
            for i in range(120)]
    hist[40]["content"], hist[70]["content"] = "Remove the onions from my shopping list", FEEDBACK
    note = lambda u: {"role": "user", "timestamp": 1769940000000, "content": f"private note for user {u}: code {u * 31}"}
    short = [{"role": "user" if i % 2 == 0 else "assistant", "timestamp": 1767261600000 + i * 3600000,
              "content": f"note {i} " + " ".join(f"tok{(i * j + j) % 211}" for j in range(25))} for i in range(1500)]

    async def main():
        tr = httpx.ASGITransport(app=server.app, raise_app_exceptions=False)
        bodies = []
        async with httpx.AsyncClient(transport=tr, base_url="http://t", timeout=None) as c:
            async def add(u, rid, msgs):
                r = await c.post("/add", json={"user_id": u, "session_id": rid, "request_id": rid, "messages": msgs})
                assert r.status_code == 200, r.text[:300]

            for s in range(0, 120, 30):        # 8 users write one history at once, session by session
                await asyncio.gather(*[add(f"u{u}", f"s{s}", hist[s:s + 30] + [note(u)]) for u in range(8)])
            await asyncio.gather(*[add(f"v{u}", "all", short) for u in range(3)])
            qs = [(f"u{u}", q + f" v{u}") for u in range(8) for q in (f"what was discussed on day {u * 3}?",
                  f"private note code for user {u}", "summarise topic 3", "What would you recommend for my shopping?")]
            qs += [("v0", f"question number {i} about tok{i % 211}") for i in range(40)]
            for _ in range(2):                 # first miss (unrounded) then hit (float16), serial: deterministic
                for u, q in qs:
                    r = await c.post("/search", json={"user_id": u, "query": q, "top_k": 50})
                    assert r.status_code == 200, r.text[:300]
                    bodies.append(r.content)
        return bodies

    bodies = asyncio.run(main())
    doc = {t: n for (k, t), n in api.billed.items() if k == "document"}
    json.dump({"sha": [hashlib.sha256(b).hexdigest() for b in bodies], "max_bills": max(doc.values()),
               "fb": doc.get(f"assistant: {FEEDBACK}", 0) + doc.get(f"user: {FEEDBACK}", 0),
               "windows": sum(len(json.loads(b)["data"]) for b in bodies)}, open(out, "w"))
    assert not mm.NET, mm.NET


def test_singleflight(base):
    tmp = Path(tempfile.mkdtemp(prefix="fix_sf_"))
    r = subprocess.run([PY, "-I", __file__, "_child_singleflight", str(SVC), str(base or ""), str(tmp)], env=tl.clean_env(tmp),
                       capture_output=True, text=True, timeout=300)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr[-3000:])
    shutil.rmtree(tmp, ignore_errors=True)
    assert r.returncode == 0, "singleflight child failed"


def test_http(base):
    tmp = Path(tempfile.mkdtemp(prefix="fix_http_"))
    res = {}
    try:
        for name, svc in (("new", SVC), ("old", base)):
            (tmp / name).mkdir()
            r = subprocess.run([PY, "-I", __file__, "_child_http", str(svc), str(tmp / name), str(tmp / f"{name}.json")],
                               env=tl.clean_env(tmp / name), capture_output=True, text=True, timeout=600)
            sys.stderr.write(r.stderr[-3000:])
            assert r.returncode == 0, f"{name} child failed"
            res[name] = json.load(open(tmp / f"{name}.json"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    new, old = res["new"], res["old"]
    assert new["sha"] == old["sha"], sum(a != b for a, b in zip(new["sha"], old["sha"]))
    assert new["max_bills"] == 1 and new["fb"] == 0, new["max_bills"]
    assert old["max_bills"] > 1 and old["fb"] > 0, "v1.0.3 billed nothing twice: this workload would not catch it"
    print(f"http: {len(new['sha'])} Searches ({new['windows']} windows) byte-identical to v1.0.3; history billed once, "
          f"v1.0.3 up to {old['max_bills']} times")


# --------------------------------------------------------------------------------------------- stub dashscope, here
def test_share(base):
    mem = tl.load_memsys(SVC, "memsys_share")
    store = mem.MemoryStore(mem.Config())
    body = json.dumps({"session_id": "a session", "messages": [{"role": "a role", "content": f"line {i} " + "x" * 300} for i in range(5)]})
    for u in ("u1", "u2"):
        store.add({**json.loads(body), "user_id": u})          # each request decoded on its own, as from a JSON body
    pairs = list(zip(store.users["u1"].msgs, store.users["u2"].msgs))
    assert len(pairs) == 5 and all(a.content is b.content and a.role is b.role and a.session_id is b.session_id for a, b in pairs)
    # role and session_id may be anything JSON: no TypeError from sys.intern, the value stays as it came
    store.add({"user_id": "u3", "session_id": 7, "messages": [{"role": None, "content": "a"}, {"role": 3, "content": "b"},
                                                              {"role": ["x"], "content": "lone \ud83d surrogate"}]})
    assert [m.role for m in store.users["u3"].msgs] == [None, 3, ["x"]] and store.users["u3"].msgs[0].session_id == 7
    assert mem.share("x") == "x" and mem.share(None) is None and mem.share(5) == 5
    work = tempfile.mkdtemp(prefix="fix_share_")
    try:
        c, _ = mm.load(SVC, "server_share1", work)
        for u in ("u1", "u2"):
            assert c.post("/add", json={**json.loads(body), "user_id": u, "request_id": "r"}, headers=H).status_code == 200
        _, srv = mm.load(SVC, "server_share2", work)              # restart: rows replayed one by one
        pairs = list(zip(srv.store.users["u1"].msgs, srv.store.users["u2"].msgs))
        assert len(pairs) == 5 and all(a.content is b.content for a, b in pairs)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("share: one object per content/role/session_id across users and after a restart; non-strings pass through")


def test_rebuild(base):
    first = [{"role": "user", "content": f"message {i} about the garden and the shopping"} for i in range(3)]
    second = [{"role": "user", "content": "Remove the onions from my shopping list"},
              {"role": "assistant", "content": "Done, the list has no onions now"}]
    q = {"user_id": "u", "query": "what is on the shopping list?", "top_k": 100}
    real = mm.stub.embed

    def run(mem, fail, indexed_first):
        """Same Adds into two stores, a failure in the first Search of one; the sizes and answers afterwards."""
        fresh, broken = mem.MemoryStore(mem.Config()), mem.MemoryStore(mem.Config())
        for s in (fresh, broken):
            s.add({"user_id": "u", "session_id": "s1", "request_id": "a", "messages": first})
            if indexed_first:
                s.search(q)
            s.add({"user_id": "u", "session_id": "s2", "request_id": "b", "messages": second})

        def once(texts, kind="document", workers=8):
            if kind == fail:
                mm.stub.embed = real
                raise RuntimeError("giving up on the API")
            return real(texts, kind, workers)

        mm.stub.embed = once
        try:
            broken.search(q)
            raise AssertionError("the failure was not injected")
        except RuntimeError:
            pass
        finally:
            mm.stub.embed = real
        got, want = broken.search(q), fresh.search(q)
        u, v = broken.users["u"], fresh.users["u"]
        return got == want, (len(u.p_owner), len(u.forgets), u.dense.shape), (len(v.p_owner), len(v.forgets), v.dense.shape)

    mem = tl.load_memsys(SVC, "memsys_rebuild")
    for fail in ("document", "query"):
        for indexed_first in (False, True):
            same, a, b = run(mem, fail, indexed_first)
            assert same and a == b, (fail, indexed_first, a, b)
    if base is not None:
        old = tl.load_memsys(base, "memsys_rebuild_old")
        _, a, b = run(old, "query", True)
        assert a[0] > b[0] and a[1] > b[1], (a, b)
        print(f"rebuild: negative control: on v1.0.3 the retry leaves {a[0]} passages instead of {b[0]}")
    print("rebuild: a failed embed in rebuild() (either kind, first or incremental build) changes nothing; the retry is clean")


def post(c, path, body):
    return c.post(path, json=body, headers=H)


def part_types(d):
    return [p["type"] for p in d["content"]]


def test_room(base):
    work = tempfile.mkdtemp(prefix="fix_room_")
    try:
        c, srv = mm.load(SVC, "server_room", work)
        assert srv.MM_CFG.max_items == 88 and srv.CFG.max_items == 100
        for s in range(110):
            text = f"coffee at the station number {s}"
            for u, content in (("pics", [{"type": "text", "text": text}, mm.img(mm.png(8, 8, str(s).encode()))]), ("words", text)):
                msg = {"role": "user", "timestamp": 1700000000000 + s * 86400000, "content": content}
                assert post(c, "/add", {"user_id": u, "session_id": f"s{s}", "request_id": f"r{s}", "messages": [msg]}).status_code == 200
        d = post(c, "/search", {"user_id": "pics", "query": "coffee station", "top_k": 100}).json()["data"]
        imgs = [x for x in d if "image_url" in part_types(x)]
        assert len(d) == 100 and len(imgs) == 12 and d[-12:] == imgs, (len(d), len(imgs))
        assert len(post(c, "/search", {"user_id": "words", "query": "coffee station", "top_k": 100}).json()["data"]) == 100
        d = post(c, "/search", {"user_id": "pics", "query": "coffee station", "top_k": 50}).json()["data"]
        assert len(d) == 50, len(d)                # top_k still wins
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("room: 88 text windows + 12 images at top_k=100 (was 100 + 0); the text path keeps 100 windows")


def data_uri(raw, mime):
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def test_media(base):
    work = tempfile.mkdtemp(prefix="fix_media_")
    max_img = 10 * 1024 * 1024

    def add(uid, mime, raw, extra=()):
        parts = [{"type": "text", "text": "a picture of a lighthouse"}, *extra, {"type": "image_url", "image_url": {"url": data_uri(raw, mime)}}]
        return post(c, "/add", {"user_id": uid, "session_id": "s", "request_id": "r", "messages": [{"role": "user", "content": parts}]})

    def images(uid):
        d = post(c, "/search", {"user_id": uid, "query": "lighthouse", "top_k": 100}).json()["data"]
        return [p["image_url"]["url"] for x in d for p in x["content"] if p["type"] == "image_url"]

    try:
        c, srv = mm.load(SVC, "server_media", work)
        webp = b"RIFF" + struct.pack("<I", 26) + b"WEBPVP8X" + struct.pack("<I", 10) + bytes(4) + (9).to_bytes(3, "little") + (19).to_bytes(3, "little")
        small = mm.png(4, 3, b"limit")
        full = mm.png(4, 3, b"limit", pad=max_img - len(small))
        assert len(full) == max_img
        for i, (declared, raw, mime) in enumerate((("image/png", mm.png(4, 3, b"ok"), "image/png"), ("image/jpeg", mm.jpeg(4, 3), "image/jpeg"),
                                                   ("image/webp", webp, "image/webp"), ("image/jpg", mm.jpeg(5, 3), "image/jpeg"),
                                                   ("image/png", mm.jpeg(6, 3), "image/jpeg"), ("image/png", full, "image/png"))):
            assert add(f"good{i}", declared, raw).status_code == 200, (i, declared)
            assert images(f"good{i}") == [data_uri(raw, mime)], i
        bad = {"gif": ("image/gif", b"GIF89a\x01\x00\x01\x00\x80\x00\x00" + bytes(16)),
               "gif as png": ("image/png", b"GIF89a\x01\x00\x01\x00\x80\x00\x00" + bytes(16)),
               "svg": ("image/svg+xml", b"<svg xmlns='http://www.w3.org/2000/svg'/>"), "bmp": ("image/bmp", b"BM" + bytes(64)),
               "not an image": ("image/png", b"hello world, not an image"),
               "one byte over": ("image/png", mm.png(4, 3, b"limit", pad=max_img - len(small) + 1))}
        files = sorted(p for p in srv.mmstore.blobs.root.rglob("*") if p.is_file())
        for name, (mime, raw) in bad.items():
            r = add(f"bad {name}", mime, raw)
            assert r.status_code == 422 and "JPEG, PNG or WebP" in r.text, (name, r.status_code, r.text[:200])
            assert not srv.DB.execute("SELECT 1 FROM adds WHERE user_id=?", (f"bad {name}",)).fetchone(), name
            assert f"bad {name}" not in srv.img_users
        assert sorted(p for p in srv.mmstore.blobs.root.rglob("*") if p.is_file()) == files
        # a good image next to a bad one: the Add is refused as a whole; the retry with the good one alone gets through
        good = mm.img(mm.png(4, 3, b"neighbour"))
        assert add("mixed", "image/gif", bad["gif"][1], extra=[good]).status_code == 422
        assert not srv.DB.execute("SELECT 1 FROM adds WHERE user_id='mixed'").fetchone()
        assert add("mixed", "image/png", mm.png(4, 3, b"neighbour")).status_code == 200 and len(images("mixed")) == 1
        # a query image of another format only loses the exact-match channel, as before
        q = [{"type": "text", "text": "lighthouse"}, {"type": "image_url", "image_url": {"url": data_uri(bad["gif"][1], "image/gif")}}]
        assert post(c, "/search", {"user_id": "good0", "query": q, "top_k": 100}).status_code == 200
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"media: 6 accepted (up to exactly 10 MiB, sniffed type wins); {len(bad)} refused with 422, nothing stored")


def test_health(base):
    import httpx
    work = tempfile.mkdtemp(prefix="fix_health_")
    real, blocked, release, lock = mm.stub.embed, [0], threading.Event(), threading.Lock()

    def stuck(texts, kind="document", workers=8):       # what a 429 backoff looks like from the thread pool
        with lock:
            blocked[0] += 1
        assert release.wait(60)
        return real(texts, kind, workers)

    async def main(app):
        tr = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=tr, base_url="http://t", timeout=None) as cl:
            adds = [asyncio.create_task(cl.post("/add", headers=H, json={
                "user_id": f"u{i}", "request_id": "r", "messages": [{"role": "user", "content": f"hello {i}"}]})) for i in range(48)]
            for _ in range(3000):
                if blocked[0] >= 40:
                    break
                await asyncio.sleep(0.01)
            assert blocked[0] == 40, blocked                # anyio's default limit: the whole pool is stuck
            try:
                await asyncio.wait_for(cl.post("/search", headers=H, json={"user_id": "u0", "query": "hello"}), 0.5)
                raise AssertionError("/search got through: the pool is not saturated")
            except asyncio.TimeoutError:
                pass
            r = await asyncio.wait_for(cl.get("/health"), 5)
            assert r.status_code == 200 and r.json() == {"status": "ok"}
            release.set()
            rs = await asyncio.gather(*adds)
            assert {x.status_code for x in rs} == {200}

    try:
        _, srv = mm.load(SVC, "server_health", work)
        assert asyncio.iscoroutinefunction(srv.health)
        mm.stub.embed = stuck
        asyncio.run(main(srv.app))
    finally:
        mm.stub.embed = real
        release.set()
        shutil.rmtree(work, ignore_errors=True)
    print("health: answers with all 40 worker threads stuck in Adds, /search does not")


def test_feedback(base):
    work = tempfile.mkdtemp(prefix="fix_fb_")
    real, seen = mm.stub.embed, []

    def spy(texts, kind="document", workers=8):
        if kind == "document":
            seen.append(list(texts))
        return real(texts, kind, workers)

    try:
        c, srv = mm.load(SVC, "server_fb", work)
        mm.stub.embed = spy
        add = lambda rid, msgs, u="u": post(c, "/add", {"user_id": u, "session_id": "s", "request_id": rid, "messages": msgs}).status_code
        assert add("a", [{"role": "user", "content": "we met at the bakery"}, {"role": "user", "content": FEEDBACK},
                         {"role": "assistant", "content": "  " + FEEDBACK}]) == 200
        assert seen == [["user: we met at the bakery"]], seen
        seen.clear()
        assert add("b", [{"role": "user", "content": FEEDBACK}]) == 200
        assert add("c", [{"role": "user", "content": [{"type": "text", "text": FEEDBACK}]}]) == 200
        assert seen == [], seen                             # nothing to embed: not even an empty call
        d = post(c, "/search", {"user_id": "u", "query": "where did we meet?", "top_k": 100}).json()["data"]
        assert len(d) == 1 and "bakery" in json.dumps(d) and "online QA feedback" not in json.dumps(d), d
        assert seen == [["user: we met at the bakery"]], seen   # the index asks for what Add embedded, a cache hit
        seen.clear()
        assert srv.DB.execute("SELECT COUNT(*) FROM adds WHERE user_id='u'").fetchone()[0] == 3     # still stored
        assert add("d", [{"role": "user", "content": None}], "v") == 500        # as in v1.0.3
        srv.CFG = replace(srv.CFG, stream_fb="raw")          # a mode that indexes them: Add must still embed them
        assert add("e", [{"role": "user", "content": FEEDBACK}], "w") == 200
        assert seen == [[f"user: {FEEDBACK}"]], seen
    finally:
        mm.stub.embed = real
        shutil.rmtree(work, ignore_errors=True)
    print("feedback: dropped messages are stored but not embedded at Add, with or without other messages")


TESTS = {"singleflight": test_singleflight, "http": test_http, "share": test_share, "rebuild": test_rebuild, "room": test_room,
         "media": test_media, "health": test_health, "feedback": test_feedback}
NEED_BASE = {"http"}


def main() -> None:
    if sys.argv[1:2] == ["_child_singleflight"]:
        return child_singleflight(Path(sys.argv[2]), Path(sys.argv[3]) if sys.argv[3] else None, Path(sys.argv[4]))
    if sys.argv[1:2] == ["_child_http"]:
        return child_http(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    tmp = Path(tempfile.mkdtemp(prefix="fix_"))
    base = tmp / "v103"
    base.mkdir()
    try:
        for f in ("server.py", "memsys.py", "timeannot.py", "dashscope.py"):
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
    print("no network calls; all fix checks passed")


if __name__ == "__main__":
    main()
