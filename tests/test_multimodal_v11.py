"""Offline checks of the v1.1 multimodal path. dashscope is replaced by an in-process stub before anything imports it
(the real module is never loaded, so no key is read), and every non-local socket connect is refused and counted.

  * images: stored once per content hash on disk; SQLite keeps image_ref parts, never base64; sizes read from headers
  * Search: ContentPart[]; image items after all text, each image from a returned window, under the count / byte /
    gpt-4o-mini token caps and top_k; the exact query-image channel; CJK bigram BM25 only on this path
  * restart (lazy load, nothing replayed), LRU eviction (also under concurrent Searches) and incremental Adds give
    byte-identical results; bad payloads get 422; diagnostic logs carry no content
  * textual users: every response byte-identical to v1.0.3 (git show v1.0.3:service/..., run in subprocesses), with
    and without image users interleaved

Run: python tests/test_multimodal_v11.py
"""
import base64, hashlib, importlib.util, json, os, random, re, shutil, socket, struct, subprocess, sys, tempfile, threading
import types, zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SVC = ROOT / "service"

# ----------------------------------------------------------------------------- offline guard and dashscope stub
NET = []
_connect, _connect_ex = socket.socket.connect, socket.socket.connect_ex


def _refuse(orig):
    def f(self, addr, *a, **k):
        if self.family != socket.AF_UNIX:
            NET.append(repr(addr))
            raise OSError("network call in offline test")
        return orig(self, addr, *a, **k)
    return f


socket.socket.connect, socket.socket.connect_ex = _refuse(_connect), _refuse(_connect_ex)
EMB_DIM = 256
DOCS = {"add": set(), "search": set()}       # document texts embedded while adding / while searching
PHASE = ["search"]


def fake_embed(texts, kind="document", workers=8):
    out = []
    for t in texts:
        if kind == "document":
            DOCS[PHASE[0]].add(t)
        v = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)).standard_normal(EMB_DIM)
        out.append((v / np.linalg.norm(v)).astype(np.float32))
    return (np.stack(out) if out else np.zeros((0, EMB_DIM), np.float32)), [{w: 1.0 for w in t.lower().split()[:20]} for t in texts]


def no_net(*a, **k):
    raise RuntimeError("network call in offline test")


stub = types.ModuleType("dashscope")
stub.EMB_DIM, stub.embed, stub.chat, stub.chat_cached, stub.rerank = EMB_DIM, fake_embed, no_net, no_net, no_net
sys.modules["dashscope"] = stub


def load(svc, name, work, **env):
    """A fresh server module over work/mem.sqlite (a 'restart' when work already holds data)."""
    for k in ("AML_CFG", "AML_CFG_MM", "AML_BLOBS", "AML_ADD_LOG"):
        os.environ.pop(k, None)
    os.environ.update(AML_DB=os.path.join(work, "mem.sqlite"), AML_SEARCH_LOG=os.path.join(work, "search_log.jsonl"),
                      AML_TOKEN="t0k", AML_MM=json.dumps(env.get("mm", {})))
    if str(svc) not in sys.path:
        sys.path.insert(0, str(svc))
    spec = importlib.util.spec_from_file_location(name, os.path.join(svc, "server.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from fastapi.testclient import TestClient
    return TestClient(m.app, raise_server_exceptions=False), m


H = {"Authorization": "Bearer t0k"}


# ----------------------------------------------------------------------------- images without Pillow
def png(w, h, salt=b"", pad=0):
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = zlib.compress(b"".join(b"\x00" + bytes(3 * w) for _ in range(h)))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", raw)
            + chunk(b"tEXt", b"k\x00" + hashlib.sha256(salt).digest() + bytes(pad)) + chunk(b"IEND", b""))


def jpeg(w, h):
    return (b"\xff\xd8\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            + b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, h, w, 1) + b"\x01\x11\x00\xff\xd9")


def uri(raw):
    return ("data:image/png;base64," if raw[:4] == b"\x89PNG" else "data:image/jpeg;base64,") + base64.b64encode(raw).decode()


def img(raw):
    return {"type": "image_url", "image_url": {"url": uri(raw)}}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


# ----------------------------------------------------------------------------- textual workload (worker mode)
def text_ops():
    rng = random.Random(7)
    en = "coffee tea movie hiking python recipe guitar paris tokyo budget gym book sister birthday dentist flight".split()
    zh = ["我昨天在拼多多买了保暖内衣", "周末和未婚妻去体育公园散步", "单位食堂的牛肉面很好吃", "下周要去合肥出差两天",
          "我给爸妈买了电热水袋", "订婚纪念日那天我们去吃了徽菜"]
    ops = []
    for s in range(30):
        msgs = []
        for i in range(rng.randint(1, 12)):
            c = rng.choice(zh) + "，" + rng.choice(zh) if s % 6 in (4, 5) else " ".join(rng.choice(en) for _ in range(rng.randint(3, 420)))
            m = {"role": "user" if i % 2 == 0 else "assistant", "content": [{"type": "text", "text": c}] if s % 6 == 3 else c}
            if (s + i) % 3:
                m["timestamp"] = 1700000000000 + s * 86400000 + i * 60000
            msgs.append(m)
        ops.append(("add", {"request_id": f"r{s}", "user_id": f"t{s % 6}", "session_id": f"s{s}", "messages": msgs}))
    ops.append(("add", {"request_id": "fb", "user_id": "t0", "session_id": "s99", "messages": [{"role": "user", "content":
                'Streaming online QA feedback: {"question": "x", "predicted_answer": "y", "score": 1}'}]}))
    ops += [("add", ops[0][1]), ("add", {"user_id": "t0"})]                      # idempotent retry; 422
    qs = ["What would you recommend for my sister's birthday?", "Now is 2023-12-01. What did I say about tokyo last week?",
          "How many books did I mention?", [{"type": "text", "text": "Which recipe did I like?"}], "coffee or tea?",
          "我给爸妈买了什么？", "订婚纪念日那天吃了什么菜？"]
    for u in ("t0", "t1", "t2", "t3", "t4", "t5", "nobody"):
        for q in qs:
            for k in (3, 100):
                b = {"query": q, "user_id": u, "top_k": k}
                if q == "coffee or tea?":
                    b["options"] = ["A. coffee", "B. tea"]
                ops.append(("search", b))
    return ops + [("search", {"user_id": "t0"})]


def img_ops(k):
    raw = png(8, 8, str(k).encode())
    u = f"img{k % 3}"
    first = [("add", {"request_id": f"pre{k}", "user_id": u, "session_id": f"ps{k}",
                      "messages": [{"role": "user", "content": "coffee in tokyo, no photo yet"}]})] if k < 15 else []
    return first + [("add", {"request_id": f"img{k}", "user_id": u, "session_id": f"is{k}", "messages": [
        {"role": "user", "timestamp": 1700000000000 + k * 3600000, "content": [
            {"type": "text", "text": f"photo {k}: coffee on the table in tokyo"}, img(raw)]},
        {"role": "assistant", "content": [{"type": "text", "text": "Nice coffee!"}]}]}),
        ("search", {"query": "coffee?", "user_id": u, "top_k": 100})]


def worker(svc, work, out, with_images):
    ops = [(o, b, True) for o, b in text_ops()]
    if with_images:
        mixed = []
        for i, op in enumerate(ops):
            mixed.append(op)
            if i % 5 == 0:
                mixed += [(o, b, False) for o, b in img_ops(i)]
        ops = mixed
    with open(out, "w", encoding="utf-8") as f:
        for phase in ("first", "restart"):           # restart: SQLite replay, then the same requests again
            for m in ("server", "memsys", "mmmem", "timeannot"):
                sys.modules.pop(m, None)
            c, _ = load(svc, "server", work)
            k = 0
            for op, body, textual in ops:
                r = c.post("/" + op, json=body, headers=H)
                if textual:
                    f.write(json.dumps({"phase": phase, "k": k, "status": r.status_code, "body": r.text}, ensure_ascii=False) + "\n")
                    k += 1
                else:
                    assert r.status_code == 200, r.text
    sys.exit(3 if NET else 0)


def textual_identical(tmp):
    base = os.path.join(tmp, "v103")
    os.makedirs(base)
    try:
        for f in ("server.py", "memsys.py", "timeannot.py"):
            src = subprocess.run(["git", "-C", str(ROOT), "show", f"v1.0.3:service/{f}"], capture_output=True, check=True).stdout
            open(os.path.join(base, f), "wb").write(src)
    except (OSError, subprocess.CalledProcessError):
        print("SKIP textual byte check: git tag v1.0.3 not available")
        return
    runs = {"v1.0.3": (base, ""), "v1.1": (str(SVC), ""), "v1.1+img": (str(SVC), "img")}
    procs = {k: subprocess.Popen([sys.executable, __file__, "--dump", d, os.path.join(tmp, k), os.path.join(tmp, k + ".jsonl"), f])
             for k, (d, f) in runs.items()}
    for k, p in procs.items():
        assert p.wait() == 0, f"worker {k} failed"
    ref = open(os.path.join(tmp, "v1.0.3.jsonl"), "rb").read()
    n = ref.count(b"\n")
    for k in ("v1.1", "v1.1+img"):
        assert open(os.path.join(tmp, k + ".jsonl"), "rb").read() == ref, f"textual responses of {k} differ from v1.0.3"
    print(f"textual users: {n}/{n} responses byte-identical to v1.0.3 (plain and with image users interleaved)")


# ----------------------------------------------------------------------------- multimodal checks
def check(data, top_k, sent, o):
    """Contract and caps of one ContentPart[] response; returns (#text items, #images, text tokens)."""
    import memsys, mmmem
    assert len(data) <= top_k
    ids = [d["id"] for d in data]
    assert len(ids) == len(set(ids)), ids
    sc = [d["score"] for d in data]
    assert all(a > b for a, b in zip(sc, sc[1:])), sc
    kinds, texts, tok, nbytes = [], [], 0, 0
    for d in data:
        c = d["content"]
        assert isinstance(c, list) and c and c[0]["type"] == "text" and c[0]["text"], d
        if d["id"].startswith("img:"):
            assert len(c) == 2 and c[1]["type"] == "image_url", d
            url = c[1]["image_url"]["url"]
            raw = base64.b64decode(url.split(",", 1)[1])
            h = sha(raw)
            assert h in sent and d["id"] == "img:" + h[:24] and c[0]["text"].startswith(f"[Image {h[:8]}] sent by the ")
            assert url.startswith("data:" + mmmem.sniff_mime(raw) + ";base64,")
            tok += mmmem.img_tokens(*mmmem.img_size(raw))
            nbytes += len(url)
            kinds.append(1)
        else:
            assert len(c) == 1
            texts.append(c[0]["text"])
            kinds.append(0)
    assert kinds == sorted(kinds), "image items must come after all text items"
    joined = "\n".join(texts)
    for d in data:
        if d["id"].startswith("img:"):
            assert d["content"][0]["text"][:16] in joined, "an image must belong to a returned window"
    text_tok = sum(memsys.ntok(t) + 8 for t in texts)
    assert sum(kinds) <= o["img_max"] and nbytes <= o["img_byte_budget"]
    assert tok <= min(o["img_tok_budget"], o["answer_tok"] - text_tok)
    return len(texts), sum(kinds), text_tok


def multimodal(tmp):
    work = os.path.join(tmp, "mm")
    c, srv = load(SVC, "server_mm1", work)
    import memsys, mmmem
    D = mmmem.MM_DEFAULTS

    def add(cl, body, status=200):
        PHASE[0] = "add"
        r = cl.post("/add", json=body, headers=H)
        PHASE[0] = "search"
        assert r.status_code == status, (status, r.status_code, r.text)
        return r

    def search(cl, body, status=200):
        r = cl.post("/search", json=body, headers=H)
        assert r.status_code == status, (status, r.status_code, r.text)
        return r

    T = 1714989420000                                    # 2024-05-06 09:57 UTC
    P1, P2, P3 = png(300, 200, b"p1"), jpeg(640, 480), png(64, 64, b"p3")
    garden = [png(64, 64, b"g%d" % k, pad=3000) for k in range(20)]
    big = png(16, 16, b"big", pad=mmmem.MAX_IMG + 1)
    fb = 'Streaming online QA feedback: {"question": "q", "predicted_answer": "zebra crossing", "score": 1}'
    # mA: a text-only Add first (text path), then images; mB shares P1; mC 20 photos; mD an oversized image is refused
    add(c, {"request_id": "a1", "user_id": "mA", "session_id": "sa1", "messages": [
        {"role": "user", "timestamp": T - 86400000 * 200, "content": "I adopted a corgi named Biscuit last spring."},
        {"role": "assistant", "timestamp": T - 86400000 * 200 + 1000, "content": "Congratulations on Biscuit!"}]})
    assert "mA" in srv.store.users and "mA" not in srv.img_users
    a2 = {"request_id": "a2", "user_id": "mA", "session_id": "sa2", "messages": [
        {"role": "user", "timestamp": T, "content": [{"type": "text", "text": "Biscuit at the beach today"},
                                                     {"type": "text", "text": "Image Caption: a corgi running on wet sand"}, img(P1)]},
        {"role": "assistant", "timestamp": T + 1000, "content": [{"type": "text", "text": "He looks so happy!"}]},
        {"role": "user", "timestamp": T + 2000, "content": fb}]}
    add(c, a2)
    assert "mA" not in srv.store.users and "mA" in srv.img_users
    add(c, a2)                                                                  # retry: idempotent
    add(c, {"request_id": "a3", "user_id": "mA", "session_id": "sa3", "messages": [
        {"role": "user", "timestamp": T + 86400000, "content": [{"type": "text", "text": "Same beach photo again, and his new bed"},
                                                                img(P1), img(P2)]},
        {"role": "assistant", "timestamp": T + 86400000 + 1000, "content": "A cosy bed for Biscuit."}]})
    add(c, {"request_id": "b1", "user_id": "mB", "session_id": "sb1", "messages": [
        {"role": "user", "timestamp": T, "content": "My neighbour repaints his fence every summer."}]})
    add(c, {"request_id": "b2", "user_id": "mB", "session_id": "sb2", "messages": [
        {"role": "user", "timestamp": T + 5000, "content": [{"type": "text", "text": "Look what I saw on my walk"}, img(P1)]}]})
    add(c, {"request_id": "c1", "user_id": "mC", "session_id": "sc1", "messages": [
        m for k, g in enumerate(garden) for m in (
            {"role": "user", "timestamp": T + k * 60000, "content": [{"type": "text", "text": f"garden photo {k}: tulips"}, img(g)]},
            {"role": "assistant", "timestamp": T + k * 60000 + 1000, "content": "Lovely tulips."})]})
    add(c, {"request_id": "d0", "user_id": "mD", "session_id": "sd1", "messages": [
        {"role": "user", "timestamp": T, "content": [{"type": "text", "text": "a huge scan of my passport"}, img(big)]}]}, 422)
    add(c, {"request_id": "d1", "user_id": "mD", "session_id": "sd1", "messages": [
        {"role": "user", "timestamp": T, "content": "a huge scan of my passport"},
        {"role": "user", "timestamp": T + 1000, "content": [{"type": "text", "text": "and a small selfie"}, img(P3)]}]})
    zh = ["我昨天在拼多多买了保暖内衣", "周末和未婚妻去体育公园散步", "单位食堂的牛肉面很好吃", "我给爸妈买了电热水袋"]
    for k, t in enumerate(zh):
        add(c, {"request_id": f"z{k}", "user_id": "mZ", "session_id": f"sz{k}", "messages": [
            {"role": "user", "timestamp": T + k * 60000, "content": [{"type": "text", "text": t}] + ([img(P3)] if k == 0 else [])}]})
    sent = {"mA": {sha(P1), sha(P2)}, "mB": {sha(P1)}, "mC": {sha(g) for g in garden}, "mD": {sha(P3)}, "mZ": {sha(P3)}}

    # ---- storage: one file per distinct image, no base64 in SQLite, image_ref with header sizes
    files = {p.name: p.read_bytes() for p in Path(work, "blobs").rglob("*") if p.is_file()}
    every = [P1, P2, P3] + garden
    assert set(files) == {sha(r) for r in every} and all(files[sha(r)] == r for r in every), "blob store"
    bodies = [b for (b,) in srv.DB.execute("SELECT body FROM adds")]
    heads = [base64.b64encode(r[:30]).decode() for r in every]
    assert not any("base64" in b or any(h in b for h in heads) for b in bodies)
    refs = {p["sha256"]: p for b in bodies for m in json.loads(b)["messages"] if isinstance(m["content"], list)
            for p in m["content"] if p["type"] == "image_ref"}
    assert (refs[sha(P1)]["w"], refs[sha(P1)]["h"], refs[sha(P1)]["mime"], refs[sha(P1)]["bytes"]) == (300, 200, "image/png", len(P1))
    assert (refs[sha(P2)]["w"], refs[sha(P2)]["h"], refs[sha(P2)]["mime"]) == (640, 480, "image/jpeg")
    assert srv.DB.execute("SELECT COUNT(*) FROM adds WHERE user_id='mA'").fetchone()[0] == 3
    assert {u for (u,) in srv.DB.execute("SELECT user_id FROM img_users")} == set(sent)
    db_bytes = sum(os.path.getsize(os.path.join(work, f)) for f in os.listdir(work) if f.startswith("mem.sqlite"))
    print(f"storage: {len(files)} image files for {len(every) + 2} image parts sent, no base64 in SQLite ({db_bytes} B)")

    # ---- Search: ContentPart[], images after text, caps
    mmq = [{"query": [{"type": "text", "text": "What is the name of my corgi?"}], "user_id": "mA", "top_k": 100},
           {"query": "Which bed did I buy for Biscuit?", "user_id": "mA", "top_k": 3},
           {"query": [img(P1)], "user_id": "mB", "top_k": 100},
           {"query": "Show me my garden photos", "user_id": "mC", "top_k": 100},
           {"query": "passport and selfie", "user_id": "mD", "top_k": 100},
           {"query": "我给爸妈买了什么？", "user_id": "mZ", "top_k": 100, "options": ["A. 电热水袋", 7, None]}]
    R1 = [search(c, q).content for q in mmq]
    got = [json.loads(r)["data"] for r in R1]
    stats = [check(d, q["top_k"], sent[q["user_id"]], D) for d, q in zip(got, mmq)]
    a_text = "\n".join(p["text"] for d in got[0] if not d["id"].startswith("img:") for p in d["content"])
    assert "corgi named Biscuit last spring" in a_text and f"[Image {sha(P1)[:8]}]" in a_text   # rows before the first image too
    assert "zebra" not in a_text and "Streaming online QA feedback" not in a_text                # stream feedback not indexed
    assert re.match(r"\[\d{4}-\d\d-\d\d \(\w{3}\) \d\d:\d\d\]\n", got[0][0]["content"][0]["text"]), "mm headers keep HH:MM"
    assert {sha(base64.b64decode(d["content"][1]["image_url"]["url"].split(",")[1])) for d in got[0][stats[0][0]:]} == sent["mA"]
    assert f"[Image {sha(P1)[:8]}]" in got[2][0]["content"][0]["text"], "exact query-image match ranks its turn first"
    assert stats[3][1] == D["img_max"], stats[3]                                                  # count cap
    assert stats[4][1] == 1 and got[4][-1]["id"] == "img:" + sha(P3)[:24]                         # > 10 MiB refused
    print("search: ContentPart[], images after text, id/score contract ok;", "items/images per query:", [s[:2] for s in stats])

    def direct(opts, q=mmq[3], cfg=None):
        st = mmmem.MMStore(cfg or srv.MM_CFG, srv.DB, srv.db_lock, srv.mmstore.blobs, opts)
        return st.search_parts(q)
    n_text, _, text_tok = stats[3]
    one = 2833 + 5667                                    # a 64x64 image: one 512-px tile
    caps = {"img_max": ({"img_max": 3}, 3), "img_tok_budget": ({"img_tok_budget": 2 * one}, 2),
            "answer_tok": ({"answer_tok": text_tok + 3 * one}, 3), "off": ({"img_max": 0}, 0),
            "img_byte_budget": ({"img_byte_budget": 2 * mmmem.b64_len(len(garden[0])) + 5}, 2)}
    for name, (o, want) in caps.items():
        d = direct(o)
        assert check(d, 100, sent["mC"], {**D, **o})[1] == want, (name, want)
    d = direct({}, {**mmq[3], "top_k": n_text + 1})
    assert check(d, n_text + 1, sent["mC"], D)[1] == 1, "images also count towards top_k"
    print("caps: count / image tokens / answer tokens / bytes / top_k / img_max=0 ok")

    # ---- S1: CJK bigrams on the multimodal path only (BM25 alone, one message per window)
    bm = replace(srv.MM_CFG, w_dense=0, w_sparse=0, w_session=0, budget=20, route={})
    zq = {"query": "我给爸妈买了什么？", "user_id": "mZ", "top_k": 1}
    assert "电热水袋" in direct({"cjk_bm25": True}, zq, bm)[0]["content"][0]["text"]
    assert "电热水袋" not in direct({"cjk_bm25": False}, zq, bm)[0]["content"][0]["text"]
    toks = lambda: memsys.bm25s.tokenize(zh + ["coffee in Tokyo"], stopwords="en", stemmer=memsys.STEMMER, return_ids=False, show_progress=False)
    real = memsys.bm25s._real.tokenize(zh + ["coffee in Tokyo"], stopwords="en", stemmer=memsys.STEMMER, return_ids=False, show_progress=False)
    assert toks() == real, "proxy must forward the very same call when the switch is off"
    with mmmem.cjk_bm25(True):
        assert toks()[3] == ["我给", "给爸", "爸妈", "妈买", "买了", "了电", "电热", "热水", "水袋"] and toks()[4] == real[4]
    print("cjk: bigram BM25 finds the Chinese answer on the mm path; text-path tokenization unchanged")

    # ---- S4: bad payloads -> 422, nothing persisted
    n_rows = srv.DB.execute("SELECT COUNT(*) FROM adds").fetchone()[0]
    bad_parts = {"remote url": {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                 "bad base64": {"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@"}},
                 "no url": {"type": "image_url", "image_url": {}}, "not a dict": "oops",
                 "text not str": {"type": "text", "text": 5}}
    for k, p in bad_parts.items():
        add(c, {"request_id": "bad-" + k, "user_id": "mNew", "session_id": "x", "messages": [{"role": "user", "content": [img(P3), p]}]}, 422)
    add(c, {"request_id": "bad-content", "user_id": "mA", "session_id": "x", "messages": [{"role": "user", "content": 5}]}, 422)
    add(c, {"request_id": "bad-msg", "user_id": "mA", "session_id": "x", "messages": ["oops"]}, 422)
    add(c, {"user_id": "mA"}, 422)
    assert srv.DB.execute("SELECT COUNT(*) FROM adds").fetchone()[0] == n_rows and "mNew" not in srv.img_users
    for q in (["oops"], {"type": "text"}):
        search(c, {"query": q, "user_id": "mA", "top_k": 5}, 422)
    search(c, {"query": "corgi", "user_id": "mA", "top_k": "five"}, 422)
    search(c, {"user_id": "mA"}, 422)
    ok = [search(c, {"query": q, "user_id": "mA", "top_k": 5}).json()["data"] for q in
          ([], [img(P2)], [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}])]
    assert all(ok) and ok[1][0]["content"][0]["text"].count(f"[Image {sha(P2)[:8]}]")
    print("bad payloads: 422 for broken parts / content / messages / query / top_k; empty and image-only queries 200")

    # ---- S3: structure-only logs
    alog = [json.loads(l) for l in open(os.path.join(work, "add_log.jsonl"))]
    slog = [json.loads(l) for l in open(os.path.join(work, "search_log.jsonl"))]
    assert alog[0]["shape"] == "TTI|T|S" and alog[0]["img_bytes"] == len(P1) and alog[0]["ts_set"] == 3
    mm_recs = [r for r in slog if r["mm"] == "img"]
    assert mm_recs and all({"n_img", "img_b64", "resident", "query_images"} <= set(r) for r in mm_recs)
    assert mm_recs[2]["q_chars"] == 0 and mm_recs[2]["query_images"] == 1 and mm_recs[5]["n_options"] == 3
    assert mm_recs[5]["q_sha"] == hashlib.sha256("我给爸妈买了什么？".encode()).hexdigest()[:12]
    assert not any({"query", "options"} & set(r) for r in slog), "question and options text must not be logged"
    for f in ("add_log.jsonl", "search_log.jsonl"):
        assert "base64" not in open(os.path.join(work, f)).read(), f
    print("logs: add_log shapes", sorted({r["shape"] for r in alog})[:4], "...; no image data in logs")

    # ---- restart: nothing replayed for image users, lazy load gives identical bytes
    c2, srv2 = load(SVC, "server_mm2", work)
    assert srv2.mmstore.resident() == (0, 0) and not set(sent) & set(srv2.store.users)
    assert [search(c2, q).content for q in mmq] == R1, "restart (lazy load) changed results"
    assert srv2.mmstore.resident()[0] == len({q["user_id"] for q in mmq})
    print("restart: image users not replayed; lazy-loaded results byte-identical")

    # ---- LRU: every user evicted after each Search; reloads (also concurrent) give identical bytes
    c3, srv3 = load(SVC, "server_mm3", work, mm={"max_passages": 1})
    for i in [5, 0, 3, 0, 2, 4, 1, 3]:
        assert search(c3, mmq[i]).content == R1[i], f"LRU reload changed results ({i})"
        assert srv3.mmstore.resident() == (0, 0)
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda i: srv3.search(mmq[i]).body, [i % len(mmq) for i in range(48)]))
    assert all(b == R1[i % len(mmq)] for i, b in enumerate(out)), "concurrent Searches under eviction differ"
    print("lru: evicted users reload byte-identical, also with 8 concurrent Searches")

    # ---- an Add after the user is resident is searchable at once and equals a from-scratch load
    add(c2, {"request_id": "a4", "user_id": "mA", "session_id": "sa4", "messages": [
        {"role": "user", "timestamp": T + 86400000 * 60, "content": [{"type": "text", "text": "Biscuit learned to swim in July"}, img(P3)]}]})
    q = {"query": "When did Biscuit learn to swim?", "user_id": "mA", "top_k": 100}
    r = search(c2, q)
    assert "learned to swim" in r.text and check(r.json()["data"], 100, sent["mA"] | {sha(P3)}, D)[1] >= 1
    c4, _ = load(SVC, "server_mm4", work)
    assert search(c4, q).content == r.content, "incremental load differs from a fresh one"
    assert DOCS["search"] <= DOCS["add"], "Search embedded a document string that Add did not (cache miss)"
    print("incremental: Add then Search sees it; same bytes as a fresh load; Search re-embeds nothing new")
    assert c.post("/search", json=mmq[0]).status_code == 401


if __name__ == "__main__":
    if sys.argv[1:2] == ["--dump"]:
        worker(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] == "img")
    tmp = tempfile.mkdtemp()
    try:
        multimodal(tmp)
        textual_identical(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert not NET, NET
    print("no network calls; all multimodal v1.1 checks passed")
