"""Offline check that no diagnostic log (search_log, add_log, coding_shape) holds the text of a question, an option or a
message, on any of the three paths. dashscope is the stub of test_multimodal_v11, sockets are refused.

  * every text sent carries a canary; afterwards every file the service wrote, databases and image blobs aside, is scanned
  * every log record must fit a fixed schema: known keys only, and strings only where they are structure
  * hashes are sha256[:12] of the question text; option counts, lengths and image counts are right
  * negative control: the same scan on the v1.0.3 server must find the question and the options

Run: python tests/test_logs_v11.py
"""
import os as _o; _o.environ.setdefault("AML_TEXT_TAG", "0")     # compares textual bytes with older versions
import json, os, re, shutil, subprocess, sys, tempfile
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_coding_v11 as tc               # boot/add/search/traj and, through it, the offline guard and stub

mm = tc.mm
ROOT, SVC, H = mm.ROOT, mm.SVC, mm.H
QUESTION = "QCANARY which café did we pick"
QUESTION_ZH = "题面金丝雀：我们选了哪家咖啡馆"
OPTIONS = ["OCANARY_A the blue one", "选项金丝雀乙"]
MESSAGE = "MCANARY we picked the corner café near the station"
CAPTION = "CCANARY a photo of the corner café"
CODE_MSG = "XCANARY tool output of parse_config"
ROLE = "RCANARY a free text role"
KEY = "KCANARY a free text field"
CANARIES = ("QCANARY", "题面金丝雀", "OCANARY", "选项金丝雀", "MCANARY", "CCANARY", "XCANARY", "RCANARY", "KCANARY")
SKIP = re.compile(r"mem(_coding)?\.sqlite")          # where messages belong


def h12(s):
    return sha256(s.encode("utf-8")).hexdigest()[:12]


def hexn(v):
    return isinstance(v, str) and re.fullmatch(r"[0-9a-f]{12}", v) is not None


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0


def count(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def name(v):
    return isinstance(v, str) and re.fullmatch(r"[A-Za-z_][\w.-]{0,31}", v, re.ASCII) is not None


TEXT = {"ts": num, "user_id": lambda v: v in USERS, "mm": lambda v: isinstance(v, bool), "top_k": count, "q_sha": hexn, "q_chars": count,
        "n_options": count, "query_images": count, "n": count, "chars": count, "sec": num}
IMG = {**TEXT, "mm": lambda v: v == "img", "n_img": count, "img_b64": count, "chars": count,
       "resident": lambda v: isinstance(v, list) and len(v) == 2 and all(count(x) for x in v)}
ADD = {"ts": num, "user_id": lambda v: v in USERS, "shape": lambda v: re.fullmatch(r"[STI?|-]*", v) is not None, "ts_set": count, "img_bytes": count}
CODE_ADD = {"op": lambda v: v == "add", "ts": num, "u": hexn, "sid": lambda v: v is None or hexn(v), "n_msgs": count,
            "roles": lambda v: all((name(k) or hexn(k)) and count(c) for k, c in v.items()), "chars": count, "max_chars": count,
            "list_content": count, "has_ts": count, "extra_keys": lambda v: all(name(k) or hexn(k) for k in v), "sec": num}
CODE_SEARCH = {"op": lambda v: v == "search", "ts": num, "u": hexn, "q_type": name, "top_k": lambda v: count(v) or name(v), "q_sha": hexn,
               "q_chars": count, "n_options": count, "n": count, "chars": count, "extra_keys": lambda v: all(name(k) or hexn(k) for k in v),
               "error": lambda v: v is None or name(v), "sec": num}
USERS = {"tU", "iU"}


def fits(rec, schema):
    assert set(rec) == set(schema), sorted(set(rec) ^ set(schema))
    for k, ok in schema.items():
        assert ok(rec[k]), (k, rec[k])


def scan(work):
    """{canary: [files holding it]} over everything the service wrote, SQLite files and blobs aside."""
    found, seen = {}, []
    for f in sorted(Path(work).rglob("*")):
        if not f.is_file() or SKIP.match(f.name) or "blobs" in f.parts:
            continue
        seen.append(f.name)
        body = f.read_bytes().decode("utf-8", "replace")
        body += json.dumps(body)                     # also the \uXXXX spelling
        for c in CANARIES:
            if c in body:
                found.setdefault(c, []).append(f.name)
    return found, seen


def traffic(c):
    """The same requests on any server: a textual user and an image user with every kind of text. Returns the Searches."""
    tc.add(c, {"request_id": "t1", "user_id": "tU", "session_id": "ts", "messages": [
        {"role": "user", "content": MESSAGE}, {"role": "assistant", "content": [{"type": "text", "text": MESSAGE}]}]})
    tc.add(c, {"request_id": "i1", "user_id": "iU", "session_id": "is", "messages": [
        {"role": "user", "content": [{"type": "text", "text": CAPTION}, mm.img(mm.png(8, 8, b"log"))]},
        {"role": "assistant", "content": MESSAGE}]})
    out = []
    for uid in ("tU", "iU"):
        for body in ({"query": QUESTION, "top_k": 5}, {"query": QUESTION_ZH, "top_k": 5, "options": OPTIONS},
                     {"query": [{"type": "text", "text": QUESTION}], "top_k": 5, "options": OPTIONS}):
            tc.search(c, {**body, "user_id": uid})
            out.append({**body, "user_id": uid})
    return out


def logs(tmp):
    work = os.path.join(tmp, "new")
    USERS.add("tU")
    c, srv = tc.boot("server_lg", work)
    sent = traffic(c)
    # the coding path
    U = "cU"
    tc.add(c, {"request_id": "c1", "user_id": U, "session_id": "cs", "messages": tc.traj("parse_config", ts=1700000000000) + [
        {"role": ROLE, "content": CODE_MSG}, {"role": {"CANARY": ROLE}, "content": CODE_MSG}, CODE_MSG]})
    tc.add(c, {"request_id": "c2", "user_id": U, "session_id": "cs2", KEY: CODE_MSG, "metadata": {"text": CODE_MSG},
               "messages": [{"role": "tool", "content": [{"type": "text", "text": CODE_MSG}]}]})
    tc.search(c, {"query": f"parse_config {QUESTION}", "user_id": U, "top_k": 5, "options": OPTIONS})
    tc.search(c, {"query": [{"type": "text", "text": QUESTION_ZH}], "user_id": U, "top_k": 3, KEY: QUESTION, "trace": "x"})
    assert U in srv.code_users and "iU" in srv.img_users and "tU" not in srv.code_users

    found, seen = scan(work)
    assert not found, found
    assert {"search_log.jsonl", "add_log.jsonl", "coding_shape.jsonl"} <= set(seen), seen
    # sanity: the texts did reach the service (messages in SQLite, so the scan looks for what was really sent)
    db = open(os.path.join(work, "mem.sqlite-wal"), "rb").read() + open(os.path.join(work, "mem.sqlite"), "rb").read()
    assert b"MCANARY" in db and b"CCANARY" in db

    slog = [json.loads(l) for l in open(os.path.join(work, "search_log.jsonl"), encoding="utf-8")]
    alog = [json.loads(l) for l in open(os.path.join(work, "add_log.jsonl"), encoding="utf-8")]
    clog = [json.loads(l) for l in open(os.path.join(work, "coding_shape.jsonl"), encoding="utf-8")]
    assert len(slog) == len(sent) and len(alog) == 1 and len(clog) == 4
    for r in slog:
        fits(r, IMG if r["mm"] == "img" else TEXT)
    fits(alog[0], ADD)
    assert alog[0]["shape"] == "TI|S" and alog[0]["img_bytes"] == len(mm.png(8, 8, b"log"))
    for r in clog:
        fits(r, CODE_ADD if r["op"] == "add" else CODE_SEARCH)
    for r, b in zip(slog, sent):
        text = b["query"] if isinstance(b["query"], str) else b["query"][0]["text"]
        assert r["user_id"] == b["user_id"] and r["q_sha"] == h12(text) and r["q_chars"] == len(text)
        assert r["n_options"] == len(b.get("options", ())) and r["top_k"] == 5
    assert clog[2]["q_sha"] == h12(f"parse_config {QUESTION}") and clog[2]["n_options"] == 2 and clog[3]["n_options"] == 0
    assert clog[3]["q_sha"] == h12(QUESTION_ZH) and clog[0]["u"] == h12(U) and clog[0]["sid"] == h12("cs")
    assert h12(ROLE) in clog[0]["roles"] and clog[1]["extra_keys"] == sorted([h12(KEY), "metadata"]), clog[:2]
    assert clog[3]["extra_keys"] == sorted([h12(KEY), "trace"]), clog[3]

    # shapes of options that are not a list; a lone surrogate must not reach the log either
    for opts, n in ((None, 0), ([], 0), ("OCANARY text", 1), ({"a": "OCANARY", "b": "OCANARY"}, 2), (["\ud800 OCANARY"], 1)):
        tc.search(c, {"query": QUESTION, "user_id": "tU", "top_k": 5, "options": opts})
        rec = json.loads(open(os.path.join(work, "search_log.jsonl"), encoding="utf-8").read().splitlines()[-1])
        assert rec["n_options"] == n, (opts, rec)
    assert not scan(work)[0]
    # a failed Search logs the error type only
    orig = srv.code_store.rank
    srv.code_store.rank = lambda *a, **k: 1 / 0
    tc.search(c, {"query": QUESTION, "user_id": U, "top_k": 5})
    srv.code_store.rank = orig
    assert not scan(work)[0]
    print(f"logs: {len(slog)} Search, {len(alog)} image Add, {len(clog)} coding records fit their schema; no text in {sorted(set(seen))}")


def negative(tmp):
    base = os.path.join(tmp, "v103")
    os.makedirs(base)
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show", "v1.0.3:service/server.py"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        print("SKIP negative control: git tag v1.0.3 not available")
        return
    open(os.path.join(base, "server.py"), "wb").write(src)
    work = os.path.join(tmp, "old")
    c, _ = mm.load(base, "server_lg_old", work)
    traffic(c)
    found, _ = scan(work)
    assert {"QCANARY", "题面金丝雀", "OCANARY", "选项金丝雀"} <= set(found), found
    print("negative control: the same scan finds question and option text in v1.0.3's search_log")


if __name__ == "__main__":
    sys.path.insert(0, str(SVC))
    tmp = tempfile.mkdtemp()
    try:
        logs(tmp)
        negative(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert not mm.NET, mm.NET
    print("no network calls; all log checks passed")
