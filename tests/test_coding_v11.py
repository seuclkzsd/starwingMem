"""Offline checks of the v1.1 coding path. dashscope is the in-process stub of test_multimodal_v11 (the real module is
never loaded, so no key is read), every non-local socket connect is refused and counted, and embed calls are counted.

  * detector: the structural / image / marker thresholds at their exact edges; no false positive on textual samples
    (local public corpora too, when present); no quadratic regex on blank lines or long runs
  * routing: sticky on the first Add, one decision per user under concurrent first Adds, image users never on the coding
    path (a routed user that sends an image moves over with its Adds), AML_CODE_DETECT=0, state restored on restart
  * no embedding call on the coding path, nothing of it in mem.sqlite, nothing replayed
  * contract (M2): content / query / top_k shapes, ids, counts, 422 / 401; fault injection degrades to 200
  * unit shapes (M3), rel_cut (S1), background prebuild (S2), structure-only log (S3)
  * restart, LRU eviction (also racing Adds) and incremental Adds give byte-identical results
  * textual users: every response byte-identical to v1.0.3 (git show v1.0.3:service/..., run in subprocesses), with
    coding users interleaved; AML_CODE_DETECT=0 is v1.0.3 for trajectory users as well

Run: python tests/test_coding_v11.py
"""
import ast, itertools, json, os, re, shutil, sqlite3, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_multimodal_v11 as mm           # offline guard, dashscope stub, load(), textual workload, png()/img()

ROOT, SVC, H = mm.ROOT, mm.SVC, mm.H
EMBEDS = [0]
_embed = mm.stub.embed


def counted_embed(*a, **k):
    EMBEDS[0] += 1
    return _embed(*a, **k)


mm.stub.embed = counted_embed
TOPICS = "parse_config socket_timeout image_resize csv_export token_refresh cache_eviction oauth_scope".split()
USER = {"role": "user", "content": "We had coffee in Tokyo and talked about the new gym near the office."}


def boot(name, work, detect="1", cfg=None, log=None):
    """A fresh server module over work/mem.sqlite (a 'restart' when work already holds data)."""
    os.environ.update(AML_CODE_DETECT=detect, AML_CFG_CODING=json.dumps({"prebuild_delay": 0} if cfg is None else cfg))
    os.environ.pop("AML_CODE_LOG", None)
    if log is not None:
        os.environ["AML_CODE_LOG"] = log
    return mm.load(SVC, name, work)


def traj(topic, n=30, ts=None, parts=False):
    """An agent trajectory of n messages; the topic word makes it findable. parts=True: ContentPart[] messages."""
    f = f"src/{topic}/core.py"
    steps = [("assistant", f'[tool_use Read] {{"file_path": "{f}"}}'),
             ("user", f'def {topic}_handler(x):\n    return x["key"]\n\ntool_use_id: toolu_1'),
             ("assistant", f'[tool_use Edit] {{"file_path": "{f}", "old_string": "x[key]", "new_string": "x.get(key)"}}'),
             ("user", f'diff --git a/{f} b/{f}\n@@ -1,2 +1,2 @@\n-    return x["key"]\n+    return x.get("key")'),
             ("assistant", '[tool_use Bash] {"command": "pytest -q tests"}'),
             ("user", "===== 3 passed in 0.12s =====")]
    msgs = [{"role": "user", "content": f"Fix the {topic} crash: {topic}_handler raises KeyError in {f} on empty input."}]
    msgs += [{"role": r, "content": c} for r, c in itertools.islice(itertools.cycle(steps), n - 2)]
    msgs.append({"role": "assistant", "content": f"Done: {topic}_handler now handles empty input."})
    for i, m in enumerate(msgs):
        if ts is not None:
            m["timestamp"] = ts + i * 1000
        if parts and i % 3 == 0:
            m["content"] = [{"type": "text", "text": m["content"]}]
    return msgs


def post(c, path, body):
    """json.dumps, not httpx's encoder: lone surrogates and Infinity are what Python clients really send."""
    return c.post(path, content=json.dumps(body), headers={**H, "content-type": "application/json"})


def add(c, body, want=200):
    r = post(c, "/add", body)
    assert r.status_code == want, (want, r.status_code, r.text[:300])
    if want == 200 and "request_id" in body:
        assert r.json()["request_id"] == body["request_id"] and r.json()["user_id"] == body["user_id"], r.text
    return r


def search(c, body, want=200):
    r = post(c, "/search", body)
    assert r.status_code == want, (want, r.status_code, r.text[:300])
    return r


def check(r, top_k=100, cards=True):
    """The contract of every coding Search response."""
    d = r.json()["data"]
    ids = [x["id"] for x in d]
    assert len(d) <= min(top_k, 5) and len(ids) == len(set(ids)), ids
    assert all(isinstance(x["id"], str) and x["id"] and isinstance(x["content"], str) and x["content"].strip()
               and isinstance(x["score"], (int, float)) for x in d), d
    assert [x["score"] for x in d] == sorted((x["score"] for x in d), reverse=True), d
    assert all(i.endswith("#card") or "#card~" in i for i in ids), ids
    assert not cards or d, "expected at least one card"
    return d


def main_rows(srv, uid):
    return srv.DB.execute("SELECT COUNT(*) FROM adds WHERE user_id=?", (uid,)).fetchone()[0]


def code_rows(srv, uid):
    return srv.CODE_DB.execute("SELECT COUNT(*) FROM adds WHERE user_id=?", (uid,)).fetchone()[0]


# ----------------------------------------------------------------------------------------------------- static
def static():
    tree = ast.parse((SVC / "codemem.py").read_text(encoding="utf-8"))
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not names & {"dashscope", "memsys", "mmmem", "server"}, names
    def rest(src):
        """dashscope.py outside embed() and its in-flight table (those: tests/test_fixes_v11.py)."""
        cut = min(i for i in (src.find("# Concurrent Adds"), src.find("def embed(")) if i >= 0)
        return src[:cut].replace("import Future, ThreadPoolExecutor", "import ThreadPoolExecutor") + src[src.index("_LLM_CACHE ="):]

    r = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "v1.0.3", "--", "service/timeannot.py"], capture_output=True)
    if r.returncode in (0, 1):
        assert r.returncode == 0, "timeannot must stay as in v1.0.3"                   # memsys: tests/test_lru.py
        old = subprocess.run(["git", "-C", str(ROOT), "show", "v1.0.3:service/dashscope.py"], capture_output=True, check=True, text=True)
        assert rest(old.stdout) == rest((SVC / "dashscope.py").read_text()), "dashscope must only differ in embed()"
    else:
        print("SKIP dashscope diff: git tag v1.0.3 not available")
    print("static: codemem imports no model client; timeannot unchanged, dashscope changed in embed() only")


# --------------------------------------------------------------------------------------------------- detector
def corpora():
    """Whole sessions of local public textual corpora, as one Add each (the worst case for the detector)."""
    p = ROOT.parent / "data_public" / "longmemeval_s" / "longmemeval_s_cleaned.json"
    if p.exists():
        for item in json.load(open(p, encoding="utf-8")):
            for s in item["haystack_sessions"]:
                yield [{"role": m["role"], "content": str(m["content"])} for m in s]
    p = ROOT.parent / "data" / "CL-bench.jsonl"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            if line.strip():
                yield [{"role": m["role"], "content": str(m["content"])} for m in json.loads(line)["messages"] if m["role"] != "system"]


def detector():
    import codemem
    hit = lambda msgs: codemem.is_trajectory_add({"user_id": "u", "messages": msgs})
    T = lambda s: {"role": "user", "content": s}
    mark, plain = T('[tool_use Bash] {"command": "ls"}'), T("we talked about the weather")
    assert hit([mark] * 5 + [plain] * 16), "21 messages, 5 marked"
    assert not hit([mark] * 4 + [plain] * 17), "fewer than 5 marked"
    assert not hit([mark] * 20), "20 messages can be an ordinary chunk"
    assert hit([mark] * 6 + [plain] * 24) and not hit([mark] * 5 + [plain] * 25), "20% of 30 messages"
    w = lambda n: T('[tool_use Bash] {"command": "ls"} ' + "word " * (n - 4))
    assert not hit([w(400)] * 5), "exactly 2,000 words can be an ordinary chunk"
    assert hit([w(400)] * 4 + [w(401)]), "2,001 words"
    assert not hit([w(800)] * 3 + [T("word " * 300)] * 7), "20% of 10 is 2, but 5 marked messages are the floor"
    assert not codemem.is_trajectory_add({"user_id": "u", "messages": [w(3000)]}, min_marked=1), "one message is never structural"
    assert codemem.is_trajectory_add({"user_id": "u", "messages": [w(1500), w(1501)]}, min_marked=1)
    pic = {"role": "user", "content": [{"type": "text", "text": "x"}, mm.img(mm.png(4, 4))]}
    assert not hit([mark] * 25 + [pic]), "an image rules the Add out"
    for k in ("[tool_use Edit] x", "tool_use_id: 7", "<tool_call>", "diff --git a/x b/x", "@@ -1,2 +1,3 @@", "+++ b/x.py",
              "Traceback (most recent call last):", '  File "a.py", line 3', "old_str", "exit code 1", "$ git status",
              "===== 2 failed, 1 passed in 3s =====", '"file_path": "a"', "Bash(command", "Edit(file_path"):
        assert hit([T(k)] * 5 + [plain] * 16), k
    for bad in (None, 5, "plain string", {"text": "x"}, [{"type": "text", "text": "y"}, 7], [None]):     # never raises
        hit([{"role": "user", "content": bad}, bad] * 11)

    talk = ["I installed Python and ran pytest on my laptop, it took a while.", "Can you recommend a good git tutorial?",
            "My sister gave me an old diff calculator.", "The exit of the highway is closed, take the next one.",
            "def is a keyword, right? My teacher said so.", "He said: file a report, line 3 was wrong.",
            "We read about Bash in class, and Grep is a funny word.", "I went to the gym and then cooked dinner.",
            "$ 500 was the price of the flight.", "Passed the exam: 3 passed, 1 failed."]
    samples = [[{"role": "user" if i % 2 else "assistant", "content": talk[i % len(talk)] * (1 + i % 3)} for i in range(n)]
               for n in (2, 20, 21, 60, 400)]
    samples += [[{"role": "user", "content": " ".join(talk) * 40}, {"role": "assistant", "content": " ".join(talk) * 40}]]
    fp = sum(hit(s) for s in samples)
    n = fp_c = 0
    for i, s in enumerate(corpora()):
        if s and i % 3 == 0:
            n += 1
            fp_c += hit(s)
    assert not fp and not fp_c, (fp, fp_c, n)

    t0 = time.process_time()                       # CPU time: a busy machine must not fail this
    hit([T("\n" * 199_000), T("= " * 99_000), T("-" * 199_000), T(" \n" * 99_000)] * 2)
    codemem.code_tokens("deadbeef" * 250)
    [codemem.code_tokens(s) for s in codemem.segments("deadbeef" * 62_500)]
    assert time.process_time() - t0 < 3, "quadratic regex"
    plain_re = re.compile(r"(?:[\w.\-]+/)+[\w.\-]+|\b[\w\-]+\." + codemem.EXT + r"\b")         # same tokens as without the guard
    for t in ("see src/a/b.py and setup.py, not a/b", "../x/y.c //a/b -c/d.e.f a.b-c/d/e https://x.org/a_b/c.md?q=1", "a-b.py c.d.py -e.py .f/g"):
        assert codemem.PATH_RE.findall(t) == plain_re.findall(t), t
    print(f"detector: thresholds exact; 0 false positives on {len(samples)} samples and {n} corpus sessions; no quadratic regex")


# ---------------------------------------------------------------------------------------------------- routing
def routing(tmp):
    work = os.path.join(tmp, "route")
    c, srv = boot("server_cr1", work)
    e0 = EMBEDS[0]
    for i in range(4):
        add(c, {"request_id": f"r{i}", "user_id": "CODE", "session_id": f"s{i}", "messages": traj(TOPICS[i], ts=1700000000000 + i * 86400000)})
    # sticky: later Adds of a coding user stay there whatever they look like (and duplicates are dropped)
    add(c, {"request_id": "e1", "user_id": "CODE", "session_id": "s9", "messages": [USER]})
    add(c, {"request_id": "e2", "user_id": "CODE", "messages": []})
    add(c, {"request_id": "r0", "user_id": "CODE", "session_id": "s0", "messages": traj(TOPICS[0])})
    assert "CODE" in srv.code_users and "CODE" not in srv.store.users and not main_rows(srv, "CODE") and code_rows(srv, "CODE") == 6
    q = {"user_id": "CODE", "query": "socket_timeout handler KeyError", "top_k": 100}
    d = check(search(c, q))
    assert d[0]["id"] == "s1#card" and "socket_timeout_handler raises KeyError" in d[0]["content"], d[0]
    assert "(edit):" in d[0]["content"] and "3 passed" in d[0]["content"] and "Done: socket_timeout" in d[0]["content"]
    assert EMBEDS[0] == e0, "coding path called the embedding client"

    # textual users: never rerouted, also when a later Add looks like a trajectory; ordinary chunks never reach the store
    add(c, {"request_id": "t1", "user_id": "TXT", "session_id": "a", "messages": [{"role": "user", "content": "I adopted a cat named Miso"}]})
    add(c, {"request_id": "t2", "user_id": "TXT", "session_id": "b", "messages": traj(TOPICS[4])})
    assert "TXT" in srv.text_users and "TXT" not in srv.code_users and main_rows(srv, "TXT") == 2 and not code_rows(srv, "TXT")
    r = search(c, {"user_id": "TXT", "query": "cat name", "top_k": 5}).json()
    assert r["data"] and "#card" not in r["data"][0]["id"]
    assert EMBEDS[0] > e0

    # the first Add with messages decides: an empty one does not pin the user
    add(c, {"request_id": "z0", "user_id": "EMPTY", "session_id": "z", "messages": []})
    add(c, {"request_id": "z1", "user_id": "EMPTY", "session_id": "z", "messages": traj(TOPICS[5])})
    assert "EMPTY" in srv.code_users
    # bad requests decide nothing
    add(c, {"user_id": "LATE", "messages": "oops"}, 422)
    add(c, {"request_id": "l", "user_id": "LATE", "messages": traj(TOPICS[5])})
    assert "LATE" in srv.code_users

    # image users never take the coding path: first Add with an image, even inside a trajectory
    pic = mm.png(8, 8, b"route")
    tmsgs = traj(TOPICS[6]) + [{"role": "user", "content": [{"type": "text", "text": "screenshot"}, mm.img(pic)]}]
    add(c, {"request_id": "m1", "user_id": "MM", "session_id": "ms", "messages": tmsgs})
    add(c, {"request_id": "m2", "user_id": "MM", "session_id": "ms2", "messages": traj(TOPICS[6])})
    assert "MM" in srv.img_users and "MM" not in srv.code_users and not code_rows(srv, "MM")
    d = search(c, {"user_id": "MM", "query": "screenshot", "top_k": 5}).json()["data"]
    assert isinstance(d[0]["content"], list)

    # a routed user that sends an image moves over with its earlier Adds: a broken image moves nothing
    add(c, {"request_id": "h1", "user_id": "HAND", "session_id": "h", "messages": traj(TOPICS[1])})
    add(c, {"request_id": "h2", "user_id": "HAND", "session_id": "h2", "messages": traj(TOPICS[2])})
    bad = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@"}}]}
    add(c, {"request_id": "h3", "user_id": "HAND", "session_id": "h3", "messages": [bad]}, 422)
    assert "HAND" in srv.code_users and code_rows(srv, "HAND") == 2 and not main_rows(srv, "HAND")
    add(c, {"request_id": "h4", "user_id": "HAND", "session_id": "h4", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "the dashboard screenshot"}, mm.img(pic)]}]})
    assert "HAND" in srv.img_users and "HAND" not in srv.code_users and "HAND" not in srv.code_store.users
    assert not code_rows(srv, "HAND") and main_rows(srv, "HAND") == 3
    add(c, {"request_id": "h1", "user_id": "HAND", "session_id": "h", "messages": traj(TOPICS[1])})      # retry after the move
    assert main_rows(srv, "HAND") == 3
    hq = {"user_id": "HAND", "query": "socket_timeout_handler raises KeyError", "top_k": 100}
    d = search(c, hq).json()["data"]
    assert isinstance(d[0]["content"], list) and "socket_timeout_handler" in json.dumps(d), "earlier Adds lost in the move"
    # ... also when the process dies between the commit and the clean-up of the coding store
    add(c, {"request_id": "x1", "user_id": "CRASH", "session_id": "x", "messages": traj(TOPICS[1])})
    forget, srv.code_store.forget = srv.code_store.forget, lambda u: 1 / 0
    add(c, {"request_id": "x2", "user_id": "CRASH", "session_id": "x2", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "photo"}, mm.img(pic)]}]}, 500)
    srv.code_store.forget = forget
    assert code_rows(srv, "CRASH") == 1 and main_rows(srv, "CRASH") == 2

    # restart: routes come back from the two SQLite files, nothing of the coding path is replayed
    R = search(c, q).content
    e1 = EMBEDS[0]
    c2, srv2 = boot("server_cr2", work)
    assert {"CODE", "EMPTY", "LATE"} <= srv2.code_users and not {"TXT", "MM", "HAND", "CRASH"} & srv2.code_users
    assert {"TXT", "MM", "HAND", "CRASH"} <= srv2.text_users and {"MM", "HAND", "CRASH"} <= srv2.img_users
    assert "CODE" not in srv2.store.users and "CODE" not in srv2.code_store.users and not srv2.code_store.users
    assert search(c2, q).content == R, "restart changed coding results"
    assert EMBEDS[0] == e1, "restart re-embedded a coding user"
    add(c2, {"request_id": "t3", "user_id": "TXT", "session_id": "c", "messages": traj(TOPICS[5])})      # sticky after restart
    add(c2, {"request_id": "e3", "user_id": "CODE", "session_id": "s10", "messages": [USER]})
    assert main_rows(srv2, "TXT") == 3 and not main_rows(srv2, "CODE") and code_rows(srv2, "CODE") == 7
    r = search(c2, {"user_id": "CRASH", "query": "photo", "top_k": 5}).json()["data"]
    assert isinstance(r[0]["content"], list), "a half-finished move must end on the image path"
    print("routing: sticky; image users off the coding path, a routed user moves over with its Adds; restart restores all routes")

    # switch: no new user is routed, trajectory users take the v1.0.x path; users already routed keep their data
    c3, srv3 = boot("server_cr3", work, detect="0")
    add(c3, {"request_id": "k", "user_id": "KILL", "session_id": "s", "messages": traj(TOPICS[0])})
    assert "KILL" in srv3.store.users and "KILL" not in srv3.code_users and main_rows(srv3, "KILL") == 1 and not code_rows(srv3, "KILL")
    assert "#card" not in search(c3, {"user_id": "KILL", "query": "parse_config", "top_k": 5}).json()["data"][0]["id"]
    n = code_rows(srv3, "CODE")
    add(c3, {"request_id": "e4", "user_id": "CODE", "session_id": "s11", "messages": [USER]})
    assert code_rows(srv3, "CODE") == n + 1 and not main_rows(srv3, "CODE")
    assert check(search(c3, q))[0]["id"] == "s1#card"
    print("switch: AML_CODE_DETECT=0 sends new trajectory users down the v1.0.x path; routed users keep being served")

    # concurrent first Adds of one user get one decision, plain and trajectory shapes mixed in any order
    import codemem
    c4, srv4 = boot("server_cr4", os.path.join(tmp, "race"))
    gate, detect = threading.Barrier(6), codemem.is_trajectory_add
    codemem.is_trajectory_add = lambda req: (detect(req), gate.wait())[0]    # all six past the fast path, none decided
    for round_ in range(30):
        uid = f"R{round_}"
        bodies = [{"request_id": f"{uid}:{i}", "user_id": uid, "session_id": f"s{i}", "messages": traj(TOPICS[i]) if i % 2 else [USER]}
                  for i in range(6)]
        with ThreadPoolExecutor(6) as ex:
            assert all(r.status_code == 200 for r in ex.map(lambda b: c4.post("/add", json=b, headers=H), bodies))
        assert (uid in srv4.code_users) != (uid in srv4.text_users), uid
        assert (code_rows(srv4, uid), main_rows(srv4, uid)) in ((6, 0), (0, 6)), uid
    codemem.is_trajectory_add = detect
    assert srv4.code_users and srv4.text_users, "both outcomes must have occurred"
    print(f"race: 30 users, 6 concurrent first Adds each: one route per user ({len(srv4.code_users)} coding, {len(srv4.text_users)} text)")


# --------------------------------------------------------------------------------------------------- contract
def contract(tmp):
    work = os.path.join(tmp, "contract")
    c, srv = boot("server_cc", work)
    e0 = EMBEDS[0]
    add(c, {"request_id": "a0", "user_id": "U", "session_id": "s0", "messages": traj("parse_config", parts=True)})
    cases = [
        {"request_id": "a1", "user_id": "U", "session_id": "s1", "messages": [{"role": "user", "content": "Fix KeyError in src/conf/loader.py parse_config"},
         {"role": "assistant", "content": [{"type": "text", "text": '[tool_use Edit] {"file_path": "src/conf/loader.py", "old_string": "if key:", "new_string": "if key is not None:"}'}]},
         {"role": "tool", "content": "===== 3 passed in 0.1s ====="}, {"role": "assistant", "content": "Done: empty keys handled."}]},
        {"request_id": "a2", "user_id": "U", "session_id": "s2", "messages": [{"role": "user", "content": None}, "not-a-dict", 7, None,
         {"role": "user", "content": [{"text": "no type"}, {"type": "other", "text": "dropped"}, 7, None]},
         {"role": None, "content": {"type": "text", "text": "dict content"}}, {"content": "no role", "timestamp": "abc"},
         {"role": 5, "content": 12345}, {"role": ["x"], "content": True}, {"role": "user", "content": "inf time", "timestamp": 1e400}]},
        {"request_id": "a3", "user_id": "U", "session_id": "s3", "messages": [{"role": "user", "content": "Huge output follows"},
         {"role": "user", "content": "x = 1\n" * 900_000}]},
        {"request_id": "a4", "user_id": "U", "session_id": "s4", "messages": [
            {"role": "user" if i % 2 else "assistant", "content": f"step {i} grep parse_config"} for i in range(600)]},
        {"request_id": "a5", "user_id": "U", "messages": []},
        {"user_id": "U", "session_id": "s6", "messages": [{"role": "user", "content": "no request id"}]},
        {"user_id": "U", "session_id": 7, "request_id": 8, "messages": [{"role": "user", "content": "numeric ids"}]},
        {"user_id": "U", "session_id": "s8", "request_id": "", "messages": [{"role": "user", "content": "empty request id one"}]},
        {"user_id": "U", "session_id": "s9", "request_id": "", "messages": [{"role": "user", "content": "empty request id two"}]},
        {"user_id": "U", "session_id": "s10", "request_id": ["odd"], "messages": [{"role": "user", "content": "list request id"}]},
    ]
    for b in cases:
        r = add(c, b)
        assert r.json()["success"] and r.json().get("request_id") == b.get("request_id"), r.text
    n = code_rows(srv, "U")
    assert n == 1 + len(cases)
    for b in (cases[1], cases[-1]):       # idempotent retries
        add(c, b)
    assert code_rows(srv, "U") == n
    add(c, {"messages": []}, 422)
    add(c, {"user_id": "", "messages": []}, 422)
    add(c, {"user_id": "U"}, 422)
    assert c.post("/add", json=cases[0]).status_code == 401 and c.post("/search", json={"user_id": "U", "query": "x"}).status_code == 401

    for q, k, cards in [("parse_config KeyError loader.py", 100, True), ([{"type": "text", "text": "empty keys"}], 3, True),
                        ("parse_config", 1, True), ("parse_config", True, True), ("parse_config", 2.9, True), ("parse_config", "3", True),
                        ("parse_config", "abc", True), ("parse_config", None, True), ("parse_config", [1], True), ("parse_config", 1e400, True),
                        ("", 100, False), ("   ", 5, False), ([], 5, False), (None, 5, False), ([{"type": "image_url", "image_url": {}}], 5, False),
                        ("zzzz qqqq", 100, False), ("?!", 100, False), ("parse_config", 0, False), ("parse_config", -3, False),
                        ({"type": "text", "text": "parse_config"}, 5, True), (12345, 5, True), ("x" * 2_000_000, 5, False)]:
        limit = 3 if k == "3" else 100 if not isinstance(k, (int, float)) or k == 1e400 else int(k)
        body = {"user_id": "U", "query": q}
        if k is not None:
            body["top_k"] = k
        d = check(post(c, "/search", body), max(limit, 0), cards)
        if limit <= 0:
            assert not d
    assert search(c, {"user_id": "nobody", "query": "x", "top_k": 5}).json() == {"data": []}
    search(c, {"query": "x"}, 422)
    search(c, {"user_id": "", "query": "x"}, 422)

    # lone surrogates are legal JSON and do occur in tool output cut in the middle of a pair
    add(c, {"request_id": "sur", "user_id": "U", "session_id": "sur", "messages": [
        {"role": "user", "content": "surrogate_marker \ud83d in a tool output"}, {"role": "assistant", "content": "second \ude00 one"}]})
    assert code_rows(srv, "U") == n + 1
    d = check(search(c, {"user_id": "U", "query": "surrogate_marker", "top_k": 5}))
    assert any("surrogate_marker" in x["content"] for x in d)
    assert EMBEDS[0] == e0, "coding path called the embedding client"
    print("contract: content / query / top_k shapes, ids, counts, surrogates, 422 / 401 ok; 0 embedding calls")


def shapes(tmp):
    import codemem
    work = os.path.join(tmp, "shapes")
    c, srv = boot("server_cs", work)
    # M3: one session_id, several Adds -> each Add is a trajectory
    for i in range(3):
        add(c, {"request_id": f"b{i}", "user_id": "V", "session_id": "same", "messages": traj(TOPICS[i])})
    d = check(search(c, {"user_id": "V", "query": "socket_timeout crash", "top_k": 100}))
    assert d[0]["id"] == "same~1#card", [x["id"] for x in d]
    # several session_ids -> one trajectory per session_id, in arrival order even when the timestamps disagree
    t = 1700000000000
    add(c, {"request_id": "c0", "user_id": "W", "session_id": "a", "messages": traj("parse_config", ts=t + 10_000_000)})
    add(c, {"request_id": "c1", "user_id": "W", "session_id": "b", "messages": traj("csv_export", ts=t)})
    add(c, {"request_id": "c2", "user_id": "W", "session_id": "a", "messages": [{"role": "user", "content": "follow-up about parse_config"},
                                                                                 {"role": "assistant", "content": "second part of the a session", "timestamp": t}]})
    d = check(search(c, {"user_id": "W", "query": "parse_config crash", "top_k": 100}))
    assert d[0]["id"] == "a#card" and "32 messages" in d[0]["content"] and d[0]["content"].splitlines()[1].startswith("user: Fix the parse_config")
    assert "second part of the a session" in d[0]["content"], "the last assistant text of the merged unit closes the card"
    assert d[0]["created_at"] == time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime((t + 10_000_000) / 1000))
    # the excerpt that matched is on the card, also when it is a whole short message of a longer trajectory
    msgs = traj("csv_export")
    msgs[7]["content"] = "note_marker: the delimiter comes from the locale"
    add(c, {"request_id": "x0", "user_id": "X", "session_id": "s", "messages": msgs})
    d = check(search(c, {"user_id": "X", "query": "note_marker delimiter locale", "top_k": 5}))
    assert "(matching excerpt): note_marker: the delimiter" in d[0]["content"], d[0]["content"]
    # a trajectory squeezed into one giant message still gives a card, with the part that matched
    add(c, {"request_id": "g0", "user_id": "G", "session_id": "first", "messages": traj("csv_export")})
    giant = "\n".join(f"step {i}: looked at module_{i}.py" for i in range(60_000)) + "\nneedle_function is defined here\n"
    add(c, {"request_id": "g1", "user_id": "G", "session_id": "giant", "messages": [{"role": "user", "content": giant}]})
    d = check(search(c, {"user_id": "G", "query": "needle_function", "top_k": 5}))
    assert d[0]["id"] == "giant#card" and "(matching excerpt):" in d[0]["content"] and "needle_function" in d[0]["content"]
    assert len(d[0]["content"]) < 12_000, "a card is a few hundred tokens whatever the message size"

    # budget and card count (S1 cut off: every unit matches)
    db = sqlite3.connect(os.path.join(work, "x.sqlite"), check_same_thread=False)
    db.execute("CREATE TABLE adds (user_id TEXT, request_id TEXT, body TEXT, seq INTEGER PRIMARY KEY AUTOINCREMENT)")
    db.execute("CREATE UNIQUE INDEX adds_req ON adds(user_id, request_id)")
    lock = threading.Lock()
    new = lambda **k: codemem.CodeStore(db, lock, codemem.CodeCfg(prebuild_delay=0, **k))
    st = new()
    for i, tp in enumerate(TOPICS):
        st.add({"request_id": f"r{i}", "user_id": "B", "session_id": f"s{i}", "messages": traj(tp) + [{"role": "assistant", "content": "shared_word " + "y " * 900}]})
    q = {"user_id": "B", "query": "shared_word crash handler", "top_k": 100}
    assert len(new(rel_cut=0).search(q)["data"]) == 5, "at most 5 cards"
    assert len(new(rel_cut=0, max_cards=9).search(q)["data"]) == len(TOPICS)
    one = new(rel_cut=0, budget=10).search(q)["data"]
    assert len(one) == 1, "the first card comes whatever the budget"
    d = new(rel_cut=0, budget=2 * codemem.ntok(one[0]["content"]) + 40).search(q)["data"]
    assert len(d) == 2 and sum(codemem.ntok(x["content"]) for x in d) <= 2 * codemem.ntok(one[0]["content"]) + 40
    assert sum(codemem.ntok(x["content"]) for x in new(rel_cut=0).search(q)["data"]) <= 8000

    # S1: a unit that matches only in a tool output, far below the best task match, loses its card
    st.add({"request_id": "n0", "user_id": "N", "session_id": "gold", "messages": [
        {"role": "user", "content": "Fix the retry_backoff bug in the connection pool, retry_backoff overflows"}, {"role": "assistant", "content": "Done."}]})
    for i in range(3):
        st.add({"request_id": f"n{i + 1}", "user_id": "N", "session_id": f"noise{i}", "messages": [
            {"role": "user", "content": f"Rename the {TOPICS[i]} module"}, {"role": "user", "content": "log line: retry_backoff=3"}, {"role": "assistant", "content": "Done."}]})
    q = {"user_id": "N", "query": "retry_backoff overflows", "top_k": 100}
    assert [x["id"] for x in new(rel_cut=0).search(q)["data"]][0] == "gold#card" and len(new(rel_cut=0).search(q)["data"]) == 4
    assert [x["id"] for x in new().search(q)["data"]] == ["gold#card"], "default rel_cut=0.5"
    print("shapes: one session per Add, merged sessions, giant message, max 5 cards, 8k budget, rel_cut=0.5")


# ----------------------------------------------------------------------------------------------- fault injection
def faults(tmp):
    work = os.path.join(tmp, "faults")
    c, srv = boot("server_cf", work)
    import codemem
    canary = "CANARY_query_text_never_logged"
    add(c, {"request_id": "1", "user_id": "U", "session_id": "s", "messages": traj("parse_config")})
    q = {"user_id": "U", "query": f"parse_config {canary}", "top_k": 5}
    good = search(c, q).content
    boom = lambda *a, **k: 1 / 0
    for name in ("rank", "card", "_get"):
        orig = getattr(codemem.CodeStore, name)
        setattr(codemem.CodeStore, name, boom)
        r = search(c, q)
        setattr(codemem.CodeStore, name, orig)
        assert r.status_code == 200 and r.json() == {"data": []}, (name, r.text)
    assert search(c, q).content == good, "the store must recover"
    # a response that cannot be serialised is one more internal failure
    orig = srv.code_store.card
    srv.code_store.card = lambda *a: "bad \ud800"
    assert search(c, q).json() == {"data": []}
    srv.code_store.card = orig

    orig = codemem.parse_add

    def bad_parse(req):
        if "poison" in json.dumps(req):
            raise RuntimeError("cannot parse")
        return orig(req)

    codemem.parse_add = bad_parse
    add(c, {"request_id": "2", "user_id": "U", "session_id": "s2", "messages": [{"role": "user", "content": "poison csv_export"}]})
    assert code_rows(srv, "U") == 2 and search(c, q).content == good
    for i, t in enumerate(("parse_config", "poison", "csv_export")):
        add(c, {"request_id": f"o{i}", "user_id": "O", "session_id": "same", "messages": traj(t)})
    oq = {"user_id": "O", "query": "csv_export crash", "top_k": 5}
    live = search(c, oq).content
    assert json.loads(live)["data"][0]["id"] == "same~2#card"
    c2, _ = boot("server_cf2", work)
    assert search(c2, oq).content == live, "a row that cannot be parsed must not renumber the units after a restart"
    codemem.parse_add = orig
    c2, _ = boot("server_cf2b", work)
    assert "poison csv_export" in search(c2, {"user_id": "U", "query": "csv_export poison", "top_k": 5}).text, "the row must survive"

    # SQLite read failure for a user that must be reloaded, log failure, both harmless
    c3, srv3 = boot("server_cf3", work)
    srv3.code_store._load = boom
    assert search(c3, q).json() == {"data": []}
    del srv3.code_store._load
    assert check(search(c3, q))
    srv3.CODE_LOG = Path(work)         # a directory: the write fails
    add(c3, {"request_id": "3", "user_id": "U", "session_id": "s3", "messages": [USER]})
    check(search(c3, q))
    srv3.CODE_LOG = Path(os.path.join(work, "coding_shape.jsonl"))
    # a write that cannot reach SQLite is a real failure: retried by the platform, nothing half-done
    insert = srv3.code_store.db
    srv3.code_store.db = None
    assert c3.post("/add", json={"request_id": "4", "user_id": "U", "session_id": "s4", "messages": [USER]}, headers=H).status_code == 500
    srv3.code_store.db = insert
    add(c3, {"request_id": "4", "user_id": "U", "session_id": "s4", "messages": [USER]})

    text = open(os.path.join(work, "coding_shape.jsonl"), encoding="utf-8").read()
    assert text.count('"error": "ZeroDivisionError"') == 4 and text.count('"error": "UnicodeEncodeError"') == 1, text
    assert canary not in text and "parse_config" not in text
    print("faults: Search degrades to 200 [] on any internal failure, unparsable Adds stay in SQLite, log failures are harmless")


# ------------------------------------------------------------------------------------- structure log, restart, LRU
def structure_log(tmp):
    work = os.path.join(tmp, "slog")
    log = os.path.join(work, "shape.jsonl")
    c, srv = boot("server_cl", work, log=log)
    add(c, {"request_id": "1", "user_id": "SECRET_USER", "session_id": "SECRET_SESSION", "extra": 1, "messages": traj("parse_config", ts=1700000000000, parts=True)})
    add(c, {"request_id": "2", "user_id": "SECRET_USER", "session_id": "SECRET_SESSION2", "messages": [{"role": "tool", "content": "SECRET_TEXT"}, "x"]})
    add(c, {"request_id": "3", "user_id": "SECRET_USER", "session_id": "SECRET_SESSION2", "SECRET key text": 1,
            "messages": [{"role": "SECRET role text", "content": "x"}, {"role": {"SECRET_ROLE": 1}, "content": "x"}]})
    search(c, {"user_id": "SECRET_USER", "query": "parse_config SECRET_QUERY", "top_k": 7, "options": ["SECRET_OPTION"]})
    search(c, {"user_id": "SECRET_USER", "query": [{"type": "text", "text": "SECRET_PART"}], "top_k": "x", "foo": 1})
    recs = [json.loads(l) for l in open(log, encoding="utf-8")]
    assert "SECRET" not in open(log, encoding="utf-8").read() and "parse_config" not in open(log, encoding="utf-8").read()
    a, s = [r for r in recs if r["op"] == "add"], [r for r in recs if r["op"] == "search"]
    assert a[2]["roles"] == {srv.sha12("SECRET role text"): 1, srv.sha12(str({"SECRET_ROLE": 1})): 1}
    assert a[2]["extra_keys"] == [srv.sha12("SECRET key text")]
    assert a[0]["n_msgs"] == 30 and a[0]["roles"] == {"user": 15, "assistant": 15} and a[0]["list_content"] == 10 and a[0]["has_ts"] == 30
    assert a[0]["extra_keys"] == ["extra"] and a[0]["u"] == a[1]["u"] and a[0]["sid"] != a[1]["sid"] and a[1]["roles"] == {"tool": 1, "str": 1}
    assert s[0]["top_k"] == 7 and s[0]["q_type"] == "str" and s[0]["n"] >= 1 and s[0]["chars"] > 0 and s[1]["top_k"] == "str" and s[1]["q_type"] == "list"
    assert s[1]["extra_keys"] == ["foo"] and s[0]["u"] == a[0]["u"] and all(r["error"] is None for r in s)
    assert s[0]["u"] == srv.sha12("SECRET_USER") and s[0]["q_sha"] == srv.sha12("parse_config SECRET_QUERY") and s[0]["n_options"] == 1
    assert s[1]["q_sha"] == srv.sha12("SECRET_PART") and s[1]["q_chars"] == 11 and s[1]["n_options"] == 0
    assert not os.path.exists(os.path.join(work, "search_log.jsonl")), "the query-logging path must not see coding users"
    c2, srv2 = boot("server_cl2", os.path.join(tmp, "slog2"), log="")
    add(c2, {"request_id": "1", "user_id": "U", "session_id": "s", "messages": traj("parse_config")})
    check(search(c2, {"user_id": "U", "query": "parse_config", "top_k": 7}))
    assert not os.path.exists(os.path.join(tmp, "slog2", "coding_shape.jsonl")) and not os.path.exists("coding_shape.jsonl")
    print("structure log: counts and hashed ids only; AML_CODE_LOG='' turns it off")


def restart_lru(tmp):
    import codemem
    work = os.path.join(tmp, "lru")
    c, srv = boot("server_cz1", work)
    qs = []
    for u in range(4):
        for i in range(3):
            add(c, {"request_id": f"r{u}{i}", "user_id": f"L{u}", "session_id": f"s{i}", "messages": traj(TOPICS[(u + i) % len(TOPICS)], ts=1700000000000 + i)})
        qs += [{"user_id": f"L{u}", "query": f"{TOPICS[u]} crash", "top_k": 100}, {"user_id": f"L{u}", "query": "handler KeyError empty input", "top_k": 2}]
    R = [search(c, q).content for q in qs]

    # restart with a cap of one user: every Search evicts and reloads, alone and concurrently
    c2, srv2 = boot("server_cz2", work, cfg={"prebuild_delay": 0, "max_users": 1})
    for i in [5, 0, 3, 7, 2, 0, 6, 1, 4]:
        assert search(c2, qs[i]).content == R[i], f"reload changed results ({i})"
        assert len(srv2.code_store.users) <= 1
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda i: search(c2, qs[i % len(qs)]).content, range(48)))
    assert all(b == R[i % len(qs)] for i, b in enumerate(out)), "concurrent reloads differ"

    # token cap too
    c3, srv3 = boot("server_cz3", work, cfg={"prebuild_delay": 0, "max_tokens": 1})
    assert [search(c3, q).content for q in qs] == R and len(srv3.code_store.users) == 1

    # an Add before any Search after a restart, and an Add after the index exists: same bytes as a fresh load
    c4, srv4 = boot("server_cz4", work)
    add(c4, {"request_id": "n1", "user_id": "L0", "session_id": "s7", "messages": traj("oauth_scope", ts=1700000000000)})
    assert not srv4.code_store.users["L0"].complete
    q = {"user_id": "L0", "query": "oauth_scope crash", "top_k": 100}
    r = check(search(c4, q))
    assert r[0]["id"] == "s7#card" and len(r) >= 1 and srv4.code_store.users["L0"].complete
    add(c4, {"request_id": "n2", "user_id": "L0", "session_id": "s8", "messages": traj("cache_eviction", ts=1700000000000)})
    a = search(c4, q).content
    q2 = {"user_id": "L0", "query": "cache_eviction crash", "top_k": 100}
    assert check(search(c4, q2))[0]["id"] == "s8#card"
    c5, _ = boot("server_cz5", work)
    assert search(c5, q).content == a and search(c5, q2).content == search(c4, q2).content, "incremental differs from a fresh load"

    # an Add that lands while the user is being read, with eviction pressure from other users, is not lost
    work2 = os.path.join(tmp, "lru2")
    c6, srv6 = boot("server_cz6", work2, cfg={"prebuild_delay": 0, "max_users": 1})
    st = srv6.code_store
    add(c6, {"request_id": "p0", "user_id": "P", "session_id": "s0", "messages": traj("parse_config")})
    add(c6, {"request_id": "o0", "user_id": "O", "session_id": "s0", "messages": traj("csv_export")})
    assert "P" not in st.users
    read = st._load

    def racing_load(uid):
        u = read(uid)
        st._load = read
        st.add({"request_id": "p1", "user_id": "P", "session_id": "s1", "messages": traj("token_refresh")})
        st.add({"request_id": "o1", "user_id": "O", "session_id": "s1", "messages": traj("csv_export")})
        return u

    st._load = racing_load
    d = check(search(c6, {"user_id": "P", "query": "token_refresh crash", "top_k": 5}))
    assert d[0]["id"] == "s1#card", "an Add that raced the reload was lost"
    q = {"user_id": "O", "query": "csv_export crash", "top_k": 5}
    a = search(c6, q).content
    uc = st.users["O"]
    uc.adds.append(uc.adds[-1])                    # the Add was committed, read by a reload, then appended by its thread
    uc.version += 1
    assert search(c6, q).content == a, "a record seen twice doubled its unit"

    # 8 threads adding and searching 6 users under a cap of 2: nothing lost, final state equals a fresh load
    work3 = os.path.join(tmp, "lru3")
    c7, srv7 = boot("server_cz7", work3, cfg={"prebuild_delay": 0.01, "max_users": 2})

    def job(i):
        u = f"T{i % 6}"
        add(c7, {"request_id": f"j{i}", "user_id": u, "session_id": f"s{i}", "messages": traj(TOPICS[i % len(TOPICS)])})
        check(search(c7, {"user_id": u, "query": f"{TOPICS[i % len(TOPICS)]} crash", "top_k": 5}))

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(job, range(48)))
    c8, _ = boot("server_cz8", work3)
    for u in range(6):
        q = {"user_id": f"T{u}", "query": "handler crash KeyError", "top_k": 100}
        assert code_rows(srv7, f"T{u}") == 8 and search(c7, q).content == search(c8, q).content, u
    print("restart / LRU: lazy reloads, eviction, racing and concurrent Adds all give byte-identical results")


def prebuild(tmp):
    work = os.path.join(tmp, "prebuild")
    c, srv = boot("server_cp", work, cfg={"prebuild_delay": 0.05})
    st = srv.code_store
    for i in range(3):
        add(c, {"request_id": f"r{i}", "user_id": "U", "session_id": f"s{i}", "messages": traj(TOPICS[i])})

    def wait(ok):
        for _ in range(500):
            if ok():
                return True
            time.sleep(0.01)
        return False

    assert wait(lambda: st.users["U"].built_version == st.users["U"].version), "index not built in the background"
    assert not st._queued and st._q.empty()
    built = st.users["U"].built
    check(search(c, {"user_id": "U", "query": "socket_timeout", "top_k": 5}))
    assert st.users["U"].built is built, "Search must use the prebuilt index"
    # an evicted user is left to its own Search: the background thread would thrash the LRU
    reads, read = [], st._load
    st._load = lambda u: reads.append(u) or read(u)
    with st.lock:
        st.users.pop("U")
    st._q.put("U")
    assert wait(st._q.empty)
    time.sleep(0.1)
    assert not reads and "U" not in st.users
    check(search(c, {"user_id": "U", "query": "socket_timeout", "top_k": 5}))
    assert reads == ["U"]
    print("prebuild: index built after the Adds pause, used by Search; evicted users are not reloaded in the background")


# ------------------------------------------------------------------------------- textual byte identity (worker mode)
def trajectory_user_ops():
    return [("add", {"request_id": "k1", "user_id": "tK", "session_id": "k", "messages": traj("parse_config", ts=1700000000000)}),
            ("add", {"request_id": "k2", "user_id": "tK", "session_id": "k2", "messages": traj("csv_export", parts=True)}),
            ("search", {"user_id": "tK", "query": "parse_config crash", "top_k": 100}),
            ("search", {"user_id": "tK", "query": "", "top_k": 5})]


def worker(svc, work, out, mode):
    """mode: v103 (no coding users) | on (coding users interleaved) | off (AML_CODE_DETECT=0)."""
    os.environ.update(AML_CODE_DETECT="0" if mode == "off" else "1", AML_CFG_CODING=json.dumps({"prebuild_delay": 0.01}))
    # a textual user whose later Adds look like trajectories stays textual
    sticky = [("add", {"request_id": "st1", "user_id": "tS", "session_id": "a", "messages": [USER]}),
              ("add", {"request_id": "st2", "user_id": "tS", "session_id": "b", "messages": traj("parse_config")}),
              ("search", {"user_id": "tS", "query": "parse_config crash and coffee in Tokyo", "top_k": 100})]
    ops = [(o, b, "t") for o, b in mm.text_ops() + sticky] + [(o, b, "k") for o, b in trajectory_user_ops()]
    if mode == "on":
        mixed = []
        for i, op in enumerate(ops):
            mixed.append(op)
            if i % 5 == 0:
                u = f"c{i % 3}"
                mixed += [("add", {"request_id": f"cr{i}", "user_id": u, "session_id": f"cs{i}", "messages": traj(TOPICS[i % len(TOPICS)])}, "c"),
                          ("search", {"user_id": u, "query": f"{TOPICS[i % len(TOPICS)]} crash", "top_k": 100}, "c")]
        ops = mixed
    with open(out, "w", encoding="utf-8") as f, open(out + ".k", "w", encoding="utf-8") as fk:
        for phase in ("first", "restart"):           # restart: SQLite replay, then the same requests again
            for m in ("server", "memsys", "mmmem", "timeannot", "codemem"):
                sys.modules.pop(m, None)
            c, _ = mm.load(svc, "server", work)
            n = 0
            for op, body, tag in ops:
                r = c.post("/" + op, json=body, headers=H)
                if tag == "c":
                    assert r.status_code == 200 and "#card" in r.text + "#card", r.text[:200]
                else:
                    (f if tag == "t" else fk).write(json.dumps({"phase": phase, "k": n, "status": r.status_code, "body": r.text}, ensure_ascii=False) + "\n")
                    n += 1
    sys.exit(3 if mm.NET else 0)


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
    runs = {"v103": base, "on": str(SVC), "off": str(SVC)}
    procs = {k: subprocess.Popen([sys.executable, __file__, "--dump", d, os.path.join(tmp, "w" + k), os.path.join(tmp, k + ".jsonl"), k])
             for k, d in runs.items()}
    for k, p in procs.items():
        assert p.wait() == 0, f"worker {k} failed"
    ref = open(os.path.join(tmp, "v103.jsonl"), "rb").read()
    n = ref.count(b"\n")
    for k in ("on", "off"):
        assert open(os.path.join(tmp, k + ".jsonl"), "rb").read() == ref, f"textual responses with detection {k} differ from v1.0.3"
    ref_k = open(os.path.join(tmp, "v103.jsonl.k"), "rb").read()
    assert open(os.path.join(tmp, "off.jsonl.k"), "rb").read() == ref_k, "AML_CODE_DETECT=0 differs from v1.0.3 for a trajectory user"
    assert b"#card" in open(os.path.join(tmp, "on.jsonl.k"), "rb").read() and b"#card" not in ref_k
    print(f"textual users: {n}/{n} responses byte-identical to v1.0.3 (coding users interleaved, switch on and off); "
          f"AML_CODE_DETECT=0: {ref_k.count(bytes([10]))}/{ref_k.count(bytes([10]))} trajectory-user responses as v1.0.3")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--dump"]:
        worker(*sys.argv[2:6])
    sys.path.insert(0, str(SVC))
    tmp = tempfile.mkdtemp()
    try:
        static()
        detector()
        routing(tmp)
        contract(tmp)
        shapes(tmp)
        faults(tmp)
        structure_log(tmp)
        restart_lru(tmp)
        prebuild(tmp)
        textual_identical(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert not mm.NET, mm.NET
    print("no network calls; all coding v1.1 checks passed")
