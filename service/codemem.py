"""Coding path (v1.1). Only users whose first Add is an agent trajectory come here (is_trajectory_add, server.py).

Past agent trajectories arrive through Add and are stored verbatim in the coding SQLite. Search ranks the user's
trajectories ("history units") with code-aware BM25 (segment channel + per-unit key document, fused by RRF) and
returns one verbatim "history card" per top unit under a small token budget. No embedding model and no LLM are
called at Add or Search time. The SQLite file is the source of truth: indexes are built lazily (in the background
after a pause in a user's Adds, else on its first Search) and evicted LRU, so a restart needs no replay.
"""
from __future__ import annotations

import datetime as dt
import json
import queue
import re
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

import bm25s
import numpy as np
import Stemmer
import tiktoken

ENC = tiktoken.get_encoding("o200k_base")
STEM = Stemmer.Stemmer("english")

# ----------------------------------------------------------------------------------------------- tokenizer
STOP = set("""a an the and or but if then else of to in on at by for with from as is are was were be been being this
that these those it its into not no yes do does did done can could should would will shall may might must have has had
i you we they he she them our your their my me us so such than too very just also only about over under after before
above below up down out off again further here there when where why how all any both each few more most other some own
same what which who whom whose""".split())
EXT = (r"(?:py|pyi|pyx|js|mjs|cjs|ts|tsx|jsx|java|go|rs|c|cc|cpp|cxx|h|hh|hpp|rb|php|cs|kt|kts|scala|swift|m|mm|sh|bash|"
       r"zsh|ps1|yaml|yml|json|toml|cfg|ini|conf|md|rst|txt|html|htm|css|scss|sql|proto|gradle|xml|lock|vue|svelte|"
       r"dart|lua|r|jl|ex|exs|erl|hs|ml|clj)")
PATH_RE = re.compile(r"(?<![\w.\-])(?:[\w.\-]+/)+[\w.\-]+|\b[\w\-]+\." + EXT + r"\b")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:(?:\.|::|->|#)[A-Za-z_][A-Za-z0-9_]*)*|[0-9]+(?:\.[0-9]+)*|[㐀-鿿]")
CAMEL1 = re.compile(r"([a-z0-9])([A-Z])")
CAMEL2 = re.compile(r"([A-Z]+)([A-Z][a-z])")
SPLIT_RE = re.compile(r"[_\s]+|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")
IDENT_RE = re.compile(r"[a-z][A-Z]|[A-Z]{2,}[a-z]|[A-Za-z][0-9]")
QUAL_RE = re.compile(r"\.|::|->|#")
_stem_cache: dict[str, str] = {}


def _stem(w: str) -> str:
    s = _stem_cache.get(w)
    if s is None:
        s = STEM.stemWord(w)
        if len(_stem_cache) < 500_000:
            _stem_cache[w] = s
    return s


def code_tokens(text: str) -> list[str]:
    """Paths -> path:/file: exact tokens; identifiers kept whole (lower-case, unstemmed) plus snake/camel parts
    (stemmed); qualified names kept whole plus parts; English words stemmed; CJK as bigrams."""
    out: list[str] = []
    for p in PATH_RE.findall(text):
        lp = p.lower().strip("./")
        if not lp or len(lp) > 200:
            continue
        out.append("path:" + lp)
        base = lp.rsplit("/", 1)[-1]
        out += ["file:" + base, "file:" + base.rsplit(".", 1)[0]]
    cjk: list[str] = []

    def flush():
        if cjk:
            out.extend("".join(cjk[i:i + 2]) for i in range(max(1, len(cjk) - 1)))
            cjk.clear()
    for m in WORD_RE.finditer(text):
        w = m.group(0)
        if "㐀" <= w[0] <= "鿿":
            cjk.append(w)
            continue
        flush()
        if w[0].isdigit():
            if len(w) <= 12:
                out.append(w)
            continue
        parts = QUAL_RE.split(w)
        if len(parts) > 1:
            out.append(w.lower())
        for p in parts:
            lw = p.lower()
            if len(lw) < 2 or lw in STOP:
                continue
            if "_" in p or IDENT_RE.search(p):
                out.append(lw)
                sub = SPLIT_RE.split(CAMEL2.sub(r"\1 \2", CAMEL1.sub(r"\1 \2", p)))
                out += [_stem(s.lower()) for s in sub if len(s) > 1 and s.lower() not in STOP]
            else:
                out.append(_stem(lw))
    flush()
    return [sys.intern(t) for t in out] or ["__empty__"]


# ------------------------------------------------------------------------------------------------ helpers
def as_text(c) -> str:
    """str | [str | {"type":"text","text":..} | other] | None -> str. Never raises."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for p in c:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict) and p.get("type", "text") == "text" and p.get("text") is not None:
                out.append(str(p.get("text")))
        return "\n".join(out)
    if c is None:
        return ""
    if isinstance(c, dict):
        return as_text([c])
    return str(c)


def ntok(t: str) -> int:
    return len(ENC.encode(t, disallowed_special=()))


def clip(t: str, n: int) -> str:
    """First n tokens of t (encodes at most ~8n chars, so a multi-MB tool output costs nothing)."""
    head = t[: n * 8]
    ids = ENC.encode(head, disallowed_special=())
    if len(ids) <= n and len(head) == len(t):
        return t
    return ENC.decode(ids[:n]) + " ...[cut]"


SEG_CHARS = 2000


def segments(text: str) -> list[str]:
    """Line-preserving chunks of <= ~SEG_CHARS chars (diffs keep their newlines); a long line is hard-split."""
    if len(text) <= SEG_CHARS:
        return [text]
    out, cur, size = [], [], 0
    for line in text.split("\n"):
        if len(line) > SEG_CHARS:
            if cur:
                out.append("\n".join(cur)); cur, size = [], 0
            cut = len(line) - len(line) % SEG_CHARS
            out += [line[i:i + SEG_CHARS] for i in range(0, cut, SEG_CHARS)]
            line = line[cut:]
        if size + len(line) + 1 > SEG_CHARS and cur:
            out.append("\n".join(cur)); cur, size = [], 0
        cur.append(line); size += len(line) + 1
    if cur:
        out.append("\n".join(cur))
    return [s for s in out if s.strip()] or [text[:SEG_CHARS]]


EDIT_RE = re.compile(r"diff --git|^@@ |^\+\+\+ |^--- a/|old_str|new_str|old_string|new_string|str_replace|apply_patch|"
                     r"\[tool_use (?:Edit|MultiEdit|Write)\]|\*\*\* (?:Begin|Update|Add) (?:Patch|File)", re.M)
TESTCMD_RE = re.compile(r"\b(?:pytest|py\.test|unittest|tox|nox|npm (?:run )?test|yarn test|go test|cargo test|mvn test|"
                        r"gradle test|runtests\.py|make test)\b")
TESTRES_RE = re.compile(r"^(?:=+ [^\n]*\b\d+ (?:passed|failed|errors?)\b[^\n]*=+|Ran \d+ tests? in [^\n]*|OK(?: \([^\n]*\))?|"
                        r"FAILED \([^\n]*\))[ \t]*$", re.M)


def _ts_ms(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _iso(ts):
    if ts is None:
        return None
    try:
        return dt.datetime.fromtimestamp(ts / 1000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return None


# ------------------------------------------------------------------------------------- track detection
# All three tracks share /add and /search and send opaque ids. Ordinary Textual Adds are cut at
# 20 messages or 2,000 words (rules.txt:1143), so an Add with >20 messages, or >=2 messages and >2,000 words, is not an
# ordinary textual chunk; on top of that the Add must look like an agent trajectory (tool calls / diffs / tracebacks /
# shell commands / test summaries in a good share of its messages) and carry no image. Decided on a user's first Add.
TRAJ_RE = re.compile(r"\[tool_use|tool_use_id|\btool_result\b|<tool_call>|\"tool_calls\"|\bfunction_call\b|"
                     r"diff --git|^@@ -\d|^\+\+\+ b/|Traceback \(most recent call last\)|^[ \t]*File \"[^\"\n]+\", line \d+|"
                     r"\bold_str(?:ing)?\b|\bnew_str(?:ing)?\b|\bstr_replace\b|\bexit code\b|\breturncode\b|"
                     r"^[ \t]*\$ (?:cd|ls|cat|grep|rg|find|sed|python3?|pytest|git|pip|npm|make)\b|"
                     r"=+ [^\n]{0,200}?\b\d+ (?:passed|failed)\b|"
                     r"\"(?:file_path|old_string|new_string|command|pattern)\"\s*:|"
                     r"\b(?:Bash|Read|Edit|MultiEdit|Write|Grep|Glob|TodoWrite)\((?:command|file_path|pattern|path)?", re.M)


def trajectory_features(req: dict) -> dict:
    n = words = marked = 0
    image = False
    for m in req.get("messages") or []:
        c = m.get("content") if isinstance(m, dict) else m
        if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in c):
            image = True
        t = as_text(c)
        n += 1
        words += len(t.split())
        if TRAJ_RE.search(t[:200_000]):
            marked += 1
    return {"n": n, "words": words, "marked": marked, "image": image}


def is_trajectory_add(req: dict, min_frac: float = 0.2, min_marked: int = 5) -> bool:
    f = trajectory_features(req)
    structural = f["n"] >= 2 and (f["n"] > 20 or f["words"] > 2000)
    return structural and not f["image"] and f["marked"] >= max(min_marked, min_frac * f["n"])


# ------------------------------------------------------------------------------------------------- config
@dataclass
class CodeCfg:
    budget: int = 8000          # returned tokens over all cards
    max_cards: int = 5
    task_cut: int = 600         # head of the unit's first user message
    edit_cut: int = 800         # all edit/diff excerpts of one card (latest edits preferred)
    edit_piece: int = 400       # one edit excerpt
    hit_cut: int = 300          # best-matching segment, if not already shown
    final_cut: int = 400        # last assistant text without edit markers
    test_cut: int = 120         # last test-result line(s)
    w_key: float = 1.0          # RRF weight of the key-document channel
    w_seg: float = 1.0          # RRF weight of the segment channel (unit score = sum of its best 5 segments)
    rrf_k: int = 60
    rel_cut: float = 0.5        # >0: drop units whose key-doc BM25 < rel_cut * best (first card always kept)
    max_users: int = 128        # LRU: users kept in memory (messages + tokens + index); SQLite keeps all
    max_tokens: int = 60_000_000  # LRU: total BM25 tokens kept in memory
    prebuild_delay: float = 0.5   # >0: a background thread builds a user's index once its Adds pause this long


@dataclass
class AddRec:
    seq: int              # SQLite seq (defines unit order; equal before and after a restart)
    sid: str
    msgs: list            # (role, text, ts_ms)
    segs: list            # (msg index, segment text, BM25 tokens) -- tokenized once, at Add time
    ntok: int


def parse_add(req: dict) -> AddRec:
    msgs = []
    for m in req.get("messages") or []:
        if not isinstance(m, dict):
            m = {"content": m}
        t = as_text(m.get("content"))
        if t.strip():
            msgs.append((str(m.get("role") or "user"), t, _ts_ms(m.get("timestamp"))))
    segs = [(mi, s, code_tokens(f"{role}: {s}")) for mi, (role, text, _) in enumerate(msgs) for s in segments(text)]
    return AddRec(0, str(req.get("session_id") or "default"), msgs, segs, sum(len(x[2]) for x in segs))


def try_parse(req: dict) -> AddRec:
    """Content we cannot parse never fails an Add: the row stays in SQLite, an empty record keeps its place."""
    try:
        return parse_add(req)
    except Exception:
        return AddRec(0, str(req.get("session_id") or "default"), [], [], 0)


@dataclass
class Built:
    keys: list            # unit key (str), in unit order
    units: list           # per unit: list of (role, text, ts)
    segs: list            # per unit: list of (msg index, segment text)
    s_owner: np.ndarray   # segment -> unit idx
    s_pos: list           # segment -> (unit idx, msg idx, seg text)
    bm_s: object
    bm_k: object


@dataclass
class UserCache:
    complete: bool        # holds every Add of this user that is in the SQLite file
    adds: list = field(default_factory=list)
    version: int = 0
    built: Built | None = None
    built_version: int = -1
    tokens: int = 0


class CodeStore:
    """Coding-track store backed by an `adds(user_id, request_id, body, seq)` SQLite table (same schema as
    server.open_db). add() persists and tokenizes; search() builds BM25 from the cached tokens (cheap), or reloads
    the user from SQLite after a restart / eviction."""

    def __init__(self, db, db_lock: threading.Lock, cfg: CodeCfg | None = None):
        self.db, self.db_lock = db, db_lock
        self.cfg = cfg or CodeCfg()
        self.users: "OrderedDict[str, UserCache]" = OrderedDict()   # LRU order
        self.lock = threading.Lock()                                # guards users / user_locks / UserCache fields
        self.user_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._q: "queue.Queue[str]" = queue.Queue()
        self._queued: set[str] = set()                              # one queue entry per user, however many Adds
        self._last_add: dict[str, float] = {}
        if self.cfg.prebuild_delay > 0:
            threading.Thread(target=self._prebuild_loop, daemon=True, name="code-prebuild").start()

    def _prebuild_loop(self):
        while True:
            uid = self._q.get()
            try:
                time.sleep(max(0.0, self._last_add.get(uid, 0.0) + self.cfg.prebuild_delay - time.time()))
                with self.lock:
                    self._queued.discard(uid)
                    ul = self.user_locks[uid]
                    resident = uid in self.users
                if resident:      # an evicted user is reloaded by its own Search, or the LRU would thrash here
                    with ul:
                        self._get(uid)
            except Exception:
                pass          # a failed prebuild only means the next Search builds the index itself

    # ---------------------------------------------------------------------------------------------- add
    def add(self, req: dict) -> dict:
        uid, rid = str(req["user_id"]), req.get("request_id")
        key = None if not rid else rid if isinstance(rid, str) else json.dumps(rid)
        body = json.dumps(req)                     # ASCII: a lone surrogate in a tool output cannot reach SQLite
        rec = try_parse(req)                       # CPU work outside every lock
        with self.db_lock:
            dup = key is not None and self.db.execute(
                "SELECT 1 FROM adds WHERE user_id=? AND request_id=?", (uid, key)).fetchone() is not None
            if not dup:
                had = self.db.execute("SELECT 1 FROM adds WHERE user_id=? LIMIT 1", (uid,)).fetchone() is not None
                rec.seq = self.db.execute("INSERT INTO adds (user_id, request_id, body) VALUES (?,?,?)",
                                          (uid, key, body)).lastrowid
                self.db.commit()
        if not dup:
            with self.lock:
                uc = self.users.get(uid)
                if uc is None:
                    uc = self.users[uid] = UserCache(complete=not had)
                uc.adds.append(rec)
                uc.tokens += rec.ntok
                uc.version += 1
                self.users.move_to_end(uid)
                self._evict(keep=uid)
                self._last_add[uid] = time.time()
                wake = self.cfg.prebuild_delay > 0 and uid not in self._queued
                if wake:
                    self._queued.add(uid)
            if wake:
                self._q.put(uid)
        return {"success": True, "request_id": rid, "user_id": req["user_id"], "session_id": req.get("session_id")}

    def _evict(self, keep: str):
        """LRU over whole users (messages + tokens + index); caller holds self.lock. SQLite keeps everything."""
        total = sum(u.tokens for u in self.users.values())
        for uid in list(self.users):
            if len(self.users) <= self.cfg.max_users and total <= self.cfg.max_tokens:
                break
            if uid != keep:
                total -= self.users.pop(uid).tokens

    def _load(self, uid: str) -> UserCache:
        with self.db_lock:
            rows = self.db.execute("SELECT seq, body FROM adds WHERE user_id=? ORDER BY seq", (uid,)).fetchall()
        uc = UserCache(complete=True)
        for seq, body in rows:
            try:
                rec = try_parse(json.loads(body))
            except ValueError:
                continue
            rec.seq = seq
            uc.adds.append(rec)
            uc.tokens += rec.ntok
        return uc

    # -------------------------------------------------------------------------------------------- build
    def key_doc(self, unit, segs) -> str:
        first = next((t for r, t, _ in unit if r == "user"), unit[0][1])
        fi = self._final_index(unit)
        edits = [s for _, s in segs if EDIT_RE.search(s)]
        parts = [clip(first, 2000)] + ([clip(unit[fi][1], 800)] if fi is not None else []) + [clip(s, 300) for s in edits[:8]]
        return "\n".join(parts)

    @staticmethod
    def _final_index(unit):
        for i in range(len(unit) - 1, -1, -1):
            r, t, _ = unit[i]
            if r == "assistant" and not EDIT_RE.search(t) and not TESTCMD_RE.search(t[:400]):
                return i
        return None

    def _build(self, adds: list) -> Built:
        adds = sorted({a.seq: a for a in adds}.values(), key=lambda a: a.seq)   # an Add can be both read and appended
        # Unit = session_id; if a user has a single session_id but several Adds, each Add is one unit.
        one_sid = len({a.sid for a in adds}) == 1 and len(adds) > 1
        units: "OrderedDict[str, tuple]" = OrderedDict()
        for k, a in enumerate(adds):
            key = f"{a.sid}~{k}" if one_sid else a.sid
            msgs, segs = units.setdefault(key, ([], []))
            off = len(msgs)                                  # arrival order, not timestamp order
            msgs.extend(a.msgs)
            segs.extend((off + mi, s, t) for mi, s, t in a.segs)
        keys = [k for k, (m, _) in units.items() if m]
        owner, pos, toks, useg = [], [], [], []
        for ui, k in enumerate(keys):
            msgs, segs = units[k]
            useg.append([(mi, s) for mi, s, _ in segs])
            for mi, s, t in segs:
                owner.append(ui)
                pos.append((ui, mi, s))
                toks.append(t)
        bm_s, bm_k = bm25s.BM25(), bm25s.BM25()
        if toks:
            bm_s.index(toks, show_progress=False)
            bm_k.index([code_tokens(self.key_doc(units[k][0], u)) for k, u in zip(keys, useg)], show_progress=False)
        return Built(keys, [units[k][0] for k in keys], useg, np.asarray(owner, np.int64), pos, bm_s, bm_k)

    def _get(self, uid: str) -> Built | None:
        """Caller holds the user's lock (one build per user at a time)."""
        with self.lock:
            uc = self.users.get(uid)
            if uc is None:
                uc = self.users[uid] = UserCache(complete=False)
            self.users.move_to_end(uid)
        if not uc.complete:                                  # slow path: restart or evicted user
            fresh = self._load(uid)
            with self.lock:                                  # keep Adds that landed while we were reading
                have = {a.seq for a in fresh.adds}
                extra = [a for a in uc.adds if a.seq not in have]
                uc.adds = fresh.adds + extra
                uc.tokens = fresh.tokens + sum(a.ntok for a in extra)
                uc.complete = True
                uc.version += 1
                if not uc.adds:
                    self.users.pop(uid, None)
                    return None
                self._evict(keep=uid)
        with self.lock:
            if uc.built is not None and uc.built_version == uc.version:
                return uc.built
            version, adds = uc.version, list(uc.adds)
        b = self._build(adds)
        with self.lock:
            if uc.version == version:
                uc.built, uc.built_version = b, version
        return b

    # ------------------------------------------------------------------------------------------- search
    def rank(self, b: Built, query: str):
        c = self.cfg
        n = len(b.keys)
        qt = [t for t in code_tokens(query) if t != "__empty__"]
        if not qt or not n:
            return [], {}, np.zeros(n)
        ss = np.asarray(b.bm_s.get_scores(qt), np.float64)
        ks = np.asarray(b.bm_k.get_scores(qt), np.float64)
        agg = defaultdict(list)
        for i in np.nonzero(ss > 0)[0]:
            agg[int(b.s_owner[i])].append(ss[i])
        su = np.zeros(n)
        for u, v in agg.items():
            su[u] = sum(sorted(v)[-5:])
        fused = np.zeros(n)
        for w, s in ((c.w_seg, su), (c.w_key, ks)):
            r = np.empty(n, np.int64)
            r[np.argsort(-s, kind="stable")] = np.arange(n)
            fused += np.where(s > 0, w / (c.rrf_k + r + 1), 0.0)
        best = {}
        for i in np.argsort(-ss, kind="stable")[:400]:
            if ss[i] <= 0:
                break
            best.setdefault(int(b.s_owner[i]), int(i))
        order = [int(i) for i in np.argsort(-fused, kind="stable") if fused[i] > 0]
        if c.rel_cut > 0 and order and ks.max() > 0:
            top = ks.max()
            order = order[:1] + [u for u in order[1:] if ks[u] >= c.rel_cut * top]
        return order, best, ks

    def card(self, b: Built, u: int, best_seg: int | None, rank: int) -> str:
        c = self.cfg
        unit, segs = b.units[u], b.segs[u]
        first_i = next((i for i, (r, _, _) in enumerate(unit) if r == "user"), 0)
        when = (_iso(next((t for _, _, t in unit if t is not None), None)) or "")[:10] or "date unknown"
        out = [f"[Past task #{rank} | {when} | {len(unit)} messages | verbatim excerpts of an earlier agent trajectory; "
               f"its approach may or may not apply here]",
               f"{unit[first_i][0]}: " + clip(unit[first_i][1], c.task_cut)]
        whole = len(unit) == 1                 # one giant message: its later segments are all the card can add
        shown_msgs = set() if whole else {first_i}
        # latest edits first (earlier ones are often abandoned attempts), then shown in chronological order
        picked, left = [], c.edit_cut
        for mi, s in reversed(segs):
            if left <= 0:
                break
            if (whole or mi != first_i) and EDIT_RE.search(s):
                t = clip(s, min(left, c.edit_piece))
                left -= ntok(t)
                picked.append((mi, t))
        for mi, t in reversed(picked):
            out.append(f"{unit[mi][0]} (edit): {t}")
            shown_msgs.add(mi)
        if best_seg is not None:
            _, mi, s = b.s_pos[best_seg]
            if mi not in shown_msgs and not (whole and unit[mi][1].startswith(s)):
                out.append(f"{unit[mi][0]} (matching excerpt): {clip(s, c.hit_cut)}")
                shown_msgs.add(mi)
        for _, t, _ in reversed(unit):
            hits = TESTRES_RE.findall(t[-20000:])
            if hits:
                out.append("last test output seen in this trajectory: "
                           + clip(" | ".join(h.strip() for h in hits[-2:]), c.test_cut))
                break
        fi = self._final_index(unit)
        if fi is not None and fi not in shown_msgs:
            out.append("assistant (final): " + clip(unit[fi][1], c.final_cut))
        return "\n".join(out).encode("utf-8", "replace").decode("utf-8")      # lone surrogates cannot go out as UTF-8

    # -------------------------------------------------------- hand-over of a user that turned out not to be coding
    def rows(self, uid: str) -> list:
        with self.db_lock:
            return self.db.execute("SELECT request_id, body FROM adds WHERE user_id=? ORDER BY seq", (uid,)).fetchall()

    def forget(self, uid: str) -> None:
        with self.lock:
            ul = self.user_locks[uid]
        with ul:
            with self.db_lock:
                self.db.execute("DELETE FROM adds WHERE user_id=?", (uid,))
                self.db.commit()
            with self.lock:
                self.users.pop(uid, None)

    def search(self, req: dict) -> dict:
        uid = str(req["user_id"])
        try:
            top_k = 100 if req.get("top_k") is None else int(req["top_k"])
        except (TypeError, ValueError, OverflowError):
            top_k = 100
        q = as_text(req.get("query"))
        if top_k <= 0 or not q.strip():
            return {"data": []}
        with self.lock:
            ul = self.user_locks[uid]
        with ul:
            b = self._get(uid)
            if b is None or not b.keys:
                return {"data": []}
            order, best, _ = self.rank(b, q)
            order = order[:min(top_k, self.cfg.max_cards)]
            data, used, seen = [], 0, set()
            for u in order:
                text = self.card(b, u, best.get(u), len(data) + 1)
                t = ntok(text)
                if data and used + t > self.cfg.budget:
                    break
                used += t
                rid = f"{b.keys[u]}#card"
                if rid in seen:
                    rid = f"{rid}~{len(data)}"
                seen.add(rid)
                ts = next((x for _, _, x in b.units[u] if x is not None), None)
                data.append({"id": rid, "content": text, "score": round(1.0 / (len(data) + 1), 4),
                             "created_at": _iso(ts)})
        return {"data": data}
