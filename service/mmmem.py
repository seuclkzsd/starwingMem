"""Multimodal path (v1.1). Only users whose Adds carried an image_url part come here.

Textual and Coding contracts allow text parts only (content_types == ["text"]), so those users never reach this
module and keep the exact v1.0.3 code path. For image users:
  * Add decodes each base64 image once, stores it content-addressed on disk (<blobs>/<sha[:2]>/<sha>) and persists the
    request with every image_url part replaced by an image_ref {sha256, mime, bytes, w, h}: no base64 in SQLite.
  * Indexed text = the text parts in order, each image replaced in place by "[Image <sha8>]", so the platform caption
    and the dialogue around the image are what retrieval sees (text-embedding-v4 + BM25 only, no generative model).
  * Search returns ContentPart[]: the verbatim text windows first (rank order, same packer as v1.0), then the images of
    those windows as separate items (rank order) under count / image-token / byte caps. If Answer truncates the prefix,
    images go before any text.
  * Users load lazily from SQLite on Search and sit in an LRU bounded by passages: nothing is replayed at start-up.
  * BM25 sees CJK runs as character bigrams -- only inside calls made from this module (thread-local switch).
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import re
import struct
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import memsys
from memsys import MemoryStore, Message, UserMem, feedback_view, iso_time, ntok, passages, qtype

MAX_IMG = 10 * 1024 * 1024                 # rules.txt:1455,1583 (decoded, per image)
MEDIA = ("image/jpeg", "image/png", "image/webp")    # rules.txt:1455; anything else gets a 4xx (rules.txt:1607)

# ----------------------------------------------------------------------------------- BM25 tokenizer switch
_tls = threading.local()
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]+")


def cjk_bigrams(text: str) -> str:
    """'订单页面' -> ' 订单 单页 页面 ' (bm25s keeps tokens of >= 2 word chars, so bigrams survive; latin text unchanged)."""
    def rep(m):
        s = m.group(0)
        return " " + " ".join(s[i:i + 2] for i in range(max(1, len(s) - 1))) + " "
    return CJK.sub(rep, text)


class _Bm25Proxy:
    """Stands in for memsys.bm25s. With the switch off (always, outside this module) it forwards the very same call."""
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def tokenize(self, texts, *a, **kw):
        if getattr(_tls, "cjk", False):
            texts = [cjk_bigrams(t) for t in texts] if isinstance(texts, list) else cjk_bigrams(texts)
        return self._real.tokenize(texts, *a, **kw)


if not isinstance(memsys.bm25s, _Bm25Proxy):
    memsys.bm25s = _Bm25Proxy(memsys.bm25s)


@contextmanager
def cjk_bm25(on: bool = True):
    old = getattr(_tls, "cjk", False)
    _tls.cjk = on
    try:
        yield
    finally:
        _tls.cjk = old


# ---------------------------------------------------------------------------------------------- images
class BadPayload(ValueError):
    pass


DATA_URI = re.compile(r"data:(image/[A-Za-z0-9.+-]+);base64,", re.I)


def sniff_mime(b: bytes) -> str | None:
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if b[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return None


def img_size(b: bytes) -> tuple[int, int]:
    """(width, height) from the PNG / JPEG / WebP / GIF header; (0, 0) when unknown. No Pillow on the server."""
    try:
        if b[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", b[16:24])
        if b[:6] in (b"GIF87a", b"GIF89a"):
            return struct.unpack("<HH", b[6:10])
        if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
            c = b[12:16]
            if c == b"VP8X":
                return 1 + int.from_bytes(b[24:27], "little"), 1 + int.from_bytes(b[27:30], "little")
            if c == b"VP8L":
                v = int.from_bytes(b[21:25], "little")
                return 1 + (v & 0x3FFF), 1 + ((v >> 14) & 0x3FFF)
            if c == b"VP8 ":
                w, h = struct.unpack("<HH", b[26:30])
                return w & 0x3FFF, h & 0x3FFF
        if b[:2] == b"\xff\xd8":
            i = 2
            while i + 9 < len(b):
                if b[i] != 0xFF:
                    i += 1
                    continue
                mk = b[i + 1]
                if mk in (0xD8, 0x01, 0xFF) or 0xD0 <= mk <= 0xD7:
                    i += 1 if mk == 0xFF else 2
                    continue
                ln = struct.unpack(">H", b[i + 2:i + 4])[0]
                if mk in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h, w = struct.unpack(">HH", b[i + 5:i + 9])
                    return w, h
                i += 2 + ln
    except (struct.error, IndexError):
        pass
    return 0, 0


def img_tokens(w: int, h: int) -> int:
    """Upper bound of what one image costs a gpt-4o-mini reader (OpenAI vision guide: 2833 + 5667 per 512-px tile at
    high detail; fit 2048 square, shortest side 768, smaller images are not enlarged). Unknown size -> 4 tiles."""
    if w <= 0 or h <= 0:
        return 2833 + 5667 * 4
    s = min(1.0, 2048 / max(w, h))
    w, h = w * s, h * s
    s = min(1.0, 768 / min(w, h))
    w, h = w * s, h * s
    return 2833 + 5667 * math.ceil(w / 512) * math.ceil(h / 512)


def b64_len(nbytes: int) -> int:
    """Length of the data URI returned for an image of nbytes (base64 + a generous prefix)."""
    return 4 * math.ceil(nbytes / 3) + 32


class Blobs:
    """Content-addressed image files; one copy per distinct image whatever the number of users/questions using it."""
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, sha: str) -> Path:
        return self.root / sha[:2] / sha

    def put(self, raw: bytes) -> str:
        sha = hashlib.sha256(raw).hexdigest()
        p = self.path(sha)
        if not p.exists():
            p.parent.mkdir(exist_ok=True)
            tmp = p.with_name(f".{sha}.{os.getpid()}.{threading.get_ident()}")
            with open(tmp, "wb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
        return sha

    def data_uri(self, sha: str, mime: str) -> str:
        return f"data:{mime};base64," + base64.b64encode(self.path(sha).read_bytes()).decode()


def _has_image_part(msg) -> bool:
    c = msg.get("content") if isinstance(msg, dict) else None
    return isinstance(c, list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in c)


def has_image(req: dict) -> bool:
    return any(_has_image_part(m) for m in req.get("messages") or [])


def _decode(url) -> tuple[bytes, str]:
    m = DATA_URI.match(url) if isinstance(url, str) else None
    if not m:
        raise BadPayload("image_url.url must be an inline base64 data URI")
    try:
        raw = base64.b64decode(url[m.end():])
    except (binascii.Error, ValueError):
        raise BadPayload("image_url.url is not valid base64")
    if not raw:
        raise BadPayload("empty image")
    return raw, m.group(1).lower()


def strip_message(msg: dict, blobs: Blobs) -> dict:
    """Message as persisted: image_url parts -> image_ref (image bytes go to blobs). Raises BadPayload."""
    if not isinstance(msg, dict):
        raise BadPayload("each message must be an object")
    c = msg.get("content", "")
    if c is None or isinstance(c, str):
        return msg
    if not isinstance(c, list):
        raise BadPayload("content must be a string or a ContentPart[] array")
    kept = []
    for p in c:
        if not isinstance(p, dict):
            raise BadPayload("each content part must be an object")
        t = p.get("type")
        if t == "text":
            if not isinstance(p.get("text"), str):
                raise BadPayload("a text part needs a string text")
            kept.append({"type": "text", "text": p["text"]})
        elif t == "image_url":
            iu = p.get("image_url")
            raw, _ = _decode(iu.get("url") if isinstance(iu, dict) else iu)
            mime = sniff_mime(raw)                     # the bytes decide, not the declared type
            if mime not in MEDIA or len(raw) > MAX_IMG:
                raise BadPayload("images must be JPEG, PNG or WebP, at most 10 MiB decoded")
            w, h = img_size(raw)
            kept.append({"type": "image_ref", "sha256": blobs.put(raw), "mime": mime,
                         "bytes": len(raw), "w": int(w), "h": int(h)})
        # other part types are outside the contract (text | image_url) and ignored
    return {**msg, "content": kept}


def msg_text(msg: dict) -> tuple[str, list[dict]]:
    """Indexed / returned text of a persisted message and its image refs in order."""
    c = msg.get("content", "")
    if c is None:
        return "", []
    if isinstance(c, str):
        return c, []
    texts, refs = [], []
    for p in c:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text" and isinstance(p.get("text"), str):
            texts.append(p["text"])
        elif p.get("type") == "image_ref":
            refs.append(p)
            texts.append(f"[Image {p['sha256'][:8]}]")
    return "\n".join(t for t in texts if t), refs


def query_parts(q) -> tuple[str, list[str]]:
    """Query text (text parts joined like memsys does) and sha256 of the query images (never stored)."""
    if isinstance(q, str):
        return q, []
    if not isinstance(q, list) or any(not isinstance(p, dict) for p in q):
        raise BadPayload("query must be a string or a ContentPart[] array")
    text = " ".join(p.get("text", "") for p in q if p.get("type") == "text" and isinstance(p.get("text"), str))
    shas = []
    for p in q:
        if p.get("type") == "image_url":
            iu = p.get("image_url")
            try:
                raw, _ = _decode(iu.get("url") if isinstance(iu, dict) else iu)
                shas.append(hashlib.sha256(raw).hexdigest())
            except BadPayload:
                pass                      # a bad query image only loses the exact-match channel
    return text, shas


# ---------------------------------------------------------------------------------------------- store
@dataclass
class MMUser(UserMem):
    imgs: dict = field(default_factory=dict)         # msg idx -> [image_ref, ...] in content order
    img_owner: dict = field(default_factory=dict)    # sha256 -> [msg idx]
    last_seq: int = 0                                # SQLite adds.seq loaded so far
    active: int = 0                                  # Searches in flight (never evicted while > 0)


# answer_tok: what text + images may cost a gpt-4o-mini reader in total. 117,760 input tokens (rules.txt:1609) minus
# ~9.8k for the question, options, instructions and image labels; images only take what the returned text leaves.
# Verbatim text in a label, so the reader takes it as a record of past conversations and not as a chat to continue.
# gpt-4o-mini, local: MemLens 6 -> 9 of 36 correct with its own prompt, 0 -> 6 with the MobileMem one; MobileMem 8 -> 11 of 80.
WINDOW = '<past_conversation index="{i}">\n{t}\n</past_conversation>'
MM_DEFAULTS = {"max_passages": 100_000, "img_max": 12, "answer_tok": 108_000, "img_tok_budget": 10 ** 9,
               "img_byte_budget": 24 * 1024 * 1024, "cjk_bm25": True}


class MMStore(MemoryStore):
    def __init__(self, cfg, db, db_lock, blobs: Blobs, opts: dict | None = None):
        super().__init__(cfg)
        self.users = {}                    # uid -> MMUser, only the loaded ones (LRU order in self.lru)
        self.lru: OrderedDict = OrderedDict()
        self.lru_lock = threading.Lock()
        self.db, self.db_lock, self.blobs = db, db_lock, blobs
        self.o = {**MM_DEFAULTS, **(opts or {})}

    def indexed(self, msg: dict) -> tuple[str | None, list[dict]]:
        """Text the index holds for a persisted message (None: not indexed, as memsys does for stream feedback)."""
        text, refs = msg_text(msg)
        return feedback_view(text, self.cfg.stream_fb), refs

    def doc_texts(self, msgs: list[dict]) -> list[str]:
        """Exactly what UserMem.rebuild will embed for these persisted messages, so Search hits the embedding cache."""
        out = []
        for m in msgs:
            text, _ = self.indexed(m)
            if text is not None:
                out += [f"{m.get('role', 'user')}: {p}" for p in passages(text)]
        return out

    # -------------------------------------------------------------- lazy load from SQLite (Add never touches memory)
    def _load(self, uid: str) -> MMUser:
        """The resident copy of uid brought up to date with SQLite, pinned against eviction until _unpin."""
        with self.lru_lock:
            u = self.users.get(uid) or MMUser()
            u.active += 1
            self.users[uid] = u
            self.lru[uid] = True
            self.lru.move_to_end(uid)
        try:
            with self.db_lock:
                rows = self.db.execute("SELECT seq, body FROM adds WHERE user_id=? AND seq>? ORDER BY seq",
                                       (uid, u.last_seq)).fetchall()
            for seq, body in rows:
                req = json.loads(body)
                rid = req.get("request_id")
                u.last_seq = seq
                if rid and rid in u.seen_requests:
                    continue
                for msg in req.get("messages") or []:
                    if not isinstance(msg, dict):
                        continue
                    if _has_image_part(msg):                   # a pre-v1.1 row that still carries base64
                        try:
                            msg = strip_message(msg, self.blobs)
                        except BadPayload:
                            pass                               # its images are dropped, the text parts kept
                    text, refs = self.indexed(msg)
                    if text is None:
                        continue
                    m = Message(len(u.msgs), req.get("session_id") or "default", msg.get("timestamp"),
                                msg.get("role", "user"), text, ntok(text))
                    if refs:
                        u.imgs[m.idx] = refs
                        for r in refs:
                            u.img_owner.setdefault(r["sha256"], []).append(m.idx)
                    u.msgs.append(m)
                if rid:
                    u.seen_requests.add(rid)
                u.dirty = True
        except BaseException:
            self._unpin(u)
            raise
        return u

    def _unpin(self, u: MMUser) -> None:
        with self.lru_lock:
            u.active -= 1

    def _evict(self) -> None:
        """Drop least recently searched users (memory only; SQLite and the embedding cache keep everything)."""
        with self.lru_lock:
            size = {k: max(len(u.p_owner), len(u.msgs)) for k, u in self.users.items()}
            total = sum(size.values())
            for k in list(self.lru):
                if total <= self.o["max_passages"]:
                    break
                if self.users[k].active:
                    continue
                total -= size[k]
                del self.users[k], self.lru[k]

    def resident(self) -> tuple[int, int]:
        with self.lru_lock:
            return len(self.users), sum(len(u.p_owner) for u in self.users.values())

    # -------------------------------------------------------------- search
    def rank_messages(self, user, query, options, trange=None, now=None, session_cap=None):
        order = super().rank_messages(user, query, options, trange, now, session_cap)
        shas = getattr(_tls, "qimgs", None)
        if shas and isinstance(user, MMUser):            # the query image is one the user sent: its turns first
            hit = list(dict.fromkeys(i for s in shas for i in user.img_owner.get(s, [])))
            if hit:
                hs = set(hit)
                order = hit + [i for i in order if i not in hs]
        _tls.order = order
        return order

    def search_parts(self, req: dict) -> list[dict]:
        """Caller holds the per-user lock (server.user_locks). Returns data[] with ContentPart[] contents."""
        uid = req["user_id"]
        qtext, qshas = query_parts(req["query"])
        opts = req.get("options")
        opts = [o for o in opts if isinstance(o, str)] if isinstance(opts, list) else None
        u = self._load(uid)
        try:
            if not u.msgs:
                return []
            over = self.cfg.route.get(qtype(qtext)) if self.cfg.route else None
            view = object.__new__(MMStore)               # route resolved here so the view keeps this class; its own
            view.__dict__ = {**self.__dict__, "users": {uid: u},     # users map, so a concurrent eviction is harmless
                             "cfg": replace(self.cfg, route={}, **(over or {}))}
            _tls.qimgs, _tls.order = qshas, []
            with cjk_bm25(self.o["cjk_bm25"]):
                res = MemoryStore.search(view, {**req, "query": qtext, "options": opts})
            order = _tls.order
        finally:
            _tls.qimgs = None
            self._unpin(u)
        top_k = int(req.get("top_k") or 100)
        out = [{"id": d["id"], "content": [{"type": "text", "text": WINDOW.format(i=k, t=d["content"])}], "score": d["score"],
                "created_at": d["created_at"]} for k, d in enumerate(res["data"], 1)]
        text_tok = sum(ntok(d["content"][0]["text"]) + 8 for d in out)
        out += self._images(u, res["data"], order, max(0, top_k - len(out)), out[-1]["score"] if out else 1.0,
                            min(self.o["img_tok_budget"], self.o["answer_tok"] - text_tok))
        self._evict()
        return out

    def _images(self, u: MMUser, data: list[dict], order: list, room: int, last: float, tok_budget: int) -> list[dict]:
        """Images of the returned windows, best-ranked turn first, then window order; capped by count, gpt-4o-mini
        image tokens and base64 bytes. Separate trailing items, so a prefix-truncating Answer drops them first."""
        o = self.o
        if not room or not o["img_max"] or tok_budget <= 0:
            return []
        in_ctx = [i for d in data for i in d.get("msg_ids") or []]
        ctx = set(in_ctx)
        cand = [int(i) for i in order[:400] if int(i) in ctx and int(i) in u.imgs] + [i for i in in_ctx if i in u.imgs]
        cap = min(o["img_max"], room)
        picked, seen, tok, nbytes = [], set(), 0, 0
        for i in cand:
            for r in u.imgs[i]:
                s = r["sha256"]
                if s in seen:
                    continue
                seen.add(s)
                b64 = b64_len(r["bytes"])
                t = img_tokens(r.get("w", 0), r.get("h", 0))
                if tok + t > tok_budget or nbytes + b64 > o["img_byte_budget"]:
                    continue
                picked.append((i, r))
                tok, nbytes = tok + t, nbytes + b64
                if len(picked) >= cap:
                    break
            if len(picked) >= cap:
                break
        items = []
        for i, r in picked:
            m = u.msgs[i]
            try:
                uri = self.blobs.data_uri(r["sha256"], r["mime"])
            except FileNotFoundError:
                continue
            label = f"[Image {r['sha256'][:8]}] sent by the {m.role} on {self.date_str(m.ts)}"
            items.append({"id": f"img:{r['sha256'][:24]}",
                          "content": [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": uri}}],
                          "score": last * 0.5 ** (len(items) + 1), "created_at": iso_time(m.ts)})
        return items
