"""Deterministic relative-time annotation: 'yesterday' -> 'yesterday [i.e. 21 February 2023]'.

Content stays verbatim; a bracketed resolution anchored on the message timestamp is inserted
after each relative expression, so the reader does not have to do calendar arithmetic.
"""
from __future__ import annotations

import datetime as dt
import re

WORDNUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "a couple of": 2, "a few": 3, "couple of": 2, "few": 3}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def d(t: dt.datetime) -> str:
    return f"{t.day} {t.strftime('%B %Y')}"


def m(t: dt.datetime) -> str:
    return t.strftime("%B %Y")


def add_months(t: dt.datetime, k: int) -> dt.datetime:
    y, mo = divmod(t.month - 1 + k, 12)
    return t.replace(year=t.year + y, month=mo + 1, day=1)


def resolve(expr: str, t: dt.datetime) -> str | None:
    e = expr.lower()
    if e in ("yesterday", "last night"):
        return d(t - dt.timedelta(days=1))
    if e in ("today", "tonight", "this morning", "this afternoon", "this evening"):
        return d(t)
    if e == "tomorrow":
        return d(t + dt.timedelta(days=1))
    if e in ("last week", "the past week", "previous week"):
        return f"the week before {d(t)}"
    if e == "next week":
        return f"the week after {d(t)}"
    if e in ("last weekend", "the past weekend"):
        return f"the weekend before {d(t)}"
    if e == "next weekend":
        return f"the weekend after {d(t)}"
    if e in ("last month", "the previous month"):
        return m(add_months(t, -1))
    if e == "next month":
        return m(add_months(t, 1))
    if e == "this month":
        return m(t)
    if e == "last year":
        return str(t.year - 1)
    if e == "next year":
        return str(t.year + 1)
    if e == "this year":
        return str(t.year)
    mm = re.fullmatch(r"(\d+|a couple of|a few|an?|one|two|three|four|five|six|seven|eight|nine|ten) (day|week|month|year)s? ago", e)
    if mm:
        n = int(mm.group(1)) if mm.group(1).isdigit() else WORDNUM[mm.group(1)]
        unit = mm.group(2)
        if unit == "day":
            return d(t - dt.timedelta(days=n))
        if unit == "week":
            return f"about {d(t - dt.timedelta(weeks=n))}"
        if unit == "month":
            return m(add_months(t, -n))
        return str(t.year - n)
    mm = re.fullmatch(r"(last|this past|next) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)", e)
    if mm:
        wd = WEEKDAYS.index(mm.group(2))
        if mm.group(1) == "next":
            delta = (wd - t.weekday()) % 7 or 7
            return d(t + dt.timedelta(days=delta))
        delta = (t.weekday() - wd) % 7 or 7
        return d(t - dt.timedelta(days=delta))
    return None


PATTERN = re.compile(
    r"\b(yesterday|last night|today|tonight|tomorrow|this (?:morning|afternoon|evening|month|year)|"
    r"(?:last|next|previous|the past) (?:week|weekend|month|year)|the previous month|"
    r"(?:\d+|a couple of|a few|an?|one|two|three|four|five|six|seven|eight|nine|ten) (?:day|week|month|year)s? ago|"
    r"(?:last|this past|next) (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.IGNORECASE)


# ---- v2 (2026-10-25): longer phrases first (v1 resolved "the day before yesterday" as yesterday), plus the
# frequent gaps listed in lit/phase2_temporal_aggregation.md 7.3. Ambiguous "on <weekday>" / "<Month> <day>" and
# event-anchored phrases are left alone.
NUMW = r"(\d+|a couple of|a few|couple of|few|an?|one|two|three|four|five|six|seven|eight|nine|ten)"
SEASONS = {"spring": (3, 5), "summer": (6, 8), "fall": (9, 11), "autumn": (9, 11), "winter": (12, 2)}


def num(w: str) -> int:
    return int(w) if w.isdigit() else WORDNUM[w]


def season(name: str, year: int) -> str:
    a, b = SEASONS[name]
    return f"winter {year}/{year + 1}" if name == "winter" else f"{name} {year}"


def resolve_v2(expr: str, t: dt.datetime) -> str | None:
    e = " ".join(expr.lower().split())
    if e == "the day before yesterday":
        return d(t - dt.timedelta(days=2))
    if e == "the day after tomorrow":
        return d(t + dt.timedelta(days=2))
    if e == "this past weekend":
        return f"the weekend before {d(t)}"
    if e == "this weekend":
        return f"the weekend of {d(t)}"
    if e == "this week":
        return f"the week of {d(t)}"
    if e in ("two weekends ago", "2 weekends ago"):
        return f"the weekend about two weeks before {d(t)}"
    mm = re.fullmatch(r"the (monday|tuesday|wednesday|thursday|friday|saturday|sunday) before last", e)
    if mm:
        delta = (t.weekday() - WEEKDAYS.index(mm.group(1))) % 7 or 7
        return d(t - dt.timedelta(days=delta + 7))
    mm = re.fullmatch(r"(?:in " + NUMW + r" (day|week|month|year)s?|" + NUMW + r" (day|week|month|year)s? from now)", e)
    if mm:
        n = num(mm.group(1) or mm.group(3))
        unit = mm.group(2) or mm.group(4)
        if unit == "day":
            return d(t + dt.timedelta(days=n))
        if unit == "week":
            return f"about {d(t + dt.timedelta(weeks=n))}"
        if unit == "month":
            return m(add_months(t, n))
        return str(t.year + n)
    mm = re.fullmatch(r"(?:the|in the) (?:past|last) " + NUMW + r" (day|week|month)s?", e)
    if mm:
        n, unit = num(mm.group(1)), mm.group(2)
        if unit == "month":
            return f"from about {d(t - dt.timedelta(days=30 * n))} to {d(t)}"
        start = t - (dt.timedelta(days=n) if unit == "day" else dt.timedelta(weeks=n))
        return f"from {d(start)} to {d(t)}"
    if e in ("a fortnight ago", "two weeks ago"):
        return f"about {d(t - dt.timedelta(weeks=2))}"
    mm = re.fullmatch(r"(last|this past|this) (spring|summer|fall|autumn|winter)", e)
    if mm:
        name = mm.group(2)
        a, b = SEASONS[name]
        if mm.group(1) == "this":     # only while inside that season; otherwise past vs coming depends on tense
            if name == "winter":
                return season(name, t.year - 1 if t.month <= 2 else t.year) if t.month in (12, 1, 2) else None
            return season(name, t.year) if a <= t.month <= b else None
        # last <season>: the most recent one that has already ended before the message date
        end_month = b
        y = t.year if t.month > end_month and name != "winter" else t.year - 1
        if name == "winter":
            y = t.year - 1 if t.month > 2 else t.year - 2
        return season(name, y)
    return resolve(expr, t)


PATTERN_V2 = re.compile(
    r"\b(the day before yesterday|the day after tomorrow|this past weekend|this weekend|this week|"
    r"two weekends ago|2 weekends ago|"
    r"the (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday) before last|"
    r"in " + NUMW + r" (?:day|week|month|year)s?|" + NUMW + r" (?:day|week|month|year)s? from now|"
    r"(?:the|in the) (?:past|last) " + NUMW + r" (?:day|week|month)s?|a fortnight ago|"
    r"(?:last|this past|this) (?:spring|summer|fall|autumn|winter)|"
    r"yesterday|last night|today|tonight|tomorrow|this (?:morning|afternoon|evening|month|year)|"
    r"(?:last|next|previous|the past) (?:week|weekend|month|year)|the previous month|"
    r"(?:\d+|a couple of|a few|an?|one|two|three|four|five|six|seven|eight|nine|ten) (?:day|week|month|year)s? ago|"
    r"(?:last|this past|next) (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.IGNORECASE)


def annotate(text: str, ts_ms: int | None, version: int = 1) -> str:
    if ts_ms is None:
        return text
    t = dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc)
    pattern, res = (PATTERN_V2, resolve_v2) if version >= 2 else (PATTERN, resolve)

    def sub(mo):
        r = res(mo.group(0), t)
        return f"{mo.group(0)} [i.e. {r}]" if r else mo.group(0)
    return pattern.sub(sub, text)
