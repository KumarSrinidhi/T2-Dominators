"""Weekly time-window schedules that arm rules only at certain times (e.g. after hours).

    "schedule": {"tz": "Asia/Kolkata", "windows": ["mon-fri 19:00-07:00", "sat,sun 00:00-24:00"],
                 "label": "after-hours"}

A window is `<days> <HH:MM>-<HH:MM>`. Days: `mon`..`sun`, comma lists (`sat,sun`), ranges (`mon-fri`, wrapping
`fri-mon`), or `daily` (the day part may be omitted, meaning daily). The end may be `24:00`. A window whose end is
earlier than its start is OVERNIGHT: it starts on each listed day and runs into the next day, so
"mon-fri 19:00-07:00" is active Mon 19:00 -> Tue 07:00, ..., Fri 19:00 -> Sat 07:00 (not Mon 00:00-07:00).
Start and end are inclusive/exclusive: 19:00 is inside, 07:00 is outside.

`tz` is an IANA zone name, `UTC`, or a fixed offset (`+05:30`, `UTC-03:00`); omitted = the server's local time.
Windows machines without the `tzdata` package cannot resolve IANA names, so a few zones that have no daylight
saving time (Asia/Kolkata, ...) fall back to their fixed offset; any other unknown zone is a configuration error.
The schedule is evaluated on the frame timestamp (wall-clock epoch seconds for live and real-time file sources).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Optional

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
DAY_ALIASES = {"daily": tuple(range(7)), "everyday": tuple(range(7)), "all": tuple(range(7)), "*": tuple(range(7)),
               "weekdays": (0, 1, 2, 3, 4), "weekends": (5, 6), "weekend": (5, 6)}
# zones without daylight-saving time, used only when the IANA database is unavailable (Windows without tzdata)
FIXED_OFFSETS = {"Asia/Kolkata": 330, "Asia/Calcutta": 330, "Asia/Kathmandu": 345, "Asia/Dhaka": 360,
                 "Asia/Karachi": 300, "Asia/Dubai": 240, "Asia/Singapore": 480, "Asia/Shanghai": 480,
                 "Asia/Hong_Kong": 480, "Asia/Tokyo": 540, "Asia/Seoul": 540, "Asia/Jakarta": 420,
                 "Asia/Bangkok": 420, "Asia/Riyadh": 180, "Europe/Moscow": 180, "Africa/Nairobi": 180,
                 "Africa/Lagos": 60, "Etc/UTC": 0, "Etc/GMT": 0, "GMT": 0}
MAX_WINDOWS = 16
_TIME = r"(\d{1,2}):(\d{2})"
_WIN_RE = re.compile(rf"^(?:(?P<days>[a-z*,\-\s]+?)\s+)?{_TIME}\s*-\s*{_TIME}$")
_OFF_RE = re.compile(r"^(?:utc|gmt)?\s*([+-])(\d{1,2})(?::?(\d{2}))?$")


def _day(tok: str, what: str) -> int:
    t = tok.strip()[:3]
    if t not in DAYS:
        raise ValueError(f"{what}: unknown day '{tok.strip()}' (use mon, tue, wed, thu, fri, sat, sun)")
    return DAYS.index(t)


def _days(spec: str, what: str) -> frozenset[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"{what}: empty day in '{spec}'")
        if part in DAY_ALIASES:
            out.update(DAY_ALIASES[part])
        elif "-" in part:
            a, b = (_day(x, what) for x in part.split("-", 1))
            i = a
            while True:  # ranges wrap around the week: fri-mon = fri, sat, sun, mon
                out.add(i)
                if i == b:
                    break
                i = (i + 1) % 7
        else:
            out.add(_day(part, what))
    return frozenset(out)


def _minutes(h: str, m: str, what: str, end: bool) -> int:
    hh, mm = int(h), int(m)
    if mm > 59 or hh > 24 or (hh == 24 and (mm or not end)):
        raise ValueError(f"{what}: invalid time {h}:{m} (00:00-23:59; 24:00 only as an end time)")
    return hh * 60 + mm


@dataclass(frozen=True)
class Window:
    text: str
    days: frozenset[int]
    start: int  # minutes after local midnight
    end: int    # minutes after local midnight, 1..1440; end <= start = overnight into the next day

    @classmethod
    def parse(cls, text: Any, what: str = "schedule window") -> "Window":
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{what} must be a string like 'mon-fri 19:00-07:00'")
        s = " ".join(text.strip().lower().split())
        m = _WIN_RE.match(s)
        if not m:
            raise ValueError(f"{what}: cannot parse '{text}' (expected '<days> HH:MM-HH:MM', "
                             "e.g. 'mon-fri 19:00-07:00' or 'sat,sun 00:00-24:00')")
        days = _days(m.group("days").replace(" ", ""), what) if m.group("days") else frozenset(range(7))
        start = _minutes(m.group(2), m.group(3), what, end=False)
        end = _minutes(m.group(4), m.group(5), what, end=True)
        if start == end or (start == 0 and end == 0):
            raise ValueError(f"{what}: '{text}' has the same start and end (use 00:00-24:00 for a whole day)")
        return cls(" ".join(text.split()), days, start, end)

    @property
    def overnight(self) -> bool:
        return self.end <= self.start

    def contains(self, weekday: int, minute: int) -> bool:
        if not self.overnight:
            return weekday in self.days and self.start <= minute < self.end
        # overnight: the evening part on a listed day, the early-morning part on the day after a listed day
        return (weekday in self.days and minute >= self.start) or ((weekday - 1) % 7 in self.days and minute < self.end)


def parse_tz(name: Any, what: str = "schedule") -> tuple[Optional[tzinfo], str]:
    """(tzinfo or None for server-local time, how it was resolved)."""
    if name in (None, "", "local"):
        return None, "server local time"
    if not isinstance(name, str):
        raise ValueError(f"{what}: tz must be a string such as 'Asia/Kolkata', 'UTC' or '+05:30'")
    s = name.strip()
    if s.upper() in ("UTC", "Z", "GMT"):
        return timezone.utc, "UTC"
    m = _OFF_RE.match(s.lower())
    if m:
        sign, hh, mm = m.group(1), int(m.group(2)), int(m.group(3) or 0)
        if hh > 14 or mm > 59:
            raise ValueError(f"{what}: invalid UTC offset '{name}'")
        off = (hh * 60 + mm) * (1 if sign == "+" else -1)
        return timezone(timedelta(minutes=off)), f"fixed offset {name}"
    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
        try:
            return ZoneInfo(s), "IANA time zone"
        except (ZoneInfoNotFoundError, ValueError):
            pass
    except ImportError:  # pragma: no cover - zoneinfo is stdlib on 3.9+
        pass
    if s in FIXED_OFFSETS:
        off = FIXED_OFFSETS[s]
        return timezone(timedelta(minutes=off), s), (f"fixed offset UTC{'+' if off >= 0 else '-'}{abs(off) // 60:02d}:"
                                                     f"{abs(off) % 60:02d} (no DST; IANA database not installed)")
    raise ValueError(f"{what}: unknown time zone '{name}' (use an IANA name such as 'Asia/Kolkata', 'UTC' or a fixed "
                     "offset such as '+05:30'; on Windows install the 'tzdata' package for other IANA names)")


class Schedule:
    """A set of weekly windows in one time zone. `active_window(t)` is the first window containing epoch time t."""

    def __init__(self, windows: list[Window], tz: Optional[tzinfo], tz_name: str, tz_note: str, label: str):
        self.windows, self.tz, self.tz_name, self.tz_note, self.label = windows, tz, tz_name, tz_note, label

    @classmethod
    def parse(cls, d: Any, what: str = "schedule") -> Optional["Schedule"]:
        """Parse a schedule object (or a bare window string / list of window strings). None/{} -> None.
        Raises ValueError with a message naming `what` on any problem."""
        if d is None or d == {} or d == [] or d == "":
            return None
        if isinstance(d, (str, list)):
            d = {"windows": [d] if isinstance(d, str) else d}
        if not isinstance(d, dict):
            raise ValueError(f"{what} must be an object like "
                             '{"tz": "Asia/Kolkata", "windows": ["mon-fri 19:00-07:00"]}')
        bad = set(d) - {"tz", "windows", "label"}
        if bad:
            raise ValueError(f"{what}: unknown key(s) {sorted(bad)} (allowed: tz, windows, label)")
        wins = d.get("windows")
        if isinstance(wins, str):
            wins = [w for w in (x.strip() for x in wins.split(";")) if w]
        if not isinstance(wins, list) or not wins:
            raise ValueError(f"{what}: windows must be a non-empty list such as [\"mon-fri 19:00-07:00\"]")
        if len(wins) > MAX_WINDOWS:
            raise ValueError(f"{what}: at most {MAX_WINDOWS} windows")
        windows = [Window.parse(w, f"{what} window {i + 1}") for i, w in enumerate(wins)]
        tz, note = parse_tz(d.get("tz"), what)
        label = d.get("label") or "active"
        if not isinstance(label, str) or len(label) > 40:
            raise ValueError(f"{what}: label must be a short string (<= 40 characters)")
        return cls(windows, tz, str(d.get("tz") or "local"), note, label.strip() or "active")

    def local(self, t: float) -> datetime:
        return datetime.fromtimestamp(t, self.tz) if self.tz is not None else datetime.fromtimestamp(t)

    def active_window(self, t: float) -> Optional[Window]:
        lt = self.local(t)
        wd, minute = lt.weekday(), lt.hour * 60 + lt.minute
        return next((w for w in self.windows if w.contains(wd, minute)), None)

    def active(self, t: float) -> bool:
        return self.active_window(t) is not None

    def describe(self, t: float, source: str = "zone") -> Optional[dict]:
        """Derivation record for an event at time t (None when no window is active)."""
        w = self.active_window(t)
        if w is None:
            return None
        return {"text": f"{self.label} window {w.text}", "window": w.text, "label": self.label, "tz": self.tz_name,
                "tz_resolution": self.tz_note, "local_time": self.local(t).strftime("%a %H:%M"), "source": source}

    def to_dict(self) -> dict:
        return {"tz": self.tz_name, "windows": [w.text for w in self.windows], "label": self.label}

    def status(self, t: float) -> dict:
        w = self.active_window(t)
        return {**self.to_dict(), "active": w is not None, "window": w.text if w else None,
                "local_time": self.local(t).strftime("%a %H:%M"), "tz_resolution": self.tz_note}
