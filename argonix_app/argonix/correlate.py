"""Event correlation and prioritisation.

Events on the same site zone (possibly from different cameras) or the same track within a time window are
grouped into one alert. Multiple distinct rules/cameras corroborate and raise the score, and so does agreement across
modalities (an access-control door event plus a video rule in the same site zone). Cross-camera grouping is spatial
corroboration only; it makes no identity claim.
"""
from __future__ import annotations

import itertools
import time
from typing import Optional

from .models import Alert, Event, is_demo
from .rules import LABEL

PRIORITY_BANDS = [(6.0, "P1"), (3.0, "P2"), (1.5, "P3"), (0.0, "P4")]
DEMO_PREFIX = "[DEMO] "  # alerts scripted by a demo scenario (argonix/scenario.py); never grouped with observed ones
ORDER = ["P1", "P2", "P3", "P4"]
LEARNED_ONLY_MAX = "P3"  # AGENTS.md: learned-model outputs are corroboration only until a deterministic rule agrees


def priority_for(score: float) -> str:
    return next(p for s, p in PRIORITY_BANDS if score >= s)


MODALITY_LABEL = {"video": "video", "access_control": "sensor"}


def is_learned(ev: Event) -> bool:
    return bool(ev.derivation.get("learned_model"))


def modality(ev: Event) -> str:
    """Where an event came from: "video" (camera analytics, the default) or a non-video source such as
    "access_control" (door controller / alarm panel, argonix.sensors)."""
    return ev.derivation.get("modality") or "video"


def explain_priority(events: list[Event]) -> tuple[float, str, dict]:
    """Score, priority and a factor breakdown the operator can read ("why was I shown this")."""
    top = max(events, key=lambda ev: ev.score())
    distinct_rules = len({ev.rule for ev in events})
    # a non-video event's camera_id names its source (e.g. "sensor:door-ctrl-1"), not a camera
    distinct_cams = len({ev.camera_id for ev in events if modality(ev) == "video"})
    modalities = sorted({modality(ev) for ev in events})
    corroboration = 1 + 0.25 * (distinct_rules + max(1, distinct_cams) - 2) + 0.25 * (len(modalities) - 1)
    score = round(top.score() * corroboration, 3)
    prio = priority_for(score)
    learned_only = all(is_learned(ev) for ev in events)
    capped = learned_only and ORDER.index(prio) < ORDER.index(LEARNED_ONLY_MAX)
    if capped:
        prio = LEARNED_ONLY_MAX
    factors = {
        "top_event": top.rule, "severity": top.severity, "zone_criticality": top.criticality,
        "confidence": top.confidence, "base_score": round(top.score(), 3),
        "distinct_rules": distinct_rules, "distinct_cameras": distinct_cams,
        "modalities": modalities,
        "cross_modal": (" + ".join(sorted({MODALITY_LABEL.get(m, m) for m in modalities}))
                        if len(modalities) > 1 else None),
        "corroboration_multiplier": round(corroboration, 3), "score": score,
        "bands": {p: s for s, p in PRIORITY_BANDS},
        "learned_model_only": learned_only,
        "cap": (f"capped at {LEARNED_ONLY_MAX}: only learned-model inferences, no deterministic rule agrees yet"
                if capped else None),
    }
    return score, prio, factors


class Correlator:
    def __init__(self, window_s: float = 15.0, model_info: Optional[dict] = None):
        self.window = window_s
        self.model_info = model_info or {}
        self.open: list[Alert] = []
        self._ids = itertools.count(1)

    def _key(self, e: Event) -> str:
        if e.zone_id == "_view" and not e.site_zone:
            # whole-view events (camera health, weapon-like object, altercation, fall, anomaly) are not about one
            # place: keyed per rule, so "camera covered" never merges with, boosts, or lifts the learned-model cap of
            # an unrelated detection. Events on the same track still corroborate (see _matches), and a whole-view
            # event mapped to a site zone joins that zone like any other.
            return f"{e.camera_id}:_view:{e.rule}"
        return e.site_zone or f"{e.camera_id}:{e.zone_id}"

    def _matches(self, a: Alert, e: Event) -> bool:
        if a.status == "dismissed" or e.t_end - a.updated_at > self.window:
            return False
        if is_demo(e) != any(is_demo(x) for x in a.events):  # demo and observed events never corroborate
            return False
        for x in a.events:
            if self._key(x) == self._key(e):
                return True
            if x.camera_id == e.camera_id and set(x.track_ids) & set(e.track_ids):
                return True
        return False

    @staticmethod
    def _score(events: list[Event]) -> float:
        return explain_priority(events)[0]

    @staticmethod
    def _title(events: list[Event], title: str) -> str:
        return DEMO_PREFIX + title if any(is_demo(x) for x in events) else title

    def add(self, e: Event) -> tuple[Alert, bool]:
        """Returns (alert, is_new)."""
        self.open = [a for a in self.open if e.t_end - a.updated_at <= self.window]
        for a in self.open:
            if self._matches(a, e):
                a.events.append(e)
                a.updated_at = e.t_end
                a.score, a.priority, a.factors = explain_priority(a.events)
                a.cameras = sorted({x.camera_id for x in a.events})
                a.zones = sorted({x.zone_name for x in a.events})
                rules = sorted({LABEL[x.rule] for x in a.events})
                a.title = self._title(a.events, f"{' + '.join(rules)} — {a.zones[0]}")
                return a, False
        score, prio, factors = explain_priority([e])
        a = Alert(id=f"A{int(time.time())}-{next(self._ids)}", title=self._title([e], f"{LABEL[e.rule]} — {e.zone_name}"),
                  priority=prio, score=score, created_at=e.t_end, updated_at=e.t_end,
                  cameras=[e.camera_id], zones=[e.zone_name], events=[e], model=dict(self.model_info),
                  factors=factors)
        self.open.append(a)
        return a, True
