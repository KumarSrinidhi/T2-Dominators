"""Core data types. Events carry observation (what was seen) separately from derivation (why it fired)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

Point = tuple[float, float]
# Non-video sources put "sensor:<source id>" in Event.camera_id (it names the reporting device, not a camera).
SENSOR_PREFIX = "sensor:"


@dataclass
class TrackObs:
    camera_id: str
    track_id: int
    t: float
    bbox: tuple[float, float, float, float]
    cls: str
    conf: float
    ground: Optional[Point] = None


@dataclass
class Event:
    rule: str
    camera_id: str
    zone_id: str
    zone_name: str
    track_ids: list[int]
    t_start: float
    t_end: float
    observation: str
    derivation: dict[str, Any]
    confidence: float
    severity: int
    criticality: int = 1
    site_zone: Optional[str] = None
    ground: Optional[Point] = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def score(self) -> float:
        return self.severity * self.criticality * self.confidence

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_demo(ev) -> bool:
    """True for events scripted by a demo scenario (an Event, or an event dict as stored in an alert body)."""
    der = (ev.get("derivation") if isinstance(ev, dict) else getattr(ev, "derivation", None)) or {}
    details = der.get("details") if isinstance(der.get("details"), dict) else {}
    # details.demo_scenario: a demo report that went through the access-control sensor ingestion
    return bool(der.get("demo_scenario") or details.get("demo_scenario") is True) or \
        (isinstance(ev, dict) and bool(ev.get("demo")))


def alert_is_demo(a: dict) -> bool:
    return bool(a.get("demo")) or any(is_demo(e) for e in a.get("events") or [])


@dataclass
class Alert:
    id: str
    title: str
    priority: str
    score: float
    created_at: float
    updated_at: float
    cameras: list[str]
    zones: list[str]
    events: list[Event]
    status: str = "new"  # new | acknowledged | dismissed | escalated
    operator_note: str = ""
    model: dict[str, Any] = field(default_factory=dict)
    explanation: Optional[str] = None  # optional VLM text, never used for triggering
    factors: dict[str, Any] = field(default_factory=dict)  # priority breakdown ("why was I shown this")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if any(is_demo(e) for e in self.events):
            d["demo"] = True
            d["demo_note"] = ("DEMO: scripted by a demo scenario (argonix/scenario.py), not observed. Excluded from "
                              "precision statistics, baselines and hotspots.")
        if any(e.derivation.get("learned_model") for e in self.events):
            d["disclaimer"] = ("Includes learned-model inferences (see each event's model and version): they can be "
                               "wrong, confidence is uncalibrated, and the operator must verify the evidence. "
                               "Decision rests with the operator.")
        else:
            d["disclaimer"] = ("Deterministic-rule alert on observed motion/position only; "
                               "confidence is heuristic and uncalibrated. Decision rests with the operator.")
        if any(e.derivation.get("modality", "video") != "video" for e in self.events):
            d["disclaimer"] += (" Access-control events are states reported by the door controller or panel; they "
                                "are not verified by video and say nothing about who was at the door.")
        return d
