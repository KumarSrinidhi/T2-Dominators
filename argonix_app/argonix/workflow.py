"""Operator workflow for alerts: an explicit status state machine, escalation with a reason, the "unclear" second-review
flag, and per-priority acknowledge SLAs.

Transitions (anything else is rejected with HTTP 409):

    new           -> acknowledged | escalated | dismissed (needs a disposition)
    acknowledged  -> escalated | resolved | dismissed (both need a disposition) | acknowledged (only to flag "unclear")
    escalated     -> acknowledged | resolved (needs a disposition)
    resolved      -> new | acknowledged      reopen: needs incident.assign / incident.merge / user.admin
    dismissed     -> new | acknowledged      reopen: same

Escalating needs a `reason` (optional `to_role` / `to_user`). The "unclear" disposition closes nothing: the alert stays
(or becomes) acknowledged and is flagged `second_review` until someone resolves or dismisses it with a final
disposition. Every decision stays with the operator; nothing here changes an alert's priority or score.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from .feedback import CLOSED, DISPOSITIONS

TRANSITIONS: dict[str, tuple[str, ...]] = {
    "new": ("acknowledged", "escalated", "dismissed"),
    "acknowledged": ("escalated", "resolved", "dismissed"),
    "escalated": ("acknowledged", "resolved"),
    "resolved": ("new", "acknowledged"),
    "dismissed": ("new", "acknowledged"),
}
STATUSES = frozenset(TRANSITIONS)
REOPEN_FROM = frozenset(CLOSED)
REOPEN_PERMS = ("incident.assign", "incident.merge", "user.admin")
UNCLEAR = "unclear"
# Alert fields set by operators. The pipeline copies them onto every re-correlated version of an alert.
OPERATOR_FIELDS = ("status", "operator_note", "disposition", "explanation", "second_review", "escalation",
                   "acknowledged_at", "acknowledged_by", "status_changed_at", "reopened_at", "reopen_count",
                   "second_review_flagged_by", "second_reviewed_by", "legal_hold",
                   "health_recovery")  # not an operator decision, but kept the same way when new events re-correlate
DEFAULT_SLA_S: dict[str, Optional[float]] = {"P1": 60.0, "P2": 300.0, "P3": 1800.0, "P4": None}
DEFAULT_FLOOD = {"threshold": 10, "window_s": 600}


class TransitionError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code, self.msg = code, msg


def sla_config(cfg: dict) -> dict[str, Optional[float]]:
    """Per-priority acknowledge SLA in seconds (None = no SLA). Site config key "sla_s" overrides the defaults."""
    raw = cfg.get("sla_s") if isinstance(cfg.get("sla_s"), dict) else {}
    out = dict(DEFAULT_SLA_S)
    for p, v in raw.items():
        if p in out:
            out[p] = None if v in (None, 0, False) else float(v)
    return out


def flood_config(cfg: dict) -> dict:
    raw = cfg.get("alert_flood") if isinstance(cfg.get("alert_flood"), dict) else {}
    return {"threshold": int(raw.get("threshold", DEFAULT_FLOOD["threshold"])),
            "window_s": float(raw.get("window_s", DEFAULT_FLOOD["window_s"]))}


def public_config(cfg: dict) -> dict:
    """What the dashboard needs to offer only legal actions and show SLA / flood state (GET /api/config)."""
    return {"transitions": {k: list(v) for k, v in TRANSITIONS.items()}, "reopen_perms": list(REOPEN_PERMS),
            "sla_s": sla_config(cfg), "flood": flood_config(cfg), "unclear": UNCLEAR}


def plan(alert: dict, body: dict, can: Callable[[str], bool], status_perm: dict, roles=(), actor: str = "operator") -> dict:
    """Validate a status request against the state machine and the caller's permissions.

    Returns the fields to write plus audit details; raises TransitionError(400 | 403 | 409)."""
    st = body.get("status")
    if st not in STATUSES:
        raise TransitionError(400, f"status must be one of {sorted(STATUSES)}")
    cur = alert.get("status") or "new"
    disp = body.get("disposition") or ""
    if disp and disp not in DISPOSITIONS:
        raise TransitionError(400, f"disposition must be one of {sorted(DISPOSITIONS)}")
    note = str(body.get("note") or "")[:2000]
    unclear = disp == UNCLEAR
    if unclear and st != "acknowledged":
        raise TransitionError(400, "'unclear' keeps the alert acknowledged for a second review: send it with "
                                   "status 'acknowledged' (use a final disposition to resolve or dismiss)")
    same = st == cur
    reopen = cur in REOPEN_FROM and not same
    # permission first (403), then legality (409): a caller without the right learns nothing about the state
    if reopen:
        if not any(can(p) for p in REOPEN_PERMS):
            raise TransitionError(403, f"reopening a {cur} alert needs one of the permissions {list(REOPEN_PERMS)} "
                                       "(supervisor or admin)")
    elif not can(status_perm[st]):
        raise TransitionError(403, f"permission '{status_perm[st]}' required")
    if same and not (st == "acknowledged" and unclear):
        raise TransitionError(409, f"alert is already {cur}")
    if not same and st not in TRANSITIONS.get(cur, ()):
        legal = ", ".join(TRANSITIONS.get(cur, ())) or "none"
        raise TransitionError(409, f"illegal transition {cur} -> {st} (from {cur} the next status can be: {legal})")
    if st in CLOSED and not disp:
        raise TransitionError(400, "closing an alert requires a disposition")
    now = time.time()
    fields: dict = {"status": st, "status_changed_at": now}
    details: dict = {"from": cur}
    if disp:
        fields["disposition"] = disp
    if st == "escalated":
        reason = str(body.get("reason") or "").strip()
        if not reason:
            raise TransitionError(400, "escalating needs a reason (field 'reason'; optional 'to_role' / 'to_user')")
        to_role = str(body.get("to_role") or "").strip() or None
        if to_role is not None and roles and to_role not in roles:
            raise TransitionError(400, f"to_role must be one of {list(roles)}")
        to_user = (str(body.get("to_user") or "").strip()[:64]) or None
        fields["escalation"] = {"reason": reason[:500], "to_role": to_role, "to_user": to_user, "t": now, "by": actor}
        details.update(reason=reason[:500], to_role=to_role, to_user=to_user)
    if cur == "new" and st != "new":
        fields.update(acknowledged_at=now, acknowledged_by=actor)   # first response: ends the acknowledge SLA clock
    if unclear:
        fields.update(second_review=True, second_review_flagged_by=actor)
        details["second_review"] = True
    elif st in CLOSED and alert.get("second_review"):
        fields.update(second_review=False, second_reviewed_by=actor)
        details["second_review_cleared"] = True
    if reopen:
        fields.update(reopened_at=now, reopen_count=int(alert.get("reopen_count") or 0) + 1, second_review=False)
        if not disp:
            fields["disposition"] = None         # the earlier verdict no longer holds (kept in the audit trail)
        if st == "new":
            fields.update(acknowledged_at=None, acknowledged_by=None)   # back in the triage queue: the SLA clock restarts at reopened_at
        details["reopen"] = True
    return {"fields": fields, "note": note, "disposition": disp, "details": details, "from": cur, "to": st}

