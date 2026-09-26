"""Install the crowd/flow/queue, unattended-object and trajectory-anomaly analyzers on a Site from its config.

    "analytics": {"enabled": true, "half_life_s": 300, "queue_window_s": 300, "line_band_px": 6},   # default ON
    "objects":   {"enabled": false, "classes": {"24": "backpack", "26": "handbag", "28": "suitcase"},
                  "static_s": 10, "unattended_s": 30, "leave_dist_m": 3.0, ...},                     # default OFF
    "anomaly":   {"enabled": false, "min_train": 250, "emit_percentile": 0.99, ...}                  # default OFF
    "vehicles":  {"enabled": false, "classes": [2, 3, 5, 7], "detector_conf": 0.35}                 # default OFF

`vehicles.enabled` asks the detector for COCO car/motorcycle/bus/truck (extra tracker) for the zone rules
`vehicle_in_zone` / `vehicle_stopped` (argonix/rules.py).

The flow analytics are passive statistics (counts, dwell, direction, density, queue) and only raise events for zone
rules an operator switched on (`counterflow`, `density`, `queue_long`, `dispersal`, `convergence`), so they are on by
default. `objects.enabled` also asks the detector for the bag classes (`detector.extra_classes`), tracked by a separate
tracker. The anomaly model is corroboration only (AGENTS.md) and off by default.

    "congestion_forecast": {"enabled": true, "emit_events": false, "min_skill": 0.05, ...}   # API on, events OFF

The zone congestion forecast (argonix/congestion.py) reads the baseline's stored minute occupancy; it answers
/api/analytics/forecast by default and raises its informational `congestion_forecast` event only with `emit_events`.
"""
from __future__ import annotations

import copy
import inspect

from . import flow  # noqa: F401  (registers the flow rules and their defaults before any zone is built)
from .rules import vehicle_classes

OBJECT_KEYS = ("min_conf", "static_s", "unattended_s", "leave_dist_m", "owner_max_dist_m", "near_dist_m", "attend_s",
               "require_owner", "forget_s", "still_px", "still_frac", "max_box_frac", "rearm_s", "criticality",
               "owner_lookback_s", "save_crops")


def _kwargs(cls, cfg: dict, skip=()) -> dict:
    names = set(inspect.signature(cls.__init__).parameters) - {"self"} - set(skip)
    return {k: v for k, v in (cfg or {}).items() if k in names}


def object_classes(cfg: dict) -> dict[int, str]:
    from .objects import DEFAULT_CLASSES
    oc = (cfg.get("objects") or {}).get("classes")
    return {int(k): str(v) for k, v in oc.items()} if oc else dict(DEFAULT_CLASSES)


def detector_config(cfg: dict) -> dict:
    """The detector config the backend should use: a COPY of `detector`, plus the object classes when the
    unattended-object analyzer is enabled and the vehicle classes when `vehicles.enabled` (the saved site config is
    not modified). Objects and vehicles share the second ("extra") tracker, IDs offset by detect.OBJECT_ID_OFFSET."""
    det = copy.deepcopy(cfg.get("detector", {}) or {})
    wanted, confs = [], []
    if (cfg.get("objects") or {}).get("enabled"):
        wanted += list(object_classes(cfg))
        confs.append((cfg.get("objects") or {}).get("detector_conf"))
    vehicles = vehicle_classes(cfg)
    if vehicles:
        wanted += list(vehicles)
        confs.append((cfg.get("vehicles") or {}).get("detector_conf"))
    if wanted:
        extra = [int(c) for c in (det.get("extra_classes") or [])]
        for c in wanted:
            if c not in extra:
                extra.append(c)
        det["extra_classes"] = extra
        confs = [float(c) for c in confs if c is not None]
        if "extra_conf" not in det and confs:
            det["extra_conf"] = min(confs)  # one threshold for the extra tracker: the lowest requested
    return det


def install(site) -> list:
    """Append the configured analyzers to `site.analyzers` and expose them as site.flow / .objects / .anomaly."""
    cfg = site.cfg
    added = []
    site.flow = site.objects = site.anomaly = None
    ac = cfg.get("analytics") or {}
    if ac.get("enabled", True):
        from .flow import FlowAnalyzer
        site.flow = FlowAnalyzer(baseline=getattr(site, "baseline", None), **_kwargs(FlowAnalyzer, ac, ("baseline",)))
        added.append(site.flow)
    oc = cfg.get("objects") or {}
    if oc.get("enabled", False):
        from .objects import UnattendedObjectAnalyzer
        site.objects = UnattendedObjectAnalyzer(labels=list(object_classes(cfg).values()),
                                                **{k: oc[k] for k in OBJECT_KEYS if k in oc})
        added.append(site.objects)
    an = cfg.get("anomaly") or {}
    if an.get("enabled", False):
        from .anomaly import TrajectoryAnomalyAnalyzer
        site.anomaly = TrajectoryAnomalyAnalyzer(data_dir=site.data,
                                                 **_kwargs(TrajectoryAnomalyAnalyzer, an, ("data_dir",)))
        added.append(site.anomaly)
    from .congestion import install as install_congestion
    cf = install_congestion(site)  # appends itself to site.analyzers only when emit_events is on
    site.analyzers.extend(added)
    return added + ([cf] if cf is not None and cf in site.analyzers else [])
