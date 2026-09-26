"""Install the weapon-like-object and altercation analyzers on a Site from its config (both off by default).

    "threats": {
      "objects": {"enabled": true, "fps": 3, "min_conf": 0.5, "k": 3, "n": 5, "person_crops": true,
                  "models": [{"weights": "data/models/yolo26n.pt", "name": "coco-yolo26n",
                              "classes": {"43": "knife", "34": "bat", "76": "scissors"}},
                             {"weights": "data/models/argonix_weapons_yolo26n.pt",
                              "classes": {"0": "knife", "1": "handgun"}}]},
      "altercation": {"enabled": true, "pose_weights": "data/models/yolo26n-pose.pt",
                      "classifier": "data/models/argonix_altercation_gru.pt", "threshold": null, "persist_s": 1.5}
    }

A model file that is missing is skipped with a warning in `site.threat_status` (the object analyzer then runs with
the models that loaded; the altercation analyzer falls back to the heuristic if the classifier is missing, and is
not installed without a pose model). `threshold: null` uses the threshold stored in the classifier checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .rules import register_rule

OBJECT_RULE, ALTERCATION_RULE = "object_threat", "altercation"
register_rule(OBJECT_RULE, "Possible weapon-like object", 3)
register_rule(ALTERCATION_RULE, "Possible physical altercation", 3)

DEFAULT_OBJECT_MODELS = [{"weights": "data/models/yolo26n.pt", "name": "coco-yolo26n",
                          "classes": {"43": "knife", "34": "bat", "76": "scissors"}}]


def _resolve(site, p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    q = Path(p)
    if q.is_absolute():
        return str(q)
    base = getattr(site, "_base", Path.cwd())
    return str(base / q) if (base / q).exists() else str(q)


def _exists_or_name(p: str) -> bool:
    """A file on disk, or a bare official name Ultralytics can download (e.g. yolo26n-pose.pt)."""
    q = Path(p)
    return q.exists() or (q.parent == Path(".") and q.name.startswith("yolo"))


def make_object_model(spec: dict, device: str, models_dir: Path):
    from .threats import UltralyticsObjectModel
    return UltralyticsObjectModel(spec["weights"], spec["classes"], name=spec.get("name", ""),
                                  version=spec.get("version", ""), device=spec.get("device", device),
                                  imgsz=int(spec.get("imgsz", 640)), conf=float(spec.get("conf", 0.25)),
                                  crop_imgsz=int(spec.get("crop_imgsz", 320)), models_dir=models_dir)


def make_pose(weights: str, device: str, imgsz: int, models_dir: Path):
    from .altercation import UltralyticsPose
    return UltralyticsPose(weights, device=device, imgsz=imgsz, models_dir=models_dir)


def make_scorer(path: str, device: str):
    from .altercation import GRUScorer
    return GRUScorer(path, device="cpu" if device == "auto" else device)


def install(site, object_model_factory: Optional[Callable] = None, pose_factory: Optional[Callable] = None,
            scorer_factory: Optional[Callable] = None) -> list:
    """Append the configured analyzers to `site.analyzers`. Returns the analyzers added (empty if disabled)."""
    cfg = (site.cfg.get("threats") or {})
    status = {"objects": "disabled", "altercation": "disabled", "warnings": []}
    site.threat_status = status
    added = []
    models_dir = Path(getattr(site, "_base", Path.cwd())) / "data" / "models"
    omf = object_model_factory or make_object_model
    pf = pose_factory or make_pose
    sf = scorer_factory or make_scorer

    oc = cfg.get("objects") or {}
    if oc.get("enabled", False):
        from .threats import ObjectThreatAnalyzer
        models = []
        for spec in oc.get("models") or DEFAULT_OBJECT_MODELS:
            spec = {**spec, "weights": _resolve(site, spec.get("weights"))}
            if not spec["weights"] or not _exists_or_name(spec["weights"]):
                status["warnings"].append(f"object model not found, skipped: {spec.get('weights')}")
                continue
            try:
                models.append(omf(spec, oc.get("device", "auto"), models_dir))
            except Exception as e:  # noqa: BLE001 - one bad model must not stop the site
                status["warnings"].append(f"object model {spec.get('weights')} failed to load: {e!r}"[:300])
        if models:
            an = ObjectThreatAnalyzer(
                models, fps=float(oc.get("fps", 3.0)), min_conf=float(oc.get("min_conf", 0.5)),
                k=int(oc.get("k", 3)), n=int(oc.get("n", 5)), labels=oc.get("labels"),
                label_min_conf=oc.get("label_min_conf"), require_person=bool(oc.get("require_person", True)),
                assoc_margin=float(oc.get("assoc_margin", 0.35)), person_crops=bool(oc.get("person_crops", True)),
                crop_below_px=float(oc.get("crop_below_px", 260)), max_crops=int(oc.get("max_crops", 4)),
                rearm_s=float(oc.get("rearm_s", 60)), criticality=int(oc.get("criticality", 2)),
                max_box_frac=float(oc.get("max_box_frac", 0.25)))
            added.append(an)
            status["objects"] = f"enabled ({', '.join(m.info.get('name', '?') for m in models)})"
        else:
            status["objects"] = "enabled but no model loaded"

    ac = cfg.get("altercation") or {}
    if ac.get("enabled", False):
        from .altercation import AltercationAnalyzer, load_meta
        device = ac.get("device", "auto")
        pose_w = _resolve(site, ac.get("pose_weights", "yolo26n-pose.pt"))
        pose = None
        try:
            pose = pf(pose_w, device, int(ac.get("pose_imgsz", 640)), models_dir)
        except Exception as e:  # noqa: BLE001
            status["warnings"].append(f"pose model {pose_w} failed to load: {e!r}"[:300])
        if pose is not None:
            scorer, meta = None, {}
            clf = _resolve(site, ac.get("classifier"))
            if clf and Path(clf).exists():
                try:
                    scorer = sf(clf, ac.get("classifier_device", "cpu"))
                    meta = dict(getattr(scorer, "meta", {}) or load_meta(clf))
                except Exception as e:  # noqa: BLE001
                    status["warnings"].append(f"altercation classifier {clf} failed to load: {e!r}"[:300])
            elif clf:
                status["warnings"].append(f"altercation classifier not found, using heuristic: {clf}")
            thr = ac.get("threshold")
            if thr is None:
                thr = meta.get("threshold", 0.6) if scorer is not None else 0.5
            info = {"name": meta.get("name", "pose-heuristic" if scorer is None else "pose-gru"),
                    "version": meta.get("version", "heuristic-v1" if scorer is None else "unknown")}
            h_thr = float(ac.get("heuristic_threshold", meta.get("heuristic_threshold", 1.0)))
            an = AltercationAnalyzer(
                pose, scorer, threshold=float(thr), model_fps=float(meta.get("fps", ac.get("model_fps", 5.0))),
                window=int(meta.get("window", ac.get("window", 8))), persist_s=float(ac.get("persist_s", 1.5)),
                pair_max_dist=float(ac.get("pair_max_dist", meta.get("pair_max_dist", 1.5))),
                min_valid=float(ac.get("min_valid", 0.75)), rearm_s=float(ac.get("rearm_s", 30)),
                criticality=int(ac.get("criticality", 2)), heuristic_scale=h_thr / 0.5 if scorer is None else 1.0,
                model_info=info)
            added.append(an)
            status["altercation"] = f"enabled ({info['name']} {info['version']})"
    site.analyzers.extend(added)
    return added
