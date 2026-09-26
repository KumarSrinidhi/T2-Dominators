"""API for demo scenarios (argonix/scenario.py) and push-notification status/test (argonix/notify.py)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Body, Depends, HTTPException

from .auth import Principal
from .scenario import ScenarioEngine

ENABLE_HINT = ("demo scenarios are off: start with --open-demo or set \"demo\": {\"scenarios\": true} in the site "
               "config")


def scenarios_dir(site) -> Path:
    d = (site.cfg.get("demo") or {}).get("scenarios_dir")
    if d:
        p = Path(d)
        return p if p.is_absolute() else site.config_path.resolve().parent.parent / p
    return site.config_path.resolve().parent.parent / "configs" / "scenarios"


def demo_router(site, need: Callable[[str], Callable], open_demo: Callable[[], bool]) -> APIRouter:
    r = APIRouter()
    engine = ScenarioEngine(site, scenarios_dir(site),
                            audit=lambda actor, action, details: site.store.audit(actor, action, "", details))
    site.scenarios = engine

    def enabled() -> bool:
        return bool(open_demo()) or bool((site.cfg.get("demo") or {}).get("scenarios"))

    def require_enabled() -> None:
        if not enabled():
            raise HTTPException(409, ENABLE_HINT)

    @r.get("/api/demo/scenarios")
    def list_scenarios(p: Principal = Depends(need("settings.edit"))):
        return {"enabled": enabled(), "hint": None if enabled() else ENABLE_HINT,
                "scenarios": engine.scenarios(), **engine.status(),
                "note": "Scenarios inject scripted events labelled DEMO through the real alert path (correlation, "
                        "incidents, push). They are excluded from precision statistics, baselines and hotspots."}

    @r.get("/api/demo/status")
    def demo_status(p: Principal = Depends(need("alert.read"))):
        return {"enabled": enabled(), **engine.status()}

    @r.post("/api/demo/scenarios/{name}/run")
    def run(name: str, p: Principal = Depends(need("settings.edit"))):
        require_enabled()
        try:
            return engine.start(name, p.username)
        except KeyError:
            raise HTTPException(404, f"unknown scenario {name!r}")
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @r.post("/api/demo/stop")
    def stop(p: Principal = Depends(need("settings.edit"))):
        return engine.stop(p.username)

    # ---- notifications ---------------------------------------------------------------------------------------
    def notifier():
        n = getattr(site, "notifier", None)
        if n is None:
            raise HTTPException(503, "notifications are not installed on this site")
        return n

    @r.get("/api/notify")
    def notify_status(p: Principal = Depends(need("settings.edit"))):
        return notifier().status()

    @r.post("/api/notify/test")
    def notify_test(body: dict = Body(default={}), p: Principal = Depends(need("settings.edit"))):
        try:
            results = notifier().test(p.username, str((body or {}).get("channel") or ""))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"results": results, "ok": all(x["ok"] for x in results)}

    return r
