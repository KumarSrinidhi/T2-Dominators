"""Compute-node registry for the System page: `GET /api/nodes` (alert.read).

Lists the inference node this site uses (a remote `argonix/infer_server.py` node from `detector.remote`, or the
in-process detector) and the VLM explainer node, each with reachability and a summary of its health/metrics.
Every probe has a short timeout and failures are reported per node, so one dead node never breaks the page.
Tokens never leave the server: the UI only sees URLs and the node's own metrics.
"""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends

PROBE_TIMEOUT_S = 3.0
CACHE_S = 2.0


def _gpu_summary(g: dict) -> dict:
    keys = ("available", "name", "util_pct", "mem_used_mb", "mem_total_mb", "torch_reserved_mb", "source")
    return {k: g.get(k) for k in keys if k in g}


async def _probe_inference(remote: dict, site) -> dict:
    import httpx

    from .remote_inference import http_base
    node = {"kind": "inference", "location": "remote", "url": remote["url"], "client_id": remote.get("client_id"),
            "state": "down", "name": None}
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S) as c:
            r = await c.get(http_base(remote["url"]) + "/metrics",
                            headers={"Authorization": f"Bearer {remote['token']}"})
        node["probe_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if r.status_code == 401:
            node.update(state="unauthorized", error="the node rejected this site's token")
        elif r.status_code != 200:
            node.update(state="error", error=f"HTTP {r.status_code}")
        else:
            m = r.json()
            node.update(state="up", name=m.get("node"), uptime_s=m.get("uptime_s"), model=m.get("model"),
                        gpu=_gpu_summary(m.get("gpu") or {}), queue_depth=m.get("queue_depth"),
                        inflight=m.get("inflight"), mean_batch=m.get("mean_batch"), batch_hist=m.get("batch_hist"),
                        latency_ms=m.get("latency_ms"), fps_in=m.get("fps_in"), clients=len(m.get("clients") or []),
                        counters=m.get("counters"))
    except Exception as e:  # noqa: BLE001 - unreachable, timeout, bad JSON
        node.update(state="down", error=type(e).__name__ + (f": {e}" if str(e) else ""))
    node["cameras"] = [{"camera": w.id, **{k: w.stats.get(k) for k in
                                            ("remote_state", "rtt_ms", "rtt_p95_ms", "dropped_remote",
                                             "remote_timeouts", "fallback_frames", "kb_per_frame", "node_ms")}}
                       for w in site.workers]
    return node


async def _probe_vlm(explainer) -> dict:
    base = getattr(explainer, "base", None)
    node = {"kind": "vlm", "location": "remote" if base else "in-process", "url": base, "state": "unknown"}
    if not hasattr(explainer, "health"):
        node["state"] = "configured"
        return node
    t0 = time.perf_counter()
    try:
        h = await asyncio.wait_for(asyncio.to_thread(explainer.health), PROBE_TIMEOUT_S + 1)
        node.update(state="up", probe_ms=round((time.perf_counter() - t0) * 1000, 1), health=h)
    except Exception as e:  # noqa: BLE001
        node.update(state="down", error=type(e).__name__ + (f": {e}" if str(e) else ""))
    return node


def nodes_router(site, need, explainer=None) -> APIRouter:
    router = APIRouter()
    cache: dict = {"t": 0.0, "v": None}

    @router.get("/api/nodes")
    async def nodes(p=Depends(need("alert.read"))):
        if cache["v"] is not None and time.time() - cache["t"] < CACHE_S:  # several open System pages share probes
            return cache["v"]
        remote = getattr(site.backend_factory, "remote", None)
        tasks = []
        if remote:
            tasks.append(_probe_inference(remote, site))
        if explainer is not None:
            tasks.append(_probe_vlm(explainer))
        out = list(await asyncio.gather(*tasks)) if tasks else []
        if not remote:
            m = dict(site.workers[0].backend.info) if site.workers else dict(site.correlator.model_info)
            out.insert(0, {"kind": "inference", "location": "in-process", "state": "up", "name": "this server",
                           "model": m, "cameras": [{"camera": w.id} for w in site.workers]})
        cache.update(t=time.time(), v={"nodes": out, "t": time.time()})
        return cache["v"]

    return router
