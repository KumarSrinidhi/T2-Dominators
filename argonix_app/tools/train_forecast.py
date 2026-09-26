"""Train + evaluate the trajectory forecaster on ETH/UCY (leave-one-scene-out), CPU only, a few minutes.

    uv run python tools/train_forecast.py                 # download (if needed), 5 LOSO folds, final model
    uv run python tools/train_forecast.py --folds eth     # one fold only (quick)
    uv run python tools/train_forecast.py --synthetic     # no download: synthetic walkers (NOT real data)

Data: the ETH (BIWI eth, hotel) and UCY (univ, zara1, zara2) pedestrian trajectories in world metres, sampled every
0.4 s, as redistributed with the leave-one-scene-out train/val/test split in the Trajectron++ repository
(github.com/StanfordASL/Trajectron-plus-plus, experiments/pedestrians/raw; repository licence MIT; the underlying
datasets are the ETH/BIWI Walking Pedestrians and UCY Crowds data, for research use per their authors).
Files are cached under data/datasets/eth_ucy/ (about 10 MB).

Protocol: 8 observed steps (3.2 s) -> 12 predicted steps (4.8 s), all windows with stride 1. For each held-out scene
the model is trained on that fold's train/ files, the epoch is chosen on val/, and ADE/FDE (metres) are reported on
test/ next to the constant-velocity baseline. The shipped model (data/models/argonix_forecast_gru.pt) is then trained
on all five scenes with the median chosen epoch count; its reported numbers are the LOSO fold numbers.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argonix.forecast import (DT, FEATURE_VERSION, OBS_LEN, PRED_LEN, ade_fde, build_model, cv_predict,  # noqa: E402
                              decode, load_eth_ucy, model_inputs, save_checkpoint, synthetic_walkers, to_world,
                              windows)

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
REPO = "StanfordASL/Trajectron-plus-plus"
API = f"https://api.github.com/repos/{REPO}/contents/experiments/pedestrians/raw/{{scene}}/{{split}}"
RAW = f"https://raw.githubusercontent.com/{REPO}/master/experiments/pedestrians/raw/{{scene}}/{{split}}/{{name}}"
DATA = ROOT / "data" / "datasets" / "eth_ucy"


def _get(url: str, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "argonix-train-forecast"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download() -> None:
    for scene in SCENES:
        for split in ("train", "val", "test"):
            d = DATA / scene / split
            if d.exists() and any(d.glob("*.txt")):
                continue
            d.mkdir(parents=True, exist_ok=True)
            names = [x["name"] for x in json.loads(_get(API.format(scene=scene, split=split)))
                     if x["name"].endswith(".txt")]
            for n in names:
                (d / n).write_bytes(_get(RAW.format(scene=scene, split=split, name=n)))
                print(f"  downloaded {scene}/{split}/{n}", flush=True)


def load_split(scene: str, split: str) -> np.ndarray:
    ws = [windows(load_eth_ucy(f)) for f in sorted((DATA / scene / split).glob("*.txt"))]
    ws = [w for w in ws if len(w)]
    return np.concatenate(ws) if ws else np.zeros((0, OBS_LEN + PRED_LEN, 2))


# ---------------------------------------------------------------------------------------------- training
def prepare(seqs: np.ndarray):
    obs, fut = seqs[:, :OBS_LEN], seqs[:, OBS_LEN:]
    disp, R, o = model_inputs(obs)
    fut_local = np.einsum("btj,bij->bti", fut - o[:, None, :], R)
    return disp, fut_local.astype(np.float32), R, o


def train(train_seqs: np.ndarray, val_seqs: np.ndarray | None, epochs: int, hidden: int = 64, seed: int = 0,
          log: bool = True):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(hidden)
    disp, fut, _, _ = prepare(train_seqs)
    X, Y = torch.from_numpy(disp), torch.from_numpy(fut)
    if val_seqs is not None and len(val_seqs):
        vd, vf, vR, vo = prepare(val_seqs)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    steps = torch.arange(1, PRED_LEN + 1, dtype=torch.float32)[None, :, None]
    best = (float("inf"), 0, None)
    bs = 256
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            x, y = X[idx], Y[idx]
            flip = (torch.rand(len(x)) < 0.5).float()[:, None, None] * torch.tensor([0.0, 1.0])  # mirror left/right
            sgn = 1 - 2 * flip
            x, y = x * sgn, y * sgn
            res, ls = model(x)
            pred = steps * x[:, -1:, :] + torch.cumsum(res, 1)
            err = pred - y
            dist = err.norm(dim=-1)
            sig = ls.exp()
            nll = 2 * ls + (err.detach() ** 2).sum(-1) / (2 * sig ** 2)
            loss = dist.mean() + 0.05 * nll.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(x)
        sched.step()
        if val_seqs is not None and len(val_seqs):
            model.eval()
            with torch.no_grad():
                res, _ = model(torch.from_numpy(vd))
            pv = to_world(decode(vd.astype(np.float64), res.numpy().astype(np.float64)), vR, vo)
            ade, fde = ade_fde(pv, val_seqs[:, OBS_LEN:])
            if ade < best[0]:
                best = (ade, ep + 1, {k: v.clone() for k, v in model.state_dict().items()})
            if log:
                print(f"    epoch {ep + 1:2d} loss {tot / len(X):.4f} val ADE {ade:.3f} FDE {fde:.3f}", flush=True)
    if best[2] is not None:
        model.load_state_dict(best[2])
    model.eval()
    return model, best[1] or epochs


def evaluate(model, seqs: np.ndarray) -> dict:
    import torch
    obs, fut = seqs[:, :OBS_LEN], seqs[:, OBS_LEN:]
    disp, R, o = model_inputs(obs)
    with torch.no_grad():
        res, ls = model(torch.from_numpy(disp))
    pred = to_world(decode(disp.astype(np.float64), res.numpy().astype(np.float64)), R, o)
    sig = np.exp(ls.numpy())
    cv = cv_predict(obs)
    g_ade, g_fde = ade_fde(pred, fut)
    c_ade, c_fde = ade_fde(cv, fut)
    g8 = ade_fde(pred[:, :8], fut[:, :8])
    c8 = ade_fde(cv[:, :8], fut[:, :8])
    err = np.linalg.norm(pred - fut, axis=-1)
    # calibration check of the spread: fraction of true positions inside the 2D 1-sigma / 2-sigma circles
    # (for an isotropic 2D Gaussian: 39.3% / 86.5%)
    cover1, cover2 = float((err <= sig).mean()), float((err <= 2 * sig).mean())
    r = lambda v: round(float(v), 3)
    return {"n_windows": int(len(seqs)), "gru": {"ade": r(g_ade), "fde": r(g_fde), "ade_3.2s": r(g8[0]),
                                                 "fde_3.2s": r(g8[1])},
            "cv": {"ade": r(c_ade), "fde": r(c_fde), "ade_3.2s": r(c8[0]), "fde_3.2s": r(c8[1])},
            "sigma_coverage": {"1sigma": r(cover1), "2sigma": r(cover2), "ideal": [0.393, 0.865]}}


def cv_rmse(seqs: np.ndarray) -> list[float]:
    e = np.linalg.norm(cv_predict(seqs[:, :OBS_LEN]) - seqs[:, OBS_LEN:], axis=-1)
    return [round(float(v), 3) for v in np.sqrt((e ** 2).mean(0) / 2)]  # per-axis sigma of an isotropic Gaussian


def hardware() -> str:
    import torch
    return f"{platform.processor() or platform.machine()} ({platform.system()} {platform.release()}), " \
           f"torch {torch.__version__}, {torch.get_num_threads()} threads, CPU"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", nargs="*", default=SCENES)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--synthetic", action="store_true", help="skip ETH/UCY; train on synthetic walkers")
    ap.add_argument("--out", default=str(ROOT / "data" / "models" / "argonix_forecast_gru.pt"))
    ap.add_argument("--no-final", action="store_true", help="only run the LOSO folds")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    source = "ETH/UCY (Trajectron++ mirror, leave-one-scene-out)"
    if not a.synthetic:
        try:
            download()
        except Exception as e:  # noqa: BLE001
            print(f"!! ETH/UCY download failed ({e!r}); FALLING BACK TO SYNTHETIC WALKERS (not real data)")
            a.synthetic = True
    folds, chosen = {}, []
    if a.synthetic:
        source = "SYNTHETIC random walkers (ETH/UCY not used)"
        tr, va, te = synthetic_walkers(6000, 1), synthetic_walkers(1000, 2), synthetic_walkers(2000, 3)
        model, ep = train(tr, va, a.epochs, a.hidden)
        folds["synthetic"] = evaluate(model, te)
        chosen.append(ep)
        all_seqs = np.concatenate([tr, va])
    else:
        for scene in a.folds:
            tr, va, te = (load_split(scene, s) for s in ("train", "val", "test"))
            print(f"== fold {scene}: train {len(tr)} val {len(va)} test {len(te)} windows", flush=True)
            ts = time.time()
            model, ep = train(tr, va, a.epochs, a.hidden, log=False)
            m = evaluate(model, te)
            m["best_epoch"], m["train_s"] = ep, round(time.time() - ts, 1)
            folds[scene] = m
            chosen.append(ep)
            print(f"   {scene}: GRU ADE/FDE {m['gru']['ade']}/{m['gru']['fde']}  vs  CV {m['cv']['ade']}/{m['cv']['fde']}"
                  f"  (best epoch {ep}, {m['train_s']} s)", flush=True)
        all_seqs = np.concatenate([load_split(s, "test") for s in SCENES])
    keys = [k for k in folds]
    avg = {m: {k: round(float(np.mean([folds[f][m][k] for f in keys])), 3) for k in ("ade", "fde", "ade_3.2s", "fde_3.2s")}
           for m in ("gru", "cv")}
    metrics = {"protocol": f"obs {OBS_LEN} x {DT}s -> pred {PRED_LEN} x {DT}s; ADE/FDE in metres; stride-1 windows",
               "source": source, "folds": folds, "average": avg, "hardware": hardware(),
               "measured_at": time.strftime("%Y-%m-%d %H:%M")}
    print(json.dumps({"average": avg}, indent=2))
    if a.no_final:
        return
    ep = int(np.median(chosen)) if chosen else a.epochs
    print(f"== final model on all data ({len(all_seqs)} windows), {ep} epochs", flush=True)
    model, _ = train(all_seqs, None, ep, a.hidden, log=False)
    params = sum(p.numel() for p in model.parameters())
    meta = {"name": "argonix-forecast-gru", "version": time.strftime("v%Y%m%d"), "feature_version": FEATURE_VERSION,
            "hidden": a.hidden, "pred_len": PRED_LEN, "obs_len": OBS_LEN, "dt_s": DT, "params": params,
            "epochs": ep, "trained_on": source, "cv_rmse_per_step": cv_rmse(all_seqs), "metrics": metrics,
            "train_total_s": round(time.time() - t0, 1)}
    save_checkpoint(a.out, model, meta)
    print(f"saved {a.out} ({params} parameters) in {meta['train_total_s']} s")


if __name__ == "__main__":
    main()
