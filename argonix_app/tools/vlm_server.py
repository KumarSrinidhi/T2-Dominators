"""Multi-model VLM explainer service for the GPU box. One model resident at a time (8 GB RAM container);
switching unloads the previous one. Uses CUDA when available, otherwise CPU-capable small models only.

    python vlm_server.py --port 8100 [--preload HuggingFaceTB/SmolVLM-500M-Instruct]
Laptop:  ssh -N -L 8100:127.0.0.1:8100 root@<box>  then  uv run python run.py --vlm-url http://127.0.0.1:8100

Endpoints: GET /health (lists "features"), GET /models, POST /load, POST /explain (one image_b64, or several
images_b64 in time order, e.g. frames of an evidence clip), POST /chat (text-only chat on the resident model, used
by Argus's natural-language investigation; it never loads a different model, so it costs no extra VRAM).
"""
import argparse
import base64
import gc
import io
import os
import threading
import time
from typing import Optional

import torch
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

CUDA = torch.cuda.is_available()
# params_b: parameters in billions; cpu: small enough for the 8 GB / 7-core container without a GPU.
REGISTRY = {
    "HuggingFaceTB/SmolVLM-256M-Instruct": {"params_b": 0.26, "cpu": True, "note": "fastest, weakest"},
    "HuggingFaceTB/SmolVLM-500M-Instruct": {"params_b": 0.5, "cpu": True, "note": "CPU default"},
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct": {"params_b": 2.2, "cpu": True, "note": "bf16 on CPU, slow"},
    "Qwen/Qwen3-VL-2B-Instruct": {"params_b": 2.1, "cpu": True, "note": "bf16 on CPU, slow"},
    "Qwen/Qwen2.5-VL-3B-Instruct": {"params_b": 3.8, "cpu": False, "note": "GPU default"},
    "Qwen/Qwen3-VL-4B-Instruct": {"params_b": 4.4, "cpu": False, "note": "GPU"},
    "OpenGVLab/InternVL3-2B-hf": {"params_b": 2.1, "cpu": False, "note": "GPU"},
    "Qwen/Qwen2.5-VL-7B-Instruct": {"params_b": 8.3, "cpu": False, "note": "GPU, ~17 GB bf16 of 20 GB VRAM"},
}
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30 if CUDA else 0.0
RESERVE_GB = float(os.environ.get("VLM_RESERVE_GB", "1.0"))  # left for the YOLO detector sharing the GPU


def plan(info: dict) -> Optional[str]:
    """How a model would load here: 'bf16', '4bit', 'cpu', or None (does not fit)."""
    if not CUDA:
        return "cpu" if info["cpu"] else None
    free = VRAM_GB - RESERVE_GB
    if info["params_b"] * 2.1 + 1.0 <= free:
        return "bf16"
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return None
    return "4bit" if info["params_b"] * 0.7 + 1.2 <= free else None


DEFAULT = next((k for k in ["Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen3-VL-2B-Instruct",
                            "HuggingFaceTB/SmolVLM-500M-Instruct"] if plan(REGISTRY[k])), None)

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8100)
ap.add_argument("--preload", default=DEFAULT)
ap.add_argument("--max-side", type=int, default=768)
ap.add_argument("--multi-max-side", type=int, default=448, help="per-frame size when several images are sent")
ap.add_argument("--max-images", type=int, default=4)
args = ap.parse_args()
FEATURES = ["explain", "multi_image", "chat"]
MAX_CHAT_CHARS = 16000
if not CUDA:
    torch.set_num_threads(int(os.environ.get("VLM_THREADS", "7")))

MODELS_DIR = os.path.expanduser(os.environ.get("VLM_MODELS_DIR", "~/argonix_vlm/models"))
SKIP = (".onnx", ".gguf", ".bin", ".h5", ".msgpack", ".ot", ".pth")


def fetch(repo: str) -> str:
    """Download a HF repo with plain HTTPS (huggingface_hub's transfer hung on this box). Resumable, skips ONNX/GGUF."""
    import json
    import urllib.request
    dest = os.path.join(MODELS_DIR, repo.replace("/", "--"))
    if os.path.exists(os.path.join(dest, ".complete")):
        return dest
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=30) as r:
        files = [s["rfilename"] for s in json.load(r)["siblings"]]
    files = [f for f in files if not f.startswith("onnx/") and not f.endswith(SKIP)]
    for f in files:
        out = os.path.join(dest, f)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out):
            continue
        state["loading"] = f"{repo}: downloading {f}"
        _parallel_get(f"https://huggingface.co/{repo}/resolve/main/{f}", out, f"{repo}: {f}")
    open(os.path.join(dest, ".complete"), "w").close()
    return dest


def _parallel_get(url: str, out: str, label: str, chunk: int = 16 << 20, workers: int = 16) -> None:
    """Ranged parallel download with per-chunk retries (the HF CDN is ~0.5 MB/s per connection from here)."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        size = int(r.headers.get("Content-Length") or 0)
    if size < chunk:  # small file: single request
        with urllib.request.urlopen(url, timeout=120) as r, open(out + ".part", "wb") as fh:
            fh.write(r.read())
        os.replace(out + ".part", out)
        return
    part, done = out + ".part", out + ".chunks"
    finished = set(open(done).read().split()) if os.path.exists(done) else set()
    with open(part, "ab") as fh:
        fh.truncate(size)
    ranges = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk) if str(s) not in finished]
    lock = threading.Lock()

    def get(rng):
        s, e = rng
        for attempt in range(8):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={s}-{e}"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = r.read()
                if len(data) != e - s + 1:
                    raise IOError("short read")
                with lock:
                    with open(part, "r+b") as fh:
                        fh.seek(s); fh.write(data)
                    with open(done, "a") as fh:
                        fh.write(f"{s}\n")
                    finished.add(str(s))
                    state["loading"] = f"{label} {len(finished) * chunk * 100 // size}%"
                return
            except Exception:
                time.sleep(2 * (attempt + 1))
        raise IOError(f"chunk {s}-{e} failed")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(get, ranges))
    os.replace(part, out)
    os.remove(done)


state = {"name": None, "model": None, "processor": None, "loading": None, "error": None}
lock = threading.Lock()
app = FastAPI()


def _load(name: str) -> None:
    info = REGISTRY.get(name)
    if info is None:
        raise HTTPException(400, f"unknown model {name}")
    how = plan(info)
    if how is None:
        raise HTTPException(409, f"{name} does not fit here (device {'cuda %.1f GB' % VRAM_GB if CUDA else 'cpu'})")
    if state["name"] == name:
        return
    state.update(model=None, processor=None, name=None, loading=name, error=None)
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()
    try:
        dtype = torch.bfloat16 if (CUDA or info["params_b"] > 1) else torch.float32
        path = fetch(name)
        state["loading"] = f"{name}: loading weights ({how})"
        proc = AutoProcessor.from_pretrained(path)
        if how == "4bit":
            from transformers import BitsAndBytesConfig
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForImageTextToText.from_pretrained(path, quantization_config=q, device_map="cuda:0").eval()
        else:
            model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype).to("cuda" if CUDA else "cpu").eval()
        state.update(model=model, processor=proc, name=name, mode=how)
        print(f"loaded {name} ({how})", flush=True)
    except Exception as e:
        state["error"] = repr(e)
        raise
    finally:
        state["loading"] = None


@app.get("/health")
def health():
    return {"device": "cuda" if CUDA else "cpu", "vram_gb": round(VRAM_GB, 1), "loaded": state["name"],
            "mode": state.get("mode"), "loading": state["loading"], "error": state["error"], "features": FEATURES,
            "max_images": args.max_images}


@app.get("/models")
def models():
    return [{"id": k, **v, "available": plan(v) is not None, "mode": plan(v), "loaded": k == state["name"]}
            for k, v in REGISTRY.items()]


@app.post("/load")
def load(body: dict = Body(...)):
    name = body["model"]
    threading.Thread(target=lambda: _locked_load(name), daemon=True).start()
    return {"loading": name}


def _locked_load(name):
    with lock:
        try:
            _load(name)
        except Exception:
            pass


def _generate(proc, model, msgs: list, images: Optional[list], max_new_tokens: int) -> tuple[str, int, int, float]:
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = (proc(text=[text], images=images, return_tensors="pt") if images
              else proc(text=[text], return_tensors="pt")).to(model.device)
    inputs = {k: (v.to(model.dtype) if v.is_floating_point() else v) for k, v in inputs.items()}
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    n_in = int(inputs["input_ids"].shape[1])
    ans = proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
    return ans.strip(), n_in, int(out.shape[1] - n_in), time.time() - t0


@app.post("/explain")
def explain(body: dict = Body(...)):
    blobs = body.get("images_b64") or ([body["image_b64"]] if body.get("image_b64") else [])
    if not isinstance(blobs, list) or not blobs or len(blobs) > args.max_images:
        raise HTTPException(400, f"send image_b64 or 1-{args.max_images} images_b64")
    side = args.max_side if len(blobs) == 1 else args.multi_max_side
    imgs = []
    for b in blobs:
        img = Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
        img.thumbnail((side, side))
        imgs.append(img)
    with lock:
        _load(body.get("model") or state["name"] or DEFAULT)
        proc, model = state["processor"], state["model"]
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": body["prompt"]}]}]
        ans, n_in, n_out, lat = _generate(proc, model, msgs, imgs, int(body.get("max_new_tokens", 120)))
        return {"text": ans, "model": state["name"], "device": "cuda" if CUDA else "cpu",
                "latency_s": round(lat, 2), "n_images": len(imgs), "prompt_tokens": n_in, "completion_tokens": n_out}


@app.post("/chat")
def chat(body: dict = Body(...)):
    """Text-only chat on the resident model: {"messages": [{"role": "system"|"user"|"assistant", "content": str}],
    "max_new_tokens": int <= 512}. Never switches models (no extra VRAM); 503 while a model is loading."""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or len(msgs) > 20:
        raise HTTPException(400, "messages must be a list of 1-20 {role, content} objects")
    total = 0
    conv = []
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") \
                or not isinstance(m.get("content"), str):
            raise HTTPException(400, "each message needs role system|user|assistant and string content")
        total += len(m["content"])
        conv.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
    if total > MAX_CHAT_CHARS:
        raise HTTPException(413, f"messages exceed {MAX_CHAT_CHARS} characters")
    if state["loading"]:
        raise HTTPException(503, f"model loading: {state['loading']}")
    max_new = max(1, min(512, int(body.get("max_new_tokens", 200))))
    with lock:
        if state["model"] is None:
            _load(DEFAULT)
        proc, model = state["processor"], state["model"]
        ans, n_in, n_out, lat = _generate(proc, model, conv, None, max_new)
        return {"text": ans, "model": state["name"], "device": "cuda" if CUDA else "cpu", "latency_s": round(lat, 2),
                "prompt_tokens": n_in, "completion_tokens": n_out}


if __name__ == "__main__":
    if args.preload:
        threading.Thread(target=lambda: _locked_load(args.preload), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
