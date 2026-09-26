"""Clients for the optional VLM server (tools/vlm_server.py on the GPU box).

The VLM only describes an alert's evidence on operator request. Its text is stored as an assistive `explanation`,
never used to trigger, score or close alerts. With a server that supports it (`/health` lists "multi_image"),
3-4 frames sampled from the evidence clip are sent in time order; otherwise the single evidence snapshot.
The same server's `/chat` endpoint serves the natural-language investigation (argonix.investigate) as a text
model on the already-loaded weights.
"""
from __future__ import annotations

import base64
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import httpx

# Kept to one plain question: sub-1B models (the CPU fallback) ignore multi-part instructions. The rule's own
# observation is shown next to this text in the UI, so the VLM does not need it. {obs} is kept for larger models.
PROMPT = ("How many people are in this image, where are they standing, and what are they doing? "
          "Answer in at most 3 short sentences and describe only what is visible.")
FRAMES_PROMPT = ("These {n} images are frames in time order from a short evidence clip (about {dur:.0f} seconds). "
                 "In at most 3 short sentences, summarise how many people are visible, where they are, and how they "
                 "move over the clip. Do not describe the frames one by one. Describe only what is visible.")
EXPLAIN_MAX_TOKENS = 160  # room for 3 sentences; the server default (120) cut answers mid-sentence


def trim_to_sentence(text: str) -> tuple[str, bool]:
    """Drop a trailing unfinished sentence (generation stopped at the token limit). Returns (text, trimmed)."""
    t = text.strip()
    if not t or t[-1] in ".!?\"')":
        return t, False
    cut = max(t.rfind(". "), t.rfind("! "), t.rfind("? "))
    return (t[:cut + 1], True) if cut > 0 else (t, False)


def sample_clip_frames(path: Path, n: int = 4, max_side: int = 640, quality: int = 85) -> tuple[list[bytes], float]:
    """n JPEG frames spread evenly over a (short) clip, plus the clip duration in seconds. Reads sequentially:
    seeking in freshly written H.264 files is unreliable, and evidence clips are only a few seconds long."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return [], 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
        frames = []
        while len(frames) < 3000:
            ok, f = cap.read()
            if not ok or f is None:
                break
            frames.append(f)
    finally:
        cap.release()
    if not frames:
        return [], 0.0
    n = max(1, min(n, len(frames)))
    idx = sorted({round(i * (len(frames) - 1) / max(1, n - 1)) for i in range(n)})
    out = []
    for i in idx:
        f = frames[i]
        h, w = f.shape[:2]
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out.append(buf.tobytes())
    return out, len(frames) / (fps if 1 <= fps <= 120 else 5.0)



class VlmBusy(RuntimeError):
    """Every VLM slot is taken: the caller should answer 429 instead of queuing an API thread behind the model."""


_SLOTS_LOCK = threading.Lock()


@contextmanager
def vlm_slot(site, wait_s: float = 2.0):
    """Hold one of the site's `vlm_max_concurrent` (default 2) VLM slots for a blocking model call.

    Each Explain / Summarise request holds an API worker thread for as long as the model takes (up to minutes on a
    slow host), so without a cap a few clicks could exhaust the thread pool and stall the whole dashboard."""
    with _SLOTS_LOCK:
        sem = getattr(site, "vlm_slots", None)
        if sem is None:
            sem = site.vlm_slots = threading.BoundedSemaphore(max(1, int((site.cfg or {}).get("vlm_max_concurrent", 2))))
    if not sem.acquire(timeout=wait_s):
        raise VlmBusy("the VLM is busy with other requests; try again in a moment")
    try:
        yield
    finally:
        sem.release()

class RemoteExplainer:
    def __init__(self, url: str, timeout_s: float = 600.0, frames: int = 4, privacy: Optional[dict] = None):
        self.base = url.rstrip("/").removesuffix("/explain").removesuffix("/chat")
        self.timeout = timeout_s
        self.frames = int(frames)
        # site config "privacy": {"blur_faces_for_vlm": true, "face_model": ...}: faces (and the head region of the
        # recorded person boxes) are blurred in every image before it leaves for the VLM server (argonix.privacy)
        privacy = privacy or {}
        self.blur_faces = bool(privacy.get("blur_faces_for_vlm"))
        self.face_model = privacy.get("face_model")
        self._features: tuple[float, set] = (0.0, set())

    def models(self) -> list[dict]:
        return httpx.get(f"{self.base}/models", timeout=10).json()

    def health(self) -> dict:
        return httpx.get(f"{self.base}/health", timeout=10).json()

    def features(self) -> set:
        """Capabilities advertised by the server (cached 60 s). Old servers advertise none."""
        t, f = self._features
        if time.monotonic() - t > 60:
            try:
                f = set(self.health().get("features") or [])
            except Exception:  # noqa: BLE001 - unreachable -> single-image request (the call itself will report)
                f = set()
            self._features = (time.monotonic(), f)
        return f

    def __call__(self, alert: dict, data_dir: Path, model: Optional[str] = None) -> str:
        e = alert["events"][-1]
        ev = e.get("evidence") or {}
        snap = ev.get("snapshot")
        clip = ev.get("clip")
        frames, dur = [], 0.0
        if self.frames > 1 and clip and (Path(data_dir) / clip).exists() and "multi_image" in self.features():
            try:
                frames, dur = sample_clip_frames(Path(data_dir) / clip, self.frames)
            except Exception:  # noqa: BLE001 - fall back to the snapshot
                frames = []
        if not snap and not frames:
            raise RuntimeError("this alert has no evidence frame to describe")
        body: dict = {"prompt": PROMPT.format(obs=e.get("observation", "")), "max_new_tokens": EXPLAIN_MAX_TOKENS}
        if snap and (Path(data_dir) / snap).exists():
            body["image_b64"] = base64.b64encode((Path(data_dir) / snap).read_bytes()).decode()
        if len(frames) >= 2:
            body["images_b64"] = [base64.b64encode(b).decode() for b in frames]
            body["prompt"] = FRAMES_PROMPT.format(n=len(frames), dur=max(1.0, dur))
        elif frames and "image_b64" not in body:
            body["image_b64"] = base64.b64encode(frames[0]).decode()
        if "image_b64" not in body and "images_b64" not in body:
            raise RuntimeError("evidence snapshot file is missing")
        if self.blur_faces:
            self._blur(body, ev.get("person_boxes"))
        if model:
            body["model"] = model
        r = httpx.post(f"{self.base}/explain", json=body, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(r.text)
        d = r.json()
        n = int(d.get("n_images") or (len(body["images_b64"]) if "images_b64" in body else 1))
        src = f"{n} frames from the evidence clip" if n > 1 else "evidence snapshot"
        if self.blur_faces:
            src += ", faces blurred before sending"
        text, trimmed = d["text"].strip(), False
        if int(d.get("completion_tokens") or 0) >= EXPLAIN_MAX_TOKENS:
            text, trimmed = trim_to_sentence(text)
        if trimmed:
            src += ", unfinished last sentence dropped at the token limit"
        return f"{text} [{d.get('model', 'vlm')} on {d.get('device', '?')}, {d.get('latency_s', '?')} s, {src}]"

    def _blur(self, body: dict, person_boxes) -> None:
        from .privacy import anonymise_jpeg, get_detector
        det = get_detector(self.face_model)

        def blur(b64: str, boxes=None) -> str:
            out, _ = anonymise_jpeg(base64.b64decode(b64), boxes, det)
            if out is None:  # never send an image we could not anonymise
                raise RuntimeError("could not anonymise an evidence image for the VLM (privacy.blur_faces_for_vlm)")
            return base64.b64encode(out).decode()
        if "image_b64" in body:  # the snapshot: its person boxes were recorded with it
            body["image_b64"] = blur(body["image_b64"], person_boxes)
        if "images_b64" in body:
            body["images_b64"] = [blur(b) for b in body["images_b64"]]

    def describe(self, images: list[bytes], prompt: str, max_new_tokens: int = 260) -> dict:
        """Free-form prompt over 1..n JPEG images (POST /explain). Returns the server's JSON: text, model, device,
        latency_s, n_images. Several images need a server that advertises "multi_image"; otherwise only the first
        is sent. Used by the multi-camera incident summary (argonix.narrative)."""
        if not images:
            raise RuntimeError("no image to describe")
        imgs = images if len(images) == 1 or "multi_image" in self.features() else images[:1]
        body: dict = {"prompt": prompt, "max_new_tokens": int(max_new_tokens)}
        if len(imgs) == 1:
            body["image_b64"] = base64.b64encode(imgs[0]).decode()
        else:
            body["images_b64"] = [base64.b64encode(b).decode() for b in imgs]
        if self.blur_faces:
            self._blur(body, None)
        r = httpx.post(f"{self.base}/explain", json=body, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        d = r.json()
        d.setdefault("n_images", len(imgs))
        return d


class ChatClient:
    """Text chat on the VLM server's resident model (POST /chat). Used by argonix.investigate."""

    def __init__(self, url: str, timeout_s: float = 45.0):
        self.base = url.rstrip("/").removesuffix("/explain").removesuffix("/chat")
        self.timeout = float(timeout_s)
        self.model_id = self.base  # translation-cache identity (cached queries are re-validated on every use)

    def chat(self, messages: list[dict], max_new_tokens: int = 200) -> dict:
        r = httpx.post(f"{self.base}/chat", json={"messages": messages, "max_new_tokens": int(max_new_tokens)},
                       timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.json()
