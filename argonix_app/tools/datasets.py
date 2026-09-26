"""Public test footage for Argus: manifest-driven, idempotent, resumable downloader + H.264 converter.

The manifest (`datasets/manifest.json`, committed) lists each dataset's official source, licence/terms, size and
what it exercises in Argus. This tool fetches the files it names into a shared root (default
`~/argonix_datasets`, override with --root or ARGONIX_DATASETS), checks sizes and SHA-256 checksums, and converts
every clip / image sequence to an H.264 mp4 (one file per camera view or sequence) under `<id>/mp4/`.

    uv run python tools/datasets.py list                       # manifest summary + what is present
    uv run python tools/datasets.py fetch   [--only caviar,ucsd] [--pin]
    uv run python tools/datasets.py convert [--only ...]
    uv run python tools/datasets.py all     [--only ...] [--pin] # fetch + convert
    uv run python tools/datasets.py verify  [--only ...]         # re-hash downloaded files against the manifest
    uv run python tools/datasets.py subset  --dest data/real/datasets [--max-mb 300]

Idempotent: finished files are skipped (size + checksum), partial downloads resume with HTTP Range (`*.part`),
conversions are skipped when the mp4 exists and its source has not changed. `--pin` writes the SHA-256 of files
that the manifest has no checksum for back into the manifest (sources rarely publish checksums, so the first
verified download pins them). Sources that are large archives can be sampled without downloading them whole:
`zip_url` entries read only the selected members of a remote zip through HTTP Range requests.

Kaggle sources need KAGGLE_API_TOKEN (Bearer) or KAGGLE_USERNAME + KAGGLE_KEY in the environment; nothing is
written to disk. Other agents may share the root: every dataset is locked while it is being written.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import httpx

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets" / "manifest.json"
STATE_NAME = ".argonix_datasets_state.json"
UA = "argonix-datasets/1 (+research test footage; resumable)"
CHUNK = 1 << 20


# ------------------------------------------------------------------------------------------------ small helpers
def log(*a, **kw) -> None:
    print(*a, flush=True, **kw)


def human(n: Optional[float]) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def default_root() -> Path:
    return Path(os.environ.get("ARGONIX_DATASETS") or Path.home() / "argonix_datasets")


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_manifest(m: dict, path: Path = MANIFEST) -> None:
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class State:
    """Per-root record of verified downloads and conversions (sizes, hashes, source stamps)."""

    def __init__(self, root: Path):
        self.path = root / STATE_NAME
        try:
            self.d = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.d = {"files": {}, "converted": {}}

    def save(self) -> None:
        # other agents may update the same root: merge with what is on disk before writing
        try:
            disk = json.loads(self.path.read_text())
        except (OSError, ValueError):
            disk = {"files": {}, "converted": {}}
        for k in ("files", "converted"):
            disk.setdefault(k, {}).update(self.d.get(k, {}))
        self.d = disk
        tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(disk, indent=1))
        os.replace(tmp, self.path)


class Lock:
    """Exclusive per-dataset lock file (stale after 6 h) so concurrent agents do not write the same files."""

    def __init__(self, path: Path, wait_s: float = 3600):
        self.path, self.wait_s = path, wait_s

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {time.time()}".encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > 6 * 3600:
                        self.path.unlink()
                        continue
                except OSError:
                    continue
                if time.time() - t0 > self.wait_s:
                    raise TimeoutError(f"{self.path} is held by another process")
                time.sleep(2)

    def __exit__(self, *a):
        try:
            self.path.unlink()
        except OSError:
            pass


# ------------------------------------------------------------------------------------------------ HTTP download
def _client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, connect=20.0), headers={"User-Agent": UA})


def _auth_headers(spec: dict) -> dict:
    if spec.get("kaggle"):
        tok = os.environ.get("KAGGLE_API_TOKEN")
        if tok:
            return {"Authorization": f"Bearer {tok}"}
        user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
        if user and key:
            import base64
            return {"Authorization": "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()}
        raise RuntimeError("Kaggle source needs KAGGLE_API_TOKEN or KAGGLE_USERNAME + KAGGLE_KEY in the environment")
    return {}


def spec_url(spec: dict) -> str:
    if spec.get("kaggle"):
        return f"https://www.kaggle.com/api/v1/datasets/download/{spec['kaggle']}"
    return spec["url"]


def download(url: str, dest: Path, headers: Optional[dict] = None, expect_size: Optional[int] = None,
             retries: int = 5) -> int:
    """Resumable download to dest (via dest.part). Returns the final size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    headers = dict(headers or {})
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        h = dict(headers)
        if have:
            h["Range"] = f"bytes={have}-"
        try:
            with _client() as c, c.stream("GET", url, headers=h) as r:
                if r.status_code == 416 and have:  # already complete
                    break
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code} for {url}")
                mode = "ab" if (have and r.status_code == 206) else "wb"
                if mode == "wb":
                    have = 0
                total = r.headers.get("content-length")
                total = int(total) + have if total is not None else None
                t0, got = time.time(), have
                with open(part, mode) as f:
                    for b in r.iter_bytes(CHUNK):
                        f.write(b)
                        got += len(b)
                        if time.time() - t0 > 30:
                            log(f"    {dest.name}: {human(got)} / {human(total)}")
                            t0 = time.time()
                if total is not None and got != total:
                    raise RuntimeError(f"short read {got} != {total}")
            break
        except (httpx.HTTPError, RuntimeError, OSError) as e:
            last_err = e
            log(f"    retry {attempt + 1}/{retries} for {dest.name}: {e}")
            time.sleep(min(60, 5 * 2 ** attempt))
    else:
        raise RuntimeError(f"download failed: {url}: {last_err}")
    size = part.stat().st_size
    if expect_size and size != expect_size:
        raise RuntimeError(f"{dest.name}: size {size} != manifest {expect_size} (source changed? re-check and re-pin)")
    os.replace(part, dest)
    return size


# ------------------------------------------------------------------------------------------------ remote zip
class RemoteZip:
    """Read selected members of a zip over HTTP Range requests (ZIP64 aware), without downloading it whole."""

    def __init__(self, url: str, headers: Optional[dict] = None):
        self.url, self.c = url, _client()
        self.h = dict(headers or {})
        r = self.c.get(url, headers={**self.h, "Range": "bytes=0-0"})
        if r.status_code != 206:
            raise RuntimeError(f"{url}: server does not honour Range requests (HTTP {r.status_code})")
        self.size = int(r.headers["content-range"].split("/")[-1])
        self.entries = self._central_directory()

    def _get(self, a: int, b: int) -> bytes:
        r = self.c.get(self.url, headers={**self.h, "Range": f"bytes={a}-{b}"})
        if r.status_code != 206:
            raise RuntimeError(f"range {a}-{b}: HTTP {r.status_code}")
        return r.content

    def _central_directory(self) -> dict:
        tail_n = min(self.size, 65536 + 22)
        tail = self._get(self.size - tail_n, self.size - 1)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise RuntimeError("zip end-of-central-directory not found")
        _, _, _, _, n, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
        if cd_off == 0xFFFFFFFF or n == 0xFFFF or cd_size == 0xFFFFFFFF:  # ZIP64
            j = tail.rfind(b"PK\x06\x07", 0, i)
            (z64_off,) = struct.unpack("<Q", tail[j + 8:j + 16])
            rec = self._get(z64_off, z64_off + 55)
            n, cd_size, cd_off = struct.unpack("<QQQ", rec[32:56])
        cd = self._get(cd_off, cd_off + cd_size - 1)
        out, p = {}, 0
        while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            (method, _t, _d, _crc, csize, usize, nlen, xlen, clen, _dn, _ia, _ea,
             off) = struct.unpack("<HHHIIIHHHHHII", cd[p + 10:p + 46])
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
            q = 0
            while q + 4 <= len(extra):  # ZIP64 extended information
                tag, sz = struct.unpack("<HH", extra[q:q + 4])
                if tag == 1:
                    vals, k = extra[q + 4:q + 4 + sz], 0
                    if usize == 0xFFFFFFFF:
                        (usize,) = struct.unpack("<Q", vals[k:k + 8]); k += 8
                    if csize == 0xFFFFFFFF:
                        (csize,) = struct.unpack("<Q", vals[k:k + 8]); k += 8
                    if off == 0xFFFFFFFF:
                        (off,) = struct.unpack("<Q", vals[k:k + 8]); k += 8
                q += 4 + sz
            out[name] = {"method": method, "csize": csize, "usize": usize, "offset": off}
            p += 46 + nlen + xlen + clen
        return out

    def select(self, patterns: Iterable[str]) -> list[str]:
        pats = list(patterns)
        return sorted(n for n in self.entries if not n.endswith("/")
                      and any(n.startswith(pt) if pt.endswith("/") else fnmatch.fnmatch(n, pt) for pt in pats))

    def extract(self, names: list[str], dest: Path, strip: int = 0, max_gap: int = 1 << 20) -> int:
        """Extract members; contiguous members are fetched with one Range request per run."""
        ents = sorted(((self.entries[n]["offset"], n) for n in names))
        runs, cur = [], []
        for off, n in ents:
            e = self.entries[n]
            end = off + 30 + len(n.encode()) + 512 + e["csize"]  # generous: local extra field is usually small
            if cur and off - cur[-1][2] > max_gap:
                runs.append(cur); cur = []
            cur.append((off, n, end))
        if cur:
            runs.append(cur)
        total = 0
        for run in runs:
            a, b = run[0][0], min(self.size - 1, max(x[2] for x in run))
            blob = self._get(a, b)
            for off, n, _ in run:
                e, p = self.entries[n], off - a
                if blob[p:p + 4] != b"PK\x03\x04":
                    raise RuntimeError(f"bad local header for {n}")
                nlen, xlen = struct.unpack("<HH", blob[p + 26:p + 30])
                data = blob[p + 30 + nlen + xlen:p + 30 + nlen + xlen + e["csize"]]
                if len(data) != e["csize"]:  # local extra field larger than the guess: fetch this member alone
                    s = off + 30 + nlen + xlen
                    data = self._get(s, s + e["csize"] - 1)
                if e["method"] == 8:
                    data = zlib.decompressobj(-15).decompress(data)
                elif e["method"] != 0:
                    raise RuntimeError(f"{n}: unsupported compression method {e['method']}")
                rel = Path(*Path(n).parts[strip:])
                out = dest / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_name(out.name + ".part")
                tmp.write_bytes(data)
                os.replace(tmp, out)
                total += len(data)
        return total


# ------------------------------------------------------------------------------------------------ archives
def safe_extract(archive: Path, dest: Path, members: Optional[list[str]] = None, strip: int = 0) -> int:
    """Extract a .zip / .tar(.gz/.bz2) without path traversal. Returns bytes written."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    n = 0

    def target(name: str) -> Optional[Path]:
        parts = Path(name).parts[strip:]
        if not parts:
            return None
        t = (dest / Path(*parts)).resolve()
        if root not in t.parents and t != root:
            raise RuntimeError(f"unsafe path in archive: {name}")
        return t

    def wanted(name: str) -> bool:
        return not members or any(name.startswith(m) if m.endswith("/") else fnmatch.fnmatch(name, m) for m in members)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir() or not wanted(info.filename):
                    continue
                t = target(info.filename)
                if t is None or (t.exists() and t.stat().st_size == info.file_size):
                    continue
                t.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(t, "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
                n += info.file_size
        return n
    with tarfile.open(archive) as tf:
        for m in tf:
            if not m.isfile() or not wanted(m.name):
                continue
            t = target(m.name)
            if t is None or (t.exists() and t.stat().st_size == m.size):
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(t, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            n += m.size
    return n


# ------------------------------------------------------------------------------------------------ fetch
def fetch_dataset(ds: dict, root: Path, state: State, pin: bool, dirty: list) -> None:
    for i, spec in enumerate(ds.get("fetch", [])):
        if spec.get("zip_url"):
            _fetch_zip_members(ds, spec, root, state)
            continue
        dest = root / spec["dest"]
        key = spec["dest"]
        rec = state.d["files"].get(key)
        want = spec.get("sha256")
        if dest.exists() and rec and rec.get("size") == dest.stat().st_size and (not want or rec.get("sha256") == want):
            log(f"  ok      {key} ({human(rec['size'])})")
        else:
            log(f"  fetch   {key}  <-  {spec_url(spec)}")
            download(spec_url(spec), dest, _auth_headers(spec), spec.get("size"))
            digest = sha256_file(dest)
            if want and digest != want:
                bad = dest.with_name(dest.name + ".bad")
                os.replace(dest, bad)
                raise RuntimeError(f"{key}: sha256 {digest} != manifest {want} (kept as {bad.name})")
            rec = state.d["files"][key] = {"size": dest.stat().st_size, "sha256": digest,
                                           "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": spec_url(spec)}
            state.save()
        if pin and not want:
            spec["sha256"], spec["size"] = rec["sha256"], rec["size"]
            dirty.append(key)
        if spec.get("extract_to"):
            out = root / spec["extract_to"]
            stamp = out / ".extracted"
            if not (stamp.exists() and stamp.read_text().strip() == rec["sha256"]):
                log(f"  extract {key} -> {spec['extract_to']}")
                n = safe_extract(dest, out, spec.get("members"), int(spec.get("strip", 0)))
                stamp.write_text(rec["sha256"])
                log(f"          {human(n)} written")
            if spec.get("delete_archive_after_extract"):
                dest.unlink(missing_ok=True)


def _fetch_zip_members(ds: dict, spec: dict, root: Path, state: State) -> None:
    out = root / spec["dest_dir"]
    stamp = out / f".members_{hashlib.sha1(json.dumps(spec['members']).encode()).hexdigest()[:10]}"
    if stamp.exists():
        log(f"  ok      {spec['dest_dir']} ({len(spec['members'])} member patterns from {spec['zip_url']})")
        return
    log(f"  zip     {spec['zip_url']} (remote members {spec['members']})")
    rz = RemoteZip(spec["zip_url"], _auth_headers(spec))
    names = rz.select(spec["members"])
    if not names:
        raise RuntimeError(f"no members matched {spec['members']} in {spec['zip_url']}")
    todo = [n for n in names if not (out / Path(*Path(n).parts[int(spec.get('strip', 0)):])).exists()]
    log(f"          {len(names)} members ({human(sum(rz.entries[n]['usize'] for n in names))}), {len(todo)} to fetch; "
        f"zip is {human(rz.size)}")
    if todo:
        rz.extract(todo, out, int(spec.get("strip", 0)))
    crcs = {n: rz.entries[n]["usize"] for n in names}
    bad = [n for n, sz in crcs.items() if (out / Path(*Path(n).parts[int(spec.get('strip', 0)):])).stat().st_size != sz]
    if bad:
        raise RuntimeError(f"size mismatch for {len(bad)} members, e.g. {bad[:3]}")
    stamp.write_text(json.dumps({"zip": spec["zip_url"], "zip_size": rz.size, "members": len(names)}))
    state.d["files"][spec["dest_dir"] + "/"] = {"size": sum(crcs.values()), "members": len(names),
                                                 "zip_size": rz.size, "url": spec["zip_url"],
                                                 "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    state.save()


# ------------------------------------------------------------------------------------------------ convert
def ffmpeg_exe(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("no ffmpeg: install imageio-ffmpeg (a project dependency) or pass --ffmpeg")
        return exe


def x264(crf: int) -> list[str]:
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-profile:v", "main", "-bf", "0",
            "-force_key_frames", "expr:gte(t,n_forced*2)", "-movflags", "+faststart"]


def video_filter(variant: str, max_width: Optional[int]) -> str:
    """Even dimensions, optional downscale to max_width. Variant "gray" mimics a black-and-white / IR-night CCTV
    stream: luma only, encoded as ordinary yuv420p H.264 with neutral chroma (what such cameras actually send), so
    decoders still return 3 equal channels. Colour information is really gone, which is the point of the comparison."""
    vf = (f"scale='trunc(min(iw,{int(max_width)})/2)*2':-2" if max_width else "scale=trunc(iw/2)*2:trunc(ih/2)*2")
    if variant == "gray":
        vf += ",format=gray"
    return vf + ",format=yuv420p"


def probe_video(p: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(p))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(fps, 3), "frames": n, "duration_s": round(n / fps, 2) if fps else None,
                "bytes": p.stat().st_size}
    finally:
        cap.release()


def _conversions(ds: dict, root: Path) -> list[dict]:
    """Expand the manifest's convert entries into jobs {inp, dst, stamp, variant, max_width, crf, start_s, max_s}.

    Entry keys: src | src_glob (video files) or frames_dir_glob + pattern + fps (image sequences, one mp4 per
    directory; name_template may use {dir} {parent} {grandparent}); dst | dst_dir; optional max_width, crf (23),
    start_s / max_s (trim), variants (e.g. ["gray"]) and variant_only (stems that get the variants)."""
    jobs = []
    for c in ds.get("convert", []):
        opts = {"max_width": c.get("max_width"), "crf": int(c.get("crf", 23)), "max_s": c.get("max_s"),
                "start_s": c.get("start_s")}
        base = []
        if "src" in c:
            srcs = [root / c["src"]]
        elif "src_glob" in c:
            srcs = [Path(q) for q in sorted(glob.glob(str(root / c["src_glob"])))]
        else:
            srcs = []
        for src in srcs:
            if src.is_file():
                dst = root / c["dst"] if "dst" in c else root / c["dst_dir"] / (src.stem + ".mp4")
                st = src.stat()
                base.append({"inp": ["-i", str(src)], "dst": dst,
                             "stamp": f"{src.name}:{st.st_size}:{int(st.st_mtime)}"})
        if "frames_dir_glob" in c:
            for d in sorted(glob.glob(str(root / c["frames_dir_glob"]))):
                d = Path(d)
                frames = sorted(d.glob(c.get("frames_ext", "*.jpg")))
                if not frames:
                    continue
                name = c.get("name_template", "{dir}").format(dir=d.name, parent=d.parent.name,
                                                               grandparent=d.parent.parent.name)
                inp = ["-framerate", str(c["fps"]), "-start_number", str(c.get("start_number", 1)),
                       "-i", str(d / c["pattern"])]
                base.append({"inp": inp, "dst": root / c["dst_dir"] / f"{name}.mp4",
                             "stamp": f"{d.name}:{len(frames)}:{sum(f.stat().st_size for f in frames[:50])}"})
        only = set(c.get("variant_only") or [])
        for j in base:
            jobs.append({**j, **opts, "variant": ""})
            for v in c.get("variants", []):
                if not only or j["dst"].stem in only:
                    jobs.append({**j, **opts, "variant": v, "dst": j["dst"].with_name(f"{j['dst'].stem}_{v}.mp4")})
    return jobs


def convert_dataset(ds: dict, root: Path, state: State, ff: str) -> None:
    for j in _conversions(ds, root):
        dst = j["dst"]
        key = dst.relative_to(root).as_posix()
        stamp = f"{j['stamp']}|{j['variant']}|{j['max_width']}|{j['crf']}|{j['start_s']}|{j['max_s']}"
        rec = state.d["converted"].get(key)
        if dst.exists() and rec and rec.get("stamp") == stamp:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + ".tmp.mp4")
        trim = (["-ss", str(j["start_s"])] if j["start_s"] else []) + (["-t", str(j["max_s"])] if j["max_s"] else [])
        cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", *j["inp"], *trim, "-an",
               "-vf", video_filter(j["variant"], j["max_width"]), *x264(j["crf"]), str(tmp)]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not tmp.exists():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed for {key}: {r.stderr.strip()[-500:]}")
        os.replace(tmp, dst)
        info = probe_video(dst)
        state.d["converted"][key] = {"stamp": stamp, "variant": j["variant"] or "original", **info,
                                     "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        state.save()
        log(f"  mp4     {key}  {info['width']}x{info['height']} @ {info['fps']} fps, {info['duration_s']} s, "
            f"{human(info['bytes'])}  ({time.time() - t0:.1f} s)")


# ------------------------------------------------------------------------------------------------ commands
def present(ds: dict, root: Path) -> dict:
    d = root / ds["id"]
    out = {"bytes": dir_size(d), "mp4": sorted(str(p.relative_to(root)).replace("\\", "/")
                                                 for p in (d / "mp4").rglob("*.mp4")) if (d / "mp4").exists() else []}
    for alt in ds.get("reuse_if_present", []):
        hits = [p for p in glob.glob(str(root / alt)) if Path(p).exists()]
        if hits:
            out["reused"] = [str(Path(h).relative_to(root)) for h in hits]
            out["bytes"] += sum(dir_size(Path(h)) if Path(h).is_dir() else Path(h).stat().st_size for h in hits)
    return out


def cmd_list(m: dict, root: Path, sel: list[dict]) -> None:
    total = 0
    for ds in sel:
        p = present(ds, root)
        total += p["bytes"]
        log(f"{ds['id']:<12} {human(ds.get('size_bytes')):>9} manifest | {human(p['bytes']):>9} on disk | "
            f"{len(p['mp4']):>3} mp4 | {ds['licence_short']}")
        if p.get("reused"):
            log(f"{'':<12} reusing {p['reused']}")
    log(f"total on disk (selected): {human(total)} ; budget {m.get('budget_gb')} GB ; root {root}")


def selected(m: dict, only: str, include_optional: bool) -> list[dict]:
    ids = [s.strip() for s in only.split(",") if s.strip()] if only else None
    out = []
    for ds in m["datasets"]:
        if ids is not None:
            if ds["id"] in ids:
                out.append(ds)
        elif ds.get("default", True) or include_optional:
            out.append(ds)
    if ids:
        unknown = set(ids) - {d["id"] for d in out}
        if unknown:
            raise SystemExit(f"unknown dataset id(s): {sorted(unknown)}")
    return out


def cmd_subset(m: dict, root: Path, sel: list[dict], dest: Path, max_mb: float) -> None:
    files = [(ds["id"], root / f) for ds in sel for f in ds.get("laptop_subset", [])]
    missing = [str(f) for _, f in files if not f.exists()]
    if missing:
        raise SystemExit(f"subset files missing (fetch/convert first): {missing[:5]}")
    total = sum(f.stat().st_size for _, f in files)
    if total > max_mb * 2**20:
        raise SystemExit(f"subset is {human(total)} > {max_mb} MB")
    for _, f in files:
        rel = f.relative_to(root)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not (out.exists() and out.stat().st_size == f.stat().st_size):
            shutil.copy2(f, out)
    log(f"copied {len(files)} files ({human(total)}) to {dest}")


def ship_files(root: Path, sel: list[dict]) -> list[Path]:
    """The curated evaluation set (manifest `ship` globs): converted clips + labels, for a remote GPU host."""
    out = []
    for ds in sel:
        for pat in ds.get("ship", []):
            out += [Path(p) for p in sorted(glob.glob(str(root / pat), recursive=True))
                    if Path(p).is_file() and not p.endswith((".part", ".tmp.mp4"))]
    return sorted(set(out))


def cmd_ship(root: Path, sel: list[dict], tar_path: Optional[str], max_mb: float) -> None:
    files = ship_files(root, sel)
    total = sum(f.stat().st_size for f in files)
    per = defaultdict(int)
    for f in files:
        per[f.relative_to(root).parts[0]] += f.stat().st_size
    log(f"ship set: {len(files)} files, {human(total)} " + ", ".join(f"{k} {human(v)}" for k, v in sorted(per.items())),
        file=sys.stderr)
    if total > max_mb * 2**20:
        raise SystemExit(f"ship set is {human(total)} > --max-mb {max_mb}; narrow --only or the manifest ship globs")
    if not tar_path:
        for f in files:
            print(f.relative_to(root).as_posix())
        return
    out = sys.stdout.buffer if tar_path == "-" else open(tar_path, "wb")
    with tarfile.open(fileobj=out, mode="w|") as tf:  # uncompressed stream: the mp4s are already compressed
        for f in files:
            tf.add(f, arcname=f.relative_to(root).as_posix())
    if out is not sys.stdout.buffer:
        out.close()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["list", "fetch", "convert", "all", "verify", "subset", "ship"])
    ap.add_argument("--tar", default=None, help="ship: write a tar of the ship set to this file ('-' = stdout)")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--only", default="", help="comma-separated dataset ids (default: every default dataset)")
    ap.add_argument("--optional", action="store_true", help="include datasets marked default: false")
    ap.add_argument("--pin", action="store_true", help="write missing sha256/size into the manifest")
    ap.add_argument("--force", action="store_true", help="ignore the size budget")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--dest", type=Path, default=ROOT / "data" / "real" / "datasets")
    ap.add_argument("--max-mb", type=float, default=300)
    a = ap.parse_args(argv)
    m = load_manifest(a.manifest)
    root = (a.root or default_root()).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    sel = selected(m, a.only, a.optional)
    state = State(root)
    if a.cmd == "list":
        cmd_list(m, root, sel)
        return 0
    if a.cmd == "subset":
        cmd_subset(m, root, sel, a.dest, a.max_mb)
        return 0
    if a.cmd == "ship":
        cmd_ship(root, sel, a.tar, a.max_mb if a.max_mb != 300 else float(m.get("ship_budget_mb", 400)))
        return 0
    if a.cmd == "verify":
        bad = 0
        for ds in sel:
            for spec in ds.get("fetch", []):
                if spec.get("zip_url") or not spec.get("dest"):
                    continue
                p = root / spec["dest"]
                if not p.exists():
                    log(f"  missing {spec['dest']}"); bad += 1; continue
                d = sha256_file(p)
                ok = not spec.get("sha256") or d == spec["sha256"]
                bad += not ok
                log(f"  {'ok ' if ok else 'BAD'}     {spec['dest']} {d[:16]}")
        return 1 if bad else 0
    planned = sum(ds.get("size_bytes", 0) for ds in sel)
    budget = float(m.get("budget_gb", 25)) * 2**30
    if planned > budget and not a.force:
        raise SystemExit(f"selected datasets need ~{human(planned)} > budget {m.get('budget_gb')} GB (use --force)")
    ff = ffmpeg_exe(a.ffmpeg) if a.cmd in ("convert", "all") else None
    dirty: list[str] = []
    failures = []
    for ds in sel:
        log(f"[{ds['id']}] {ds['name']}")
        p = present(ds, root)
        if p.get("reused") and not ds.get("fetch"):
            log(f"  reuse   {p['reused']} (already in the shared root; not downloading a second copy)")
        try:
            with Lock(root / ds["id"] / ".lock"):
                if a.cmd in ("fetch", "all"):
                    fetch_dataset(ds, root, state, a.pin, dirty)
                if a.cmd in ("convert", "all"):
                    convert_dataset(ds, root, state, ff)
        except Exception as e:  # noqa: BLE001 - one dead source must not stop the others
            failures.append((ds["id"], str(e)))
            log(f"  FAILED  {ds['id']}: {e}")
    if dirty:
        save_manifest(m, a.manifest)
        log(f"pinned sha256 for {len(dirty)} file(s) in {a.manifest}")
    for i, e in failures:
        log(f"FAILED {i}: {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
