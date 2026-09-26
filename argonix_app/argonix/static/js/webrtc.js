// WebRTC (WHEP) live player for the "Smooth" view mode. The SDP offer goes to Argus
// (POST /api/cameras/{cam}/whep, session cookie + CSRF header), which forwards it to the MediaMTX restreamer; media
// then flows peer-to-peer over ICE (LAN). Full camera frame rate, the camera's own H.264, no analytics overlay.
// Non-trickle: the offer is sent once ICE gathering finishes (host candidates only, so this is quick on a LAN).
import { enc } from "./util.js";

const GATHER_MS = 2000;       // cap on ICE gathering before the offer is sent anyway
const FIRST_FRAME_MS = 10000; // no decoded frame by then -> failed (e.g. camera offline, H.265 in a browser without HEVC)
const DISCONNECT_MS = 4000;   // "disconnected" this long -> failed

export const webrtcSupported = () => typeof RTCPeerConnection === "function";

function gathered(pc) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise(res => {
    const t = setTimeout(done, GATHER_MS);
    function done() { clearTimeout(t); pc.removeEventListener("icegatheringstatechange", chk); res(); }
    function chk() { if (pc.iceGatheringState === "complete") done(); }
    pc.addEventListener("icegatheringstatechange", chk);
  });
}

async function detail(r) {
  const t = await r.text().catch(() => "");
  try { const j = JSON.parse(t); return typeof j.detail === "string" ? j.detail : t; } catch { return t.slice(0, 200); }
}

/**
 * new WhepPlayer(camId, videoEl, { onPlaying(), onFail(reason) })
 * start() resolves when the session is negotiated; onPlaying fires on the first decoded frame; onFail fires once for
 * any failure (negotiation, no frames, connection lost). stop() tears the session down (DELETE through the proxy).
 */
export class WhepPlayer {
  constructor(cam, video, { onPlaying = () => {}, onFail = () => {} } = {}) {
    this.cam = cam; this.video = video; this.onPlaying = onPlaying; this.onFail = onFail;
    this.pc = null; this.location = null; this.timers = []; this.closed = false; this.playing = false;
  }

  fail(reason) {
    if (this.closed) return;
    const cb = this.onFail;
    this.stop();
    cb(reason);
  }

  async start() {
    if (!webrtcSupported()) { this.fail("this browser has no WebRTC"); return; }
    const pc = this.pc = new RTCPeerConnection({ iceServers: [] });
    pc.addTransceiver("video", { direction: "recvonly" });
    pc.addTransceiver("audio", { direction: "recvonly" });  // MediaMTX's answer expects one; the <video> stays muted
    pc.ontrack = e => {
      if (e.track.kind !== "video") return;
      this.video.srcObject = e.streams[0] || new MediaStream([e.track]);
      this.video.play().catch(() => {});
    };
    pc.onconnectionstatechange = () => {
      const st = pc.connectionState;
      if (st === "failed") this.fail("network path to the restreamer failed (UDP 8189 blocked?)");
      else if (st === "disconnected") this.timers.push(setTimeout(() => { if (pc.connectionState === "disconnected") this.fail("connection lost"); }, DISCONNECT_MS));
    };
    const onFrame = () => {
      if (this.playing || this.closed) return;
      this.playing = true; this.onPlaying();
    };
    this.video.addEventListener("playing", onFrame, { once: true });
    this.timers.push(setTimeout(() => { if (!this.playing) this.fail("no video received (camera offline, or a codec this browser cannot play)"); }, FIRST_FRAME_MS));
    try {
      await pc.setLocalDescription(await pc.createOffer());
      await gathered(pc);
      if (this.closed) return;
      const r = await fetch(`/api/cameras/${enc(this.cam)}/whep`, {
        method: "POST", credentials: "same-origin", body: pc.localDescription.sdp,
        headers: { "Content-Type": "application/sdp", "X-Argonix-CSRF": "1" },
      });
      if (r.status !== 201) { this.fail(`${r.status}: ${await detail(r) || r.statusText}`); return; }
      this.location = r.headers.get("Location");
      const answer = await r.text();
      if (this.closed) { this.teardown(); return; }
      await pc.setRemoteDescription({ type: "answer", sdp: answer });
    } catch (e) {
      this.fail(String(e?.message || e));
    }
  }

  teardown() {
    const loc = this.location;
    this.location = null;
    // keepalive: the DELETE still goes out when the page is being left
    if (loc && loc.startsWith("/api/")) fetch(loc, { method: "DELETE", credentials: "same-origin", keepalive: true, headers: { "X-Argonix-CSRF": "1" } }).catch(() => {});
  }

  stop() {
    if (this.closed) return;
    this.closed = true;
    this.timers.forEach(clearTimeout); this.timers = [];
    try { this.pc?.close(); } catch { /* already closed */ }
    this.pc = null;
    this.teardown();
    if (this.video.srcObject) { this.video.srcObject = null; }
  }
}
