"""Multi-frame clip explanation and the /chat client (request shapes only; no network, no model)."""
import cv2
import httpx
import numpy as np

from argonix.vlm import EXPLAIN_MAX_TOKENS, ChatClient, RemoteExplainer, sample_clip_frames, trim_to_sentence


def write_clip(path, n=20, fps=5):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for i in range(n):
        f = np.zeros((120, 160, 3), np.uint8)
        cv2.rectangle(f, (5 + i * 6, 40), (25 + i * 6, 90), (255, 255, 255), -1)
        vw.write(f)
    vw.release()


def test_sample_clip_frames_even_spread(tmp_path):
    write_clip(tmp_path / "c.mp4")
    frames, dur = sample_clip_frames(tmp_path / "c.mp4", 4)
    assert len(frames) == 4 and all(b[:2] == b"\xff\xd8" for b in frames) and 3.0 <= dur <= 5.0
    assert sample_clip_frames(tmp_path / "missing.mp4", 4) == ([], 0.0)


def test_explainer_sends_clip_frames_when_server_supports_it(tmp_path, monkeypatch):
    (tmp_path / "evidence").mkdir()
    write_clip(tmp_path / "evidence" / "c.mp4")
    (tmp_path / "evidence" / "s.jpg").write_bytes(cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes())
    alert = {"events": [{"observation": "person track #1", "evidence": {"snapshot": "evidence/s.jpg",
                                                                        "clip": "evidence/c.mp4"}}]}
    seen = {}

    def fake_get(url, timeout):
        return httpx.Response(200, json={"features": ["explain", "multi_image", "chat"]}, request=httpx.Request("GET", url))

    def fake_post(url, json, timeout):
        seen.update(json)
        return httpx.Response(200, json={"text": "Two frames show one person walking right.", "model": "m",
                                         "latency_s": 1.0, "n_images": len(json.get("images_b64") or [1])},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)
    out = RemoteExplainer("http://x:8100")(alert, tmp_path)
    assert len(seen["images_b64"]) == 4 and "frames in time order" in seen["prompt"] and seen["image_b64"]
    assert out.endswith("4 frames from the evidence clip]")
    # an old server (no features) gets the single snapshot and the original prompt
    seen.clear()
    monkeypatch.setattr(httpx, "get", lambda url, timeout: httpx.Response(200, json={}, request=httpx.Request("GET", url)))
    out = RemoteExplainer("http://x:8100")(alert, tmp_path)
    assert "images_b64" not in seen and "people" in seen["prompt"] and out.endswith("evidence snapshot]")
    assert seen["max_new_tokens"] == EXPLAIN_MAX_TOKENS


def test_trim_to_sentence():
    assert trim_to_sentence("One person walks. Then the person on the") == ("One person walks.", True)
    assert trim_to_sentence("One person walks left.") == ("One person walks left.", False)
    assert trim_to_sentence("no sentence end at all") == ("no sentence end at all", False)


def test_explainer_drops_unfinished_sentence_only_at_token_limit(tmp_path, monkeypatch):
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / "s.jpg").write_bytes(cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes())
    alert = {"events": [{"observation": "person track #1", "evidence": {"snapshot": "evidence/s.jpg"}}]}
    reply = {"text": "Two people stand on the lawn. One of them walks to the", "model": "m", "latency_s": 1.0, "n_images": 1}
    monkeypatch.setattr(httpx, "get", lambda url, timeout: httpx.Response(200, json={}, request=httpx.Request("GET", url)))
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: httpx.Response(
        200, json={**reply, "completion_tokens": EXPLAIN_MAX_TOKENS}, request=httpx.Request("POST", url)))
    out = RemoteExplainer("http://x:8100")(alert, tmp_path)
    assert out.startswith("Two people stand on the lawn. [") and "dropped at the token limit" in out
    # below the limit the text is passed through untouched
    monkeypatch.setattr(httpx, "post", lambda url, json, timeout: httpx.Response(
        200, json={**reply, "completion_tokens": 20}, request=httpx.Request("POST", url)))
    assert RemoteExplainer("http://x:8100")(alert, tmp_path).startswith(reply["text"] + " [")


def test_chat_client_request_shape(monkeypatch):
    seen = {}

    def fake_post(url, json, timeout):
        seen["url"], seen["body"] = url, json
        return httpx.Response(200, json={"text": "{}", "model": "Qwen/Qwen2.5-VL-3B-Instruct", "latency_s": 0.5},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", fake_post)
    r = ChatClient("http://127.0.0.1:8100/explain").chat([{"role": "user", "content": "hi"}], max_new_tokens=50)
    assert seen["url"] == "http://127.0.0.1:8100/chat" and seen["body"]["max_new_tokens"] == 50 and r["text"] == "{}"
