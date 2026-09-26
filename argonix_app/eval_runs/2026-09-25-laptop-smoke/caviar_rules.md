# CAVIAR rule precision / recall - SMOKE TEST, NOT A RESULT

> **Smoke test, not a result.** Tiny sample run on a CPU laptop only to prove the script works end to end. Do not quote these numbers.

- Measured 2026-09-25 on AMD64 Family 23 Model 160 Stepping 0, AuthenticAMD (host `ByteForge`, Windows-11-10.0.26100-SP0)
- Detector `yolo26n.pt` imgsz 480 conf 0.3 + ByteTrack, analytics 5 fps, device cpu
- Space: **image_px** (CAVIAR publishes no ground-plane calibration: zones and thresholds are in image pixels; running uses body-heights/s)
- Rules: `{"loitering": {"min_seconds": 20}, "crowding": {"max_count": 1, "min_seconds": 3}, "running": {"speed_bh_s": 1.5, "min_seconds": 1.0}}`; altercation analyzer off
- Matching: an event is a true positive when its interval overlaps a ground-truth episode (+-tolerance) of the target behaviour in the same clip AND, in a shared frame of that overlap, the centre of one of the event's track boxes is within proximity_px of the episode's box (group box or a member's box). Many-to-one both ways: precision = true events / events, recall = episodes hit / episodes. Tolerance 2.0 s, proximity 30.0 px.
- Ground-truth episodes: CVML labels: running = object movement 'running' >= 1 s; fighting = group context/situation 'fighting' >= 1 s; meeting = group context 'meeting' >= 3 s; browsing_or_inactive_20s = object situation browsing/inactive or movement inactive >= 20 s; left_object = role 'leaving object'. Runs merged over <= 0.5 s gaps.
- Dataset: CAVIAR first set (EC Funded CAVIAR project/IST 2001 37540), clips: Fight_RunAway1

## colour

| Ground truth | Argus rule | Role | GT episodes | Events | Precision (Wilson 95 %) | Recall (Wilson 95 %) |
|---|---|---|---|---|---|---|
| browsing_or_inactive_20s | loitering | primary: dwell >= 20 s | 0 | 0 | n/a | n/a |
| running | running | primary | 2 | 0 | n/a | 0.00 (0.00-0.66) |
| meeting | crowding | proxy: > 1 person in view is not a meeting | 0 | 0 | n/a | n/a |
| fighting | altercation (not run) | primary (only with --threats) | 1 | 0 | n/a | 0.00 (0.00-0.79) |
| fighting | running | secondary | 1 | 0 | n/a | 0.00 (0.00-0.79) |
| left_object | - | no Argus rule | 0 | 0 | n/a | n/a |

1 clips, 22.3 s, 0 tracks (>= 10 samples), 0.05 person detections / frame, detect 12120.2 ms median, wall 1076.7 s. Events by rule: {}

