# CAVIAR rule precision / recall

- Measured 2026-09-26 on NVIDIA GeForce RTX 4050 Laptop GPU (host `LEGION`, Windows-10-10.0.26200-SP0)
- Detector `yolo26s.pt` imgsz 960 conf 0.1 + ByteTrack, analytics 5 fps, device cuda:0
- Space: **image_px** (CAVIAR publishes no ground-plane calibration: zones and thresholds are in image pixels; running uses body-heights/s)
- Rules: `{"loitering": {"min_seconds": 20}, "crowding": {"max_count": 1, "min_seconds": 3}, "running": {"speed_bh_s": 1.5, "min_seconds": 1.0}}`; altercation analyzer off
- Matching: an event is a true positive when its interval overlaps a ground-truth episode (+-tolerance) of the target behaviour in the same clip AND, in a shared frame of that overlap, the centre of one of the event's track boxes is within proximity_px of the episode's box (group box or a member's box). Many-to-one both ways: precision = true events / events, recall = episodes hit / episodes. Tolerance 2.0 s, proximity 30.0 px.
- Ground-truth episodes: CVML labels: running = object movement 'running' >= 1 s; fighting = group context/situation 'fighting' >= 1 s; meeting = group context 'meeting' >= 3 s; browsing_or_inactive_20s = object situation browsing/inactive or movement inactive >= 20 s; left_object = role 'leaving object'. Runs merged over <= 0.5 s gaps.
- Dataset: CAVIAR first set (EC Funded CAVIAR project/IST 2001 37540), clips: Browse1, Browse2, Browse3, Browse4, Browse_WhileWaiting1, Browse_WhileWaiting2, Fight_Chase, Fight_OneManDown, Fight_RunAway1, Fight_RunAway2, LeftBag, LeftBag_AtChair, LeftBag_BehindChair, LeftBag_PickedUp, LeftBox, Meet_Crowd, Meet_Split_3rdGuy, Meet_WalkSplit, Meet_WalkTogether1, Meet_WalkTogether2, Rest_FallOnFloor, Rest_InChair, Rest_SlumpOnFloor, Rest_WiggleOnFloor, Walk1, Walk2, Walk3

## colour

| Ground truth | Argus rule | Role | GT episodes | Events | Precision (Wilson 95 %) | Recall (Wilson 95 %) |
|---|---|---|---|---|---|---|
| browsing_or_inactive_20s | loitering | primary: dwell >= 20 s | 4 | 22 | 0.04 (0.01-0.22) | 0.25 (0.05-0.70) |
| running | running | primary | 8 | 5 | 0.00 (0.00-0.43) | 0.00 (0.00-0.32) |
| meeting | crowding | proxy: > 1 person in view is not a meeting | 10 | 25 | 0.16 (0.06-0.35) | 0.40 (0.17-0.69) |
| fighting | altercation (not run) | primary (only with --threats) | 4 | 0 | n/a | 0.00 (0.00-0.49) |
| fighting | running | secondary | 4 | 5 | 0.00 (0.00-0.43) | 0.00 (0.00-0.49) |
| left_object | - | no Argus rule | 4 | 0 | n/a | 0.00 (0.00-0.49) |

27 clips, 1040.9 s, 98 tracks (>= 10 samples), 1.28 person detections / frame, detect 674.1 ms median, wall 161.2 s. Events by rule: {"crowding": 25, "loitering": 22, "running": 5}

