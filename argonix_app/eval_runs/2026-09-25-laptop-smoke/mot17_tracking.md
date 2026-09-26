# MOT17 tracking accuracy - SMOKE TEST, NOT A RESULT

> **Smoke test, not a result.** Truncated sequence(s) on a CPU laptop, only to prove the script works end to end. Do not quote these numbers.

- Measured 2026-09-25 on AMD64 Family 23 Model 160 Stepping 0, AuthenticAMD (host `ByteForge`, Windows-11-10.0.26100-SP0)
- Detector `yolo26n.pt` imgsz 480 conf 0.3, device cpu
- gt: MOT17 train gt.txt: class 1 (pedestrian) rows with the consider flag set; tracker boxes matched (Hungarian, IoU >= 0.5) to distractor classes 2/7/8/12 (person on vehicle, static person, distractor, reflection) are removed before scoring (TrackEval MotChallenge2DBox preprocessing); no visibility threshold (as the official evaluation)
- metrics: CLEAR-MOT, Identity and HOTA as TrackEval (tools/eval_metrics.py): IoU >= 0.5 for MOTA/IDF1; MOTP = mean IoU of matches; HOTA averaged over alpha 0.05..0.95; combined = pooled counts
- tracker: the app's production path: frames downscaled to analytics_width 960, detector (conf 0.3, person class) + Ultralytics ByteTrack (bytetrack.yaml, track_buffer counted in processed frames), person tracks only; no public detections are used (private-detector setting)
- settings: {"5fps": "analytics rate: every round(fps/5)-th frame is processed and ONLY those frames are scored (GT subsampled the same way)", "full": "every frame processed and scored"}
- video: img1 JPEG frames re-encoded to H.264 CRF 18 (one frame per image) for the file-source pipeline
- Dataset: MOT17 train sequences (MOTChallenge, Milan et al. 2016, arXiv 1603.00831; research use)

| Sequence | Setting | Frames scored | HOTA | DetA | AssA | MOTA | IDF1 | MOTP | IDSW | FP | FN | det_recall | det_precision | Detect ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT17-09-FRCNN | 5fps | 25 | 0.5204 | 0.4454 | 0.6108 | 0.4706 | 0.6578 | 0.8183 | 2 | 12 | 85 | 0.5455 | 0.8947 | 3235.6 |
| MOT17-09-FRCNN | full | 150 | 0.5648 | 0.4984 | 0.6419 | 0.491 | 0.6799 | 0.8538 | 2 | 107 | 456 | 0.5892 | 0.8594 | 3533.0 |
| **combined (1 seq)** | 5fps | - | 0.5204 | 0.4454 | 0.6108 | 0.4706 | 0.6578 | 0.8183 | 2 | 12 | 85 | 0.5455 | 0.8947 | - |
| **combined (1 seq)** | full | - | 0.5648 | 0.4984 | 0.6419 | 0.491 | 0.6799 | 0.8538 | 2 | 107 | 456 | 0.5892 | 0.8594 | - |
