# MOT17 tracking accuracy

- Measured 2026-09-26 on NVIDIA GeForce RTX 4050 Laptop GPU (host `LEGION`, Windows-10-10.0.26200-SP0)
- Detector `yolo26s.pt` imgsz 960 conf 0.1, device cuda:0
- gt: MOT17 train gt.txt: class 1 (pedestrian) rows with the consider flag set; tracker boxes matched (Hungarian, IoU >= 0.5) to distractor classes 2/7/8/12 (person on vehicle, static person, distractor, reflection) are removed before scoring (TrackEval MotChallenge2DBox preprocessing); no visibility threshold (as the official evaluation)
- metrics: CLEAR-MOT, Identity and HOTA as TrackEval (tools/eval_metrics.py): IoU >= 0.5 for MOTA/IDF1; MOTP = mean IoU of matches; HOTA averaged over alpha 0.05..0.95; combined = pooled counts
- tracker: the app's production path: frames downscaled to analytics_width 960, detector (conf 0.1, person class) + Ultralytics ByteTrack (bytetrack.yaml, track_buffer counted in processed frames), person tracks only; no public detections are used (private-detector setting)
- settings: {"5fps": "analytics rate: every round(fps/5)-th frame is processed and ONLY those frames are scored (GT subsampled the same way)", "full": "every frame processed and scored"}
- video: img1 JPEG frames re-encoded to H.264 CRF 18 (one frame per image) for the file-source pipeline
- Dataset: MOT17 train sequences (MOTChallenge, Milan et al. 2016, arXiv 1603.00831; research use)

| Sequence | Setting | Frames scored | HOTA | DetA | AssA | MOTA | IDF1 | MOTP | IDSW | FP | FN | det_recall | det_precision | Detect ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT17-02-FRCNN | 5fps | 100 | 0.3017 | 0.2639 | 0.3472 | 0.2774 | 0.3822 | 0.7908 | 45 | 164 | 2031 | 0.3448 | 0.867 | 61.9 |
| MOT17-04-FRCNN | 5fps | 175 | 0.4563 | 0.3992 | 0.5242 | 0.4292 | 0.5487 | 0.8254 | 62 | 362 | 4101 | 0.4827 | 0.9136 | 64.7 |
| MOT17-05-FRCNN | 5fps | 279 | 0.3489 | 0.396 | 0.3099 | 0.3828 | 0.4525 | 0.7197 | 173 | 295 | 959 | 0.5852 | 0.821 | 60.8 |
| MOT17-09-FRCNN | 5fps | 87 | 0.4196 | 0.4702 | 0.38 | 0.5079 | 0.5712 | 0.7713 | 69 | 78 | 288 | 0.6742 | 0.8843 | 59.7 |
| MOT17-10-FRCNN | 5fps | 109 | 0.1956 | 0.2972 | 0.1319 | 0.2554 | 0.2337 | 0.7378 | 223 | 94 | 1272 | 0.4039 | 0.9017 | 70.0 |
| MOT17-11-FRCNN | 5fps | 150 | 0.4157 | 0.4485 | 0.3938 | 0.4768 | 0.4987 | 0.7786 | 75 | 118 | 629 | 0.5996 | 0.8887 | 64.0 |
| MOT17-13-FRCNN | 5fps | 150 | 0.1626 | 0.1608 | 0.1687 | 0.1183 | 0.187 | 0.7465 | 187 | 35 | 1827 | 0.2139 | 0.9342 | 66.1 |
| **combined (7 seq)** | 5fps | - | 0.3719 | 0.3474 | 0.4052 | 0.3538 | 0.4437 | 0.7848 | 834 | 1146 | 11107 | 0.4516 | 0.8887 | - |
