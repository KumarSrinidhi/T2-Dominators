# MOT17 tracking accuracy

- Measured 2026-09-26 on NVIDIA GeForce RTX 4050 Laptop GPU (host `LEGION`, Windows-10-10.0.26200-SP0)
- Detector `yolo26s.pt` imgsz 480 conf 0.1, device cuda:0
- gt: MOT17 train gt.txt: class 1 (pedestrian) rows with the consider flag set; tracker boxes matched (Hungarian, IoU >= 0.5) to distractor classes 2/7/8/12 (person on vehicle, static person, distractor, reflection) are removed before scoring (TrackEval MotChallenge2DBox preprocessing); no visibility threshold (as the official evaluation)
- metrics: CLEAR-MOT, Identity and HOTA as TrackEval (tools/eval_metrics.py): IoU >= 0.5 for MOTA/IDF1; MOTP = mean IoU of matches; HOTA averaged over alpha 0.05..0.95; combined = pooled counts
- tracker: the app's production path: frames downscaled to analytics_width 960, detector (conf 0.1, person class) + Ultralytics ByteTrack (bytetrack.yaml, track_buffer counted in processed frames), person tracks only; no public detections are used (private-detector setting)
- settings: {"5fps": "analytics rate: every round(fps/5)-th frame is processed and ONLY those frames are scored (GT subsampled the same way)", "full": "every frame processed and scored"}
- video: img1 JPEG frames re-encoded to H.264 CRF 18 (one frame per image) for the file-source pipeline
- Dataset: MOT17 train sequences (MOTChallenge, Milan et al. 2016, arXiv 1603.00831; research use)

| Sequence | Setting | Frames scored | HOTA | DetA | AssA | MOTA | IDF1 | MOTP | IDSW | FP | FN | det_recall | det_precision | Detect ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT17-02-FRCNN | 5fps | 100 | 0.2493 | 0.1641 | 0.3804 | 0.1658 | 0.2658 | 0.7984 | 32 | 81 | 2473 | 0.2023 | 0.8856 | 129.3 |
| MOT17-04-FRCNN | 5fps | 175 | 0.3111 | 0.1998 | 0.4876 | 0.2066 | 0.3242 | 0.8182 | 17 | 205 | 6068 | 0.2346 | 0.9007 | 106.0 |
| MOT17-05-FRCNN | 5fps | 279 | 0.3434 | 0.3767 | 0.3169 | 0.3737 | 0.4483 | 0.7274 | 138 | 259 | 1051 | 0.5454 | 0.8296 | 64.2 |
| MOT17-09-FRCNN | 5fps | 87 | 0.3738 | 0.4442 | 0.3178 | 0.4548 | 0.501 | 0.7699 | 83 | 62 | 337 | 0.6188 | 0.8982 | 110.3 |
| MOT17-10-FRCNN | 5fps | 109 | 0.1889 | 0.2222 | 0.164 | 0.2118 | 0.2213 | 0.7588 | 100 | 58 | 1524 | 0.2858 | 0.9132 | 135.9 |
| MOT17-11-FRCNN | 5fps | 150 | 0.4238 | 0.4162 | 0.4376 | 0.4729 | 0.5152 | 0.7857 | 54 | 70 | 704 | 0.5519 | 0.9253 | 116.6 |
| MOT17-13-FRCNN | 5fps | 150 | 0.135 | 0.0874 | 0.2106 | 0.0792 | 0.1309 | 0.7563 | 71 | 9 | 2060 | 0.1136 | 0.967 | 102.7 |
| **combined (7 seq)** | 5fps | - | 0.297 | 0.2338 | 0.3836 | 0.2369 | 0.3318 | 0.7794 | 495 | 744 | 14217 | 0.298 | 0.8903 | - |
