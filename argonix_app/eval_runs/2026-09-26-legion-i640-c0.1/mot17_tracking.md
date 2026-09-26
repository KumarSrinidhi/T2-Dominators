# MOT17 tracking accuracy

- Measured 2026-09-26 on NVIDIA GeForce RTX 4050 Laptop GPU (host `LEGION`, Windows-10-10.0.26200-SP0)
- Detector `yolo26s.pt` imgsz 640 conf 0.1, device cuda:0
- gt: MOT17 train gt.txt: class 1 (pedestrian) rows with the consider flag set; tracker boxes matched (Hungarian, IoU >= 0.5) to distractor classes 2/7/8/12 (person on vehicle, static person, distractor, reflection) are removed before scoring (TrackEval MotChallenge2DBox preprocessing); no visibility threshold (as the official evaluation)
- metrics: CLEAR-MOT, Identity and HOTA as TrackEval (tools/eval_metrics.py): IoU >= 0.5 for MOTA/IDF1; MOTP = mean IoU of matches; HOTA averaged over alpha 0.05..0.95; combined = pooled counts
- tracker: the app's production path: frames downscaled to analytics_width 960, detector (conf 0.1, person class) + Ultralytics ByteTrack (bytetrack.yaml, track_buffer counted in processed frames), person tracks only; no public detections are used (private-detector setting)
- settings: {"5fps": "analytics rate: every round(fps/5)-th frame is processed and ONLY those frames are scored (GT subsampled the same way)", "full": "every frame processed and scored"}
- video: img1 JPEG frames re-encoded to H.264 CRF 18 (one frame per image) for the file-source pipeline
- Dataset: MOT17 train sequences (MOTChallenge, Milan et al. 2016, arXiv 1603.00831; research use)

| Sequence | Setting | Frames scored | HOTA | DetA | AssA | MOTA | IDF1 | MOTP | IDSW | FP | FN | det_recall | det_precision | Detect ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT17-02-FRCNN | 5fps | 100 | 0.2761 | 0.2165 | 0.3544 | 0.2294 | 0.3394 | 0.788 | 35 | 104 | 2250 | 0.2742 | 0.891 | 113.1 |
| MOT17-04-FRCNN | 5fps | 175 | 0.3611 | 0.2532 | 0.5166 | 0.2605 | 0.4027 | 0.8359 | 31 | 232 | 5600 | 0.2936 | 0.9094 | 128.4 |
| MOT17-05-FRCNN | 5fps | 279 | 0.3482 | 0.3862 | 0.3167 | 0.3698 | 0.4376 | 0.7261 | 160 | 297 | 1000 | 0.5675 | 0.8154 | 61.5 |
| MOT17-09-FRCNN | 5fps | 87 | 0.3922 | 0.4784 | 0.326 | 0.5181 | 0.5062 | 0.7803 | 59 | 60 | 307 | 0.6527 | 0.9058 | 103.1 |
| MOT17-10-FRCNN | 5fps | 109 | 0.1818 | 0.2581 | 0.1312 | 0.2385 | 0.2105 | 0.7548 | 154 | 60 | 1411 | 0.3388 | 0.9234 | 95.6 |
| MOT17-11-FRCNN | 5fps | 150 | 0.4135 | 0.4363 | 0.3999 | 0.4755 | 0.4897 | 0.7774 | 79 | 90 | 655 | 0.5831 | 0.9105 | 96.2 |
| MOT17-13-FRCNN | 5fps | 150 | 0.1491 | 0.1157 | 0.1957 | 0.096 | 0.1567 | 0.7562 | 116 | 15 | 1970 | 0.1523 | 0.9593 | 111.5 |
| **combined (7 seq)** | 5fps | - | 0.3223 | 0.2734 | 0.3868 | 0.2749 | 0.3686 | 0.7853 | 634 | 858 | 13193 | 0.3486 | 0.8916 | - |
