# MOT17 tracking accuracy

- Measured 2026-09-26 on NVIDIA GeForce RTX 4050 Laptop GPU (host `LEGION`, Windows-10-10.0.26200-SP0)
- Detector `yolo26s.pt` imgsz 480 conf 0.3, device cuda:0
- gt: MOT17 train gt.txt: class 1 (pedestrian) rows with the consider flag set; tracker boxes matched (Hungarian, IoU >= 0.5) to distractor classes 2/7/8/12 (person on vehicle, static person, distractor, reflection) are removed before scoring (TrackEval MotChallenge2DBox preprocessing); no visibility threshold (as the official evaluation)
- metrics: CLEAR-MOT, Identity and HOTA as TrackEval (tools/eval_metrics.py): IoU >= 0.5 for MOTA/IDF1; MOTP = mean IoU of matches; HOTA averaged over alpha 0.05..0.95; combined = pooled counts
- tracker: the app's production path: frames downscaled to analytics_width 960, detector (conf 0.3, person class) + Ultralytics ByteTrack (bytetrack.yaml, track_buffer counted in processed frames), person tracks only; no public detections are used (private-detector setting)
- settings: {"5fps": "analytics rate: every round(fps/5)-th frame is processed and ONLY those frames are scored (GT subsampled the same way)", "full": "every frame processed and scored"}
- video: img1 JPEG frames re-encoded to H.264 CRF 18 (one frame per image) for the file-source pipeline
- Dataset: MOT17 train sequences (MOTChallenge, Milan et al. 2016, arXiv 1603.00831; research use)

| Sequence | Setting | Frames scored | HOTA | DetA | AssA | MOTA | IDF1 | MOTP | IDSW | FP | FN | det_recall | det_precision | Detect ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT17-02-FRCNN | 5fps | 100 | 0.2429 | 0.1617 | 0.3664 | 0.1784 | 0.2548 | 0.8007 | 27 | 36 | 2484 | 0.1987 | 0.9448 | 94.5 |
| MOT17-02-FRCNN | full | 600 | 0.2847 | 0.1788 | 0.4552 | 0.1961 | 0.2923 | 0.8115 | 37 | 343 | 14558 | 0.2165 | 0.9214 | 71.5 |
| MOT17-04-FRCNN | 5fps | 175 | 0.289 | 0.1705 | 0.493 | 0.1717 | 0.2868 | 0.816 | 15 | 198 | 6354 | 0.1985 | 0.8883 | 105.8 |
| MOT17-04-FRCNN | full | 1050 | 0.3096 | 0.1906 | 0.5045 | 0.198 | 0.3199 | 0.835 | 37 | 1009 | 37096 | 0.22 | 0.912 | 77.2 |
| MOT17-05-FRCNN | 5fps | 279 | 0.3389 | 0.3827 | 0.3027 | 0.3893 | 0.4405 | 0.7322 | 145 | 215 | 1052 | 0.545 | 0.8542 | 61.8 |
| MOT17-05-FRCNN | full | 837 | 0.4343 | 0.4286 | 0.443 | 0.4821 | 0.5939 | 0.7469 | 98 | 732 | 2752 | 0.6021 | 0.8505 | 58.8 |
| MOT17-09-FRCNN | 5fps | 87 | 0.3724 | 0.4381 | 0.3192 | 0.4604 | 0.4922 | 0.7743 | 80 | 48 | 349 | 0.6052 | 0.9177 | 104.4 |
| MOT17-09-FRCNN | full | 525 | 0.4503 | 0.5227 | 0.3906 | 0.5968 | 0.5351 | 0.807 | 67 | 387 | 1693 | 0.6821 | 0.9037 | 79.3 |
| MOT17-10-FRCNN | 5fps | 109 | 0.185 | 0.2179 | 0.1607 | 0.2095 | 0.2158 | 0.7605 | 101 | 49 | 1537 | 0.2798 | 0.9241 | 100.4 |
| MOT17-10-FRCNN | full | 654 | 0.3224 | 0.2987 | 0.3492 | 0.3371 | 0.4033 | 0.7708 | 91 | 481 | 7939 | 0.3816 | 0.9106 | 79.1 |
| MOT17-11-FRCNN | 5fps | 150 | 0.4239 | 0.4142 | 0.4396 | 0.4723 | 0.5164 | 0.7858 | 53 | 68 | 708 | 0.5493 | 0.927 | 89.5 |
| MOT17-11-FRCNN | full | 900 | 0.4951 | 0.4916 | 0.503 | 0.5299 | 0.5673 | 0.8297 | 40 | 687 | 3709 | 0.6069 | 0.8929 | 71.6 |
| MOT17-13-FRCNN | 5fps | 150 | 0.133 | 0.0825 | 0.2162 | 0.0749 | 0.1256 | 0.759 | 66 | 8 | 2076 | 0.1067 | 0.9688 | 103.6 |
| MOT17-13-FRCNN | full | 750 | 0.258 | 0.1808 | 0.3737 | 0.2157 | 0.3145 | 0.7556 | 83 | 192 | 8856 | 0.2393 | 0.9355 | 76.5 |
| **combined (7 seq)** | 5fps | - | 0.2863 | 0.2213 | 0.3767 | 0.2263 | 0.3141 | 0.779 | 487 | 622 | 14560 | 0.2811 | 0.9015 | - |
| **combined (7 seq)** | full | - | 0.3415 | 0.2582 | 0.4556 | 0.2797 | 0.3856 | 0.8034 | 453 | 3831 | 76603 | 0.3179 | 0.9031 | - |
