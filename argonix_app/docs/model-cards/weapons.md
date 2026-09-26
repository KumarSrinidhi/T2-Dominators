# Model card: weapon-like object detector (experimental)

**Task.** Boxes for knives and handguns, on frames and on person crops, at 3 fps. An event ("possible weapon-like object: handgun, near track 12") needs 3 of 5 detections at confidence ≥ 0.5 with a person nearby. The event describes an object, never a person.

**Model.** YOLO26n fine-tuned by `tools/train_weapons.py` (50 epochs at 640 px, batch 16, 95 min on an RTX 4050 Laptop, 2026-09-25), merged by NMS with the COCO classes knife, bat and scissors. Weights: `data/models/argonix_weapons_yolo26n.pt`, version `v1-9ae9b3aa`, not in git.

**Training data.** Kaggle `raghavnanjappan/weapon-dataset-for-yolov5`, 4,154 images, split 3,197 / 325 / 324 with near-duplicates (dHash) kept in one split. The dataset licence is **unknown**: personal demo use only; don't redistribute the weights.

**Measured** (RTX 4050 Laptop, 2026-09-25, `tools/eval_threats.py`, `tools/compare_pretrained.py`)

| Test | Result |
|---|---|
| Held-out test split: handgun / knife AP50 | 0.906 / 0.605 |
| Synthetic CCTV views (`simuletic/cctv-weapon-dataset`), class-agnostic recall at 0.5 | 0.08 |
| RWF-2000 NonFight (no weapons), full runtime path | 137–151 false events per hour |
| Third-party YOLOv8n (`Subh775`) on the same tests | handgun 0.685, knife 0.525, 94 false events per hour |

**Known limitations.** It does not generalise to CCTV viewpoints and raises roughly one false event every 30 seconds on normal footage. It is a demonstration of the corroboration-only design, not a working detector.

**What it does not do.** It never raises an alert above P3 on its own and never labels a person.

**Default.** Off (`"threats"` config).
