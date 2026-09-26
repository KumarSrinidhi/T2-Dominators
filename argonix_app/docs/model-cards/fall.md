# Model card: person on the ground / possible fall (experimental)

**Task.** Reports that a tracked person's posture changed from upright to horizontal quickly and stayed horizontal near the ground (`person_down`). The text describes posture only. It makes no claim about injury, health, cause or intent.

**Model.** A transparent per-track state machine (`argonix/fall.py`) over tracker boxes and YOLO26n-pose keypoints: upright reference height, box aspect, torso angle, hip height, a rapid-descent check and a 3 s persistence window. An optional classifier hook exists; **no classifier is trained or shipped**.

**Training data.** None; the thresholds are hand-set and were not tuned to the test clips. The pose model is Ultralytics YOLO26n-pose (COCO keypoints; AGPL-3.0).

**Measured** (laptop CPU, 2026-09-25, `tools/eval_fall.py`; a sanity check, not a benchmark)
- CAVIAR on-the-floor clips: **0 of 3 detected**. The detector barely sees the small person, so no upright history exists.
- 4.6 minutes of normal walking: 0 false alarms.
- Synthetic box and keypoint sequences: behaves as designed (falls vs sitting, bending, slow lying down, getting up, edge truncation).

**Known limitations.** Unvalidated on real falls. High wide-angle cameras foreshorten a lying body; furniture hides posture; a track ID change during the fall removes the upright reference.

**Default.** Off. The project manual advises against fall detection for this demo; it is built on request and labelled experimental everywhere.
