# Model card: person and object detector

**Task.** Bounding boxes for people (and, when configured, COCO bags, knives, bats, scissors and vehicles) in one frame. ByteTrack links the boxes into per-camera tracks. It does not classify intent, behaviour or identity.

**Model.** Ultralytics YOLO26n (CPU default) or YOLO26s (GPU), COCO-pretrained, used as released. On a GPU one batched model serves all cameras (`argonix/inference.py`), in fp16 or as a TensorRT engine.

**Training data.** COCO 2017, by Ultralytics. Not fine-tuned by this project.

**Licence.** Ultralytics code and weights: AGPL-3.0, or an Ultralytics enterprise licence.

**Measured** (details in the README, "GPU: one batched detector" and "Measured, not claimed")
- Laptop CPU: about 200 ms per frame for 2 cameras at imgsz 480.
- RTX 3050 Laptop: 6–14 ms per detection call; batched TensorRT YOLO26s handled about 20 cameras at 10 fps.
- Real clips (OpenCV `vtest.avi`, a Kaggle street clip): lawn intrusions visually checked as true positives; one bollard false positive, fixed with a detection mask.
- Small people (about 40 px tall, 384×288 CAVIAR lobby clips) are barely detected, even at imgsz 800.

**Known limitations.** Recall drops for small, occluded or far-away people and in low light. Greyscale and IR frames are converted and optionally contrast-enhanced (`imaging.py`), but IR accuracy has not been measured. Track IDs can switch in crowds.

**What it does not do.** It does not identify anyone and does not decide alerts. Its boxes are one input to the rules (`rules.py`), and every rule event cites the rule, threshold, zone and evidence.
