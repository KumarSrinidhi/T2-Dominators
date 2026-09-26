# Model cards

One card per model Argus can run. Each says what the model does, what it was trained on, what was measured and where, and what it must not be used for.

**One rule covers every card.** A model produces observations or corroboration only. It never makes or implies a threat determination about a person. Alerts come from deterministic rules and baselines. The correlator caps an alert that only a learned model supports at **P3** until a rule agrees, and every alert needs an operator's decision. No model here does face recognition, and none was trained on DukeMTMC.

| Card | Model | Role | Default |
|---|---|---|---|
| [detector.md](detector.md) | YOLO26n / YOLO26s (COCO) + ByteTrack | Primary input: person and object boxes feeding the rules | On |
| [weapons.md](weapons.md) | Fine-tuned YOLO26n, knife + handgun | Corroboration only, capped at P3 | Off |
| [altercation.md](altercation.md) | YOLO26n-pose + heuristic + pair-GRU | Corroboration only, capped at P3 | Off |
| [fall.md](fall.md) | YOLO26n-pose + rule state machine | Posture rule ("person on the ground"), experimental | Off |
| [handoff.md](handoff.md) | HandoffNet MLP (5.6k params) | Next-camera hint, research-grade | Off |
| [reid.md](reid.md) | OSNet-AIN (torchreid, MSMT17) | Cross-camera journey links, research-grade | Off |
| [forecast.md](forecast.md) | Trajectory GRU (25.5k params) | Path prediction, labelled as a prediction | Off |
| [vlm.md](vlm.md) | Qwen2.5-VL-3B / SmolVLM | Assistive text on request, never evidence | Off (separate service) |

Weights trained for this project are not in git. Their licences depend on the training data, noted on each card. The project's own code licence has not been chosen yet.
