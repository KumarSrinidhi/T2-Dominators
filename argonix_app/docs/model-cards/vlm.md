# Model card: vision-language explainer (assistive)

**Task.** On an operator's request, describes an alert's evidence frames in plain language, summarises a multi-camera incident, or answers a natural-language investigation question. Output is labelled "assistive, not evidence" with the model name.

**Model.** A separate service (`tools/vlm_server.py`) with a registry of models, loaded to fit the available VRAM (bf16, 4-bit, or CPU):
- GPU default: Qwen2.5-VL-3B-Instruct (Qwen Research Licence), 4-bit on a 6 GB GPU.
- CPU fallback: SmolVLM-500M-Instruct (Apache-2.0).

Weights are used as released; nothing is fine-tuned.

**Measured**
- Qwen2.5-VL-3B 4-bit on an RTX 3050 Laptop: about 2 s per explanation, with descriptions that matched the frames in spot checks.
- SmolVLM-500M on a laptop CPU: about 16 s per explanation, and it hallucinated. Don't rely on the CPU fallback.

**Safeguards.**
- Incident summaries must cite alert IDs, and every cited ID must belong to the incident. Otherwise the text is refused (`investigate.check_summary`).
- `"privacy": {"blur_faces_for_vlm": true}` blurs faces and head regions before any image leaves for the VLM.
- The VLM never changes priority, status or dispositions, and never raises alerts.

**Known limitations.** VLMs can describe things that aren't in the frame, misread small or low-light details, and sound more confident than they are. Always check the clip.
