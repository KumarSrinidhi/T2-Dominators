# Model card: cross-camera re-identification (research-grade)

**Task.** Links a person's tracks across cameras into one opaque global ID for one journey. The ID is a number, not a name or identity. Links are probabilistic, revisable and purged after 1 hour; embeddings are dropped 5 minutes after a track segment ends.

**Model.** OSNet-AIN x1.0 person Re-ID embeddings (torchreid), fp16 batched on GPU, on quality-filtered body crops. The appearance likelihood is fused with the learned topology prior, travel time and direction consistency (`argonix/global_ids.py`); a track joins a global ID only at P ≥ 0.6. Without the weights, a weak hand-crafted HSV/HOG embedder is used and labelled as such.

**Training data.** Pretrained by the torchreid authors on MSMT17. torchreid code is MIT; the MSMT17-trained weights are for non-commercial research use. **No DukeMTMC.** No face recognition anywhere.

**Measured** (RTX 4050 Laptop, 2026-09-25, `tools/eval_reid.py`)

| Test | Result |
|---|---|
| Single camera (`vtest.avi`), same vs different track, ROC AUC colour / grey | 0.891 / 0.885 (hand-crafted 0.782 / 0.738) |
| Rendered 3-camera video, 31 true links, colour: fused precision / recall | 0.61 / 0.36 (IDF1 0.67); appearance alone linked nothing correctly |
| Same, grey footage | 0.50 / 0.13 |

**Known limitations.** Expect wrong links: precision's Wilson 95 % interval is 0.36–0.78 on 31 links. Clothing changes, similar outfits, greyscale and IR all hurt appearance matching.

**What it does not do.** It raises no alerts and is never evidence of who someone is. Operators can split, re-link or merge links; corrections are audited.

**Default.** Off (`"reid"` config).
