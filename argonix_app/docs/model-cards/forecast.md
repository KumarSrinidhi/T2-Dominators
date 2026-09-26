# Model card: trajectory forecast (experimental, research-grade)

**Task.** Predicts where a tracked person may be over the next 4.8 s from that person's own recent motion. It is a **prediction, not an observation**, and the UI and API label it as one.

**Model.** A GRU with 25.5k parameters (`argonix/forecast.py`) predicting a residual on top of constant velocity plus a per-step spread (σ), in a heading-aligned frame. Without a checkpoint it falls back to constant velocity.

**Training data.** ETH (eth, hotel) and UCY (univ, zara1, zara2) world-metre trajectories, leave-one-scene-out splits as redistributed in the Trajectron++ repository (research use per the data's authors). Trained on CPU in about 18 minutes (`tools/train_forecast.py`).

**Measured** (laptop CPU, 2026-09-25)
- ETH/UCY leave-one-scene-out, 8 observed → 12 predicted steps: average ADE / FDE **0.526 / 1.113 m** vs constant velocity 0.534 / 1.147 m. Roughly tied: it wins on 2 scenes and loses on 2.
- Argus's own tracks on one campus clip (262 windows): ADE / FDE 0.935 / 1.829 m vs constant velocity 0.999 / 2.034 m.
- The spread is not well calibrated (1σ coverage 29 % to 77 % across scenes, ideal 39 %).

**Known limitations.** Single-mode (can't represent "left or right"), no scene context, no other people. Uncalibrated cameras use a rough body-height scale.

**What it does not do.** The optional `predicted_zone_entry` event is severity 1 and capped at P3: an early hint, never a primary alarm.

**Default.** Off (`"forecast"` config).
