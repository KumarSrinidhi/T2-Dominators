# Model card: next-camera prediction (research-grade)

**Task.** When a person track leaves a camera's view, estimates which camera it will appear in next (or that it leaves coverage) and roughly when. It uses motion only: no appearance, no identity.

**Model.** HandoffNet (`argonix/handoff.py`): an MLP with 5.6k parameters on 46 motion features plus a camera one-hot, with a log-normal travel-time head. Softmax confidence is temperature-scaled. CPU only. By default the served ETA is the per-edge empirical median once an edge has 8 samples.

**Training data.** Cold start from `tools/sim_handoff.py`, a floor-plan simulator of walkers projected through each camera's homography. It is then fine-tuned online from self-supervised labels (exits associated with entries inside a travel-time gate). No person data leaves the site.

**Measured** (laptop CPU, synthetic data only, 2026-09-25)

| Setting | NN top-1 | Baseline top-1 |
|---|---|---|
| Held-out simulation, 375 exits | 0.960 | 0.877 |
| Rendered 3-camera video, 62 exits | 0.855 (95 % CI 0.75–0.92) | 0.806 (0.69–0.89), difference not significant |

**Known limitations.** The cold start is optimistic here because the simulator is the same floor plan that generated the video. A real site needs its topology described and weeks of traffic. Labels are ambiguous whenever several people move at once.

**What it does not do.** It is an operator hint, never evidence about a person, and it raises no alerts.

**Default.** Off (`"handoff"` config).
