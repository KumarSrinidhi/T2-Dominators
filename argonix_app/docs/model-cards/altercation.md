# Model card: possible physical altercation (experimental)

**Task.** Scores pairs of nearby people from their pose over short windows. The event reads "possible physical altercation between tracks 4 and 9". It describes an interaction, never a person, and must be checked on the clip.

**Model.** YOLO26n-pose keypoints feed a transparent heuristic score (proximity × the fastest wrist motion toward the other person). A pair-GRU (37k parameters) trained with multiple-instance learning can corroborate the heuristic. Weights: `data/models/argonix_altercation_gru.pt`, version `v1-9ddf25dd`, not in git.

**Training data.** RWF-2000 pose windows (1,353 train clips, 22,944 windows). Hockey Fight is used only as an out-of-domain test. Both datasets are for research use under their authors' terms.

**Measured** (RTX 4050 Laptop, 2026-09-25, clip level, threshold chosen on val)

| Test | Pair-GRU | Heuristic only |
|---|---|---|
| RWF-2000 test, 400 clips: ROC AUC | 0.681 | 0.647 |
| Hockey Fight, 1,000 clips: ROC AUC | 0.656 | 0.646 |
| RWF-2000 NonFight, full runtime path: false events per hour | 21.6 | 0 |
| RWF-2000 Fight, full runtime path: clips with an event | 11 of 200 | 1 of 200 |

**Known limitations.** Only slightly better than the heuristic and far from a usable fight detector. Many clips produce no scorable pair window at all (small or occluded people, one person visible). Contact sports and hugs look similar in pose.

**What it does not do.** It never raises an alert above P3 on its own, never infers intent, and never labels anyone as aggressor or victim.

**Default.** Off (`"threats"` config).
