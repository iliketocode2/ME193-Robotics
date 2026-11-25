# Step network (retrained)

Threshold 0.60 (best held-out F1 0.983), min gap 0.6 s, numpy/PyTorch max difference 7.9e-08.

Counts are strides of the instrumented foot (one per two footsteps).

| Recording | Set | Tracker labels | Network | Matched | Extra | Missed | Human footsteps (P8) | True strides (sim) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stairs_20261102-222719 | train | 8 | 9 | 8 | 1 | 0 | 17 |  |
| trial50_20261102-223545 | held out | 26 | 29 | 25 | 4 | 1 | 50 |  |
| turn_20261102-222638 | train | 24 | 24 | 24 | 0 | 0 | 46 |  |
| walk_20261102-222548 | train | 26 | 26 | 26 | 0 | 0 | 35 |  |
| sim00 | train | 53 | 53 | 48 | 5 | 5 |  | 53 |
| sim01 | train | 55 | 55 | 55 | 0 | 0 |  | 55 |
| sim02 | train | 68 | 68 | 68 | 0 | 0 |  | 68 |
| sim03 | train | 56 | 56 | 56 | 0 | 0 |  | 56 |
| sim04 | train | 47 | 47 | 47 | 0 | 0 |  | 47 |
| sim05 | train | 73 | 71 | 67 | 4 | 6 |  | 71 |
| sim06 | train | 52 | 52 | 52 | 0 | 0 |  | 52 |
| sim07 | train | 54 | 54 | 54 | 0 | 0 |  | 54 |
| sim08 | train | 57 | 57 | 57 | 0 | 0 |  | 57 |
| sim09 | train | 53 | 53 | 53 | 0 | 0 |  | 53 |
| sim10 | held out | 63 | 63 | 63 | 0 | 0 |  | 63 |
| sim11 | held out | 57 | 57 | 57 | 0 | 0 |  | 57 |
