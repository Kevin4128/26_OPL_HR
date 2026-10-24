# BoT-SORT reference

Tracker and XYWH Kalman equations ported to C++ from:
https://github.com/NirAharon/BoT-SORT/tree/251985436d6712aaf682aaaf5f71edb4987224bd

Reference files: `tracker/bot_sort.py`, `tracker/kalman_filter.py`, `tracker/matching.py`.
Copyright (c) 2022 Nir Aharon. MIT license included beside this file.

OPL adaptations: OpenCV matrices and existing sparse optical-flow GMC; OSNet
instead of FastReID; timestamps scaled to nominal 30 Hz steps (capped at 200 ms);
configurable consecutive-hit confirmation and predicted-box publishing limits;
no appearance updates from overlapping/ambiguous crops or weak detections;
existing enrolled-identity uncertainty and observed/depth/TF contracts.

The association stages follow BoT-SORT: tracked+lost versus strong detections,
remaining tracked versus weak detections, tentative versus remaining strong
detections, then births and tracked/lost duplicate removal. Appearance uses a
normalized 0.9 EMA, proximity masking, and minimum of score-fused IoU cost and
half cosine distance. The appearance parameter remains minimum cosine similarity.
