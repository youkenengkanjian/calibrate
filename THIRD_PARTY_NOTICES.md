# Third-party sources

This repository contains a research modification of
[WildCamera](https://github.com/ShngJZ/WildCamera), the implementation of
*Tame a Wild Camera: In-the-Wild Monocular Camera Calibration* (NeurIPS 2023)
by Shengjie Zhu, Abhinav Kumar, Masa Hu, and Xiaoming Liu.
The original Apache License 2.0 is retained in `LICENSE`.
Upstream sources and authors are recorded in this file.

The `geocalib/` package and the GeoCalib-derived camera calibration / LM
utilities in `tools/geocalib.py` and `tools/lm_optimizer.py` originate from
[GeoCalib](https://github.com/cvg/GeoCalib). Its upstream license is retained
in `licenses/GeoCalib-LICENSE`.

The source comments in `geocalib/modules.py` identify components adapted
from [SegNeXt](https://github.com/Visual-Attention-Network/SegNeXt).
Existing source comments and attribution have been preserved.

Local research changes include the incidence / confidence prediction head,
confidence-aware calibration, and confidence ablation and visualization
scripts. Downloaded datasets and model checkpoints are distributed
separately by their original providers.
