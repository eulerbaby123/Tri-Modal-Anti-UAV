# uavRGBTE


this is the offical code and dataset for Tri-Modal Anti-UAV: A Comprehensive Benchmarking Dataset for
UAV-Targeted Detection.


Abstract
The proliferation of unmanned aerial vehicles (UAVs) necessitates
robust anti-UAV detection systems. While multi-modal fusion (e.g.,
RGB-Thermal) improves resilience, performance bottlenecks persist
in extreme scenarios like motion blur and low contrast. Event cam-
eras offer high dynamic range and temporal resolution but suffer
from inherent data quality fluctuations, which existing datasets
fail to systematically capture. To bridge this gap, we introduce Tri-
Modal Anti-UAV : the first tri-modal (RGB, Thermal, Event) dataset
specifically designed for anti-UAV research. It features 1,060 syn-
chronised image triplets across diverse scenarios (e.g., high-altitude
tiny targets, poor illumination), with a unique emphasis on pre-
serving the full spectrum of event data quality—from dense to
sparse/noisy streams. Building on this benchmarking dataset, we
propose ATMF-Net, an Adaptive Tri-Modal Fusion network that
dynamically modulates event modality contributions based on real-
time reliability estimation. Our lightweight architecture integrates
a Mixture-of-Experts framework and Self-Guided Fusion, achieving
high efficiency while outperforming non-adaptive fusion. Rigorous
benchmarking validates Tri-Modal Anti-UAV ’s challenging nature:
event-only detection performs poorly (3.57% mAP50), yet adaptive
tri-modal fusion elevates accuracy to 89.9% mAP50. Our dataset
provides a critical resource for developing event-aware, robust
anti-UAV detectors. Data, code, and models are open-sourced at:
https://github.com/eulerbaby123/Tri-Modal-Anti-UAV.
