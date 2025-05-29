# uavRGBTE


this is the offical code and dataset for uavRGBTE


Abstract
Detecting Unmanned Aerial Vehicles (UAVs) effectively is crucial
for public safety but challenging in dynamic environments, espe-
cially for small, agile targets. While RGB and Infrared (IR) sensors
have limitations, event cameras offer promise with high temporal
resolution, yet their data quality varies significantly (from infor-
mative to noisy) depending on conditions, a critical issue often
overlooked that impacts multi-modal fusion reliability. Existing
datasets also lack systematic coverage of this event data quality
diversity.
To address these gaps, we introduce uavRGBTE, a novel open-
source tri-modal (RGB, Thermal, Event) benchmark dataset for
Counter-UAS (C-UAS) applications. uavRGBTE uniquely features a
comprehensive spectrum of event data quality alongside diverse,
challenging scenarios (e.g., small high-altitude targets, complex
backgrounds, extreme lighting). Building on this, we propose the
Adaptive Tri-Modal Fusion Network (ATMF-Net), which dynami-
cally assesses event stream reliability to modulate its contribution
in the fusion process. This adaptive approach aims to maximize
high-quality event information utility while mitigating interference
from poor-quality data, enhancing detection robustness in variable
real-world conditions.
Our contributions include: (1) The uavRGBTE dataset, filling a
critical need for diverse event data quality in C-UAS research. (2)
ATMF-Net, a novel event-aware adaptive fusion architecture. (3)
Comprehensive benchmarking of uni-modal, dual-modal, and our
tri-modal fusion methods on uavRGBTE. 
