# Tri-Modal Anti-UAV Dataset and ATMF-Net

[![Dataset](https://img.shields.io/badge/Dataset-Download-blue.svg)](https://drive.google.com/drive/folders/1t_oaJZuSyBd7W4oW93-T_WvN4_0cdBuE?usp=drive_link)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue.svg)](https://github.com/eulerbaby123/Tri-Modal-Anti-UAV)

Welcome to the **Tri-Modal Anti-UAV** project! This repository contains the official code and dataset for the paper **"Adaptive Tri-Modal Fusion for Robust Anti-UAV Detection with Event Fluctuation Awareness"**.

This project aims to provide a comprehensive tri-modal (RGB, Thermal, Event) anti-UAV detection benchmark dataset, **Tri-Modal Anti-UAV**, and proposes an effective adaptive fusion network, **ATMF-Net**, for UAV object detection.

## ⚖️ Ethical Considerations and Privacy

All data in this dataset were collected in public or controlled areas and do not involve personal privacy information. Any persons appearing in the images (if any) have been blurred or otherwise de-identified. All code, datasets, and related resources provided in this project are permitted by default for any non-commercial use in scientific research, education, or personal experiments, without requiring special authorization from the authors. For commercial applications, please contact the authors.

## Table of Contents

- [Tri-Modal Anti-UAV Dataset and ATMF-Net](#tri-modal-anti-uav-dataset-and-atmf-net)
  - [⚖️ Ethical Considerations and Privacy](#️-ethical-considerations-and-privacy)
  - [Table of Contents](#table-of-contents)
  - [📝 Introduction](#-introduction)
  - [📸 Dataset: Tri-Modal Anti-UAV](#-dataset-tri-modal-anti-uav)
    - [Data Overview](#data-overview)
    - [Data Collection and Processing](#data-collection-and-processing)
    - [Dataset Statistics](#dataset-statistics)
    - [Scene Showcase](#scene-showcase)
    - [Data Download](#data-download)
  - [🔧 Image Alignment](#-image-alignment)
  - [🚀 Models and Weights](#-models-and-weights)
    - [ATMF-Net](#atmf-net)
    - [LW-MoESGF (RGB+IR)](#lw-moesgf-rgbir)
    - [Other Experimental Resources](#other-experimental-resources)
  - [📊 Main Results](#-main-results)
    - [Tables](#tables)
    - [Figures](#figures)
    - [Dataset Supplementary Notes](#dataset-supplementary-notes)
  - [🛠️ Installation](#️-installation)
    - [Environment Requirements](#environment-requirements)
    - [Installation Steps](#installation-steps)
    - [Key Dependencies](#key-dependencies)
  - [⚙️ Usage](#️-usage)
    - [Data Preparation](#data-preparation)
    - [Training](#training)
    - [Evaluation](#evaluation)

## 📝 Introduction

The proliferation of unmanned aerial vehicles (UAVs) necessitates
robust anti-UAV detection systems. While multi-modal fusion (e.g.,
RGB-Thermal) improves resilience, performance bottlenecks persist
in extreme scenarios like motion blur and low contrast. Event cam-
eras offer high dynamic range and temporal resolution but suffer
from inherent data quality fluctuations, which existing datasets
fail to systematically capture. To bridge this gap, we introduce
Tri-Modal Anti-UAV : the first tri-modal (RGB, Thermal, Event)
dataset specifically designed for anti-UAV research. It features 1,060
synchronised image triplets across diverse scenarios (e.g., high-
altitude tiny targets, poor illumination, environmental background
interference), with a unique emphasis on preserving the full spec-
trum of event data quality—from dense to sparse/noisy streams.
Building on this benchmarking dataset, we propose ATMF-Net,
an Adaptive Tri-Modal Fusion network that dynamically modu-
lates event modality contributions based on real-time reliability
estimation. Our lightweight architecture integrates a Mixture-of-
Experts framework and Self-Guided Fusion, achieving high effi-
ciency while outperforming non-adaptive fusion. Rigorous bench-
marking validates Tri-Modal Anti-UAV ’s challenging nature: event-
only detection performs poorly (9.76% mAP50), yet adaptive tri-
modal fusion elevates accuracy to 89.9% mAP50. Our dataset pro-
vides a critical resource for developing event-aware, robust anti-
UAV detectors.

**Key Features:**
*   The first tri-modal dataset (RGB, Thermal Infrared, Event Camera) specifically for anti-UAV detection.
*   Systematically captures and preserves event data streams of varying quality, from dense and clear to sparse and noisy, closely reflecting real-world application scenarios and providing a critical resource for developing event-aware and robust detection algorithms.
*   Contains 1,060 synchronized image triplets covering diverse complex scenarios (e.g., high-altitude small targets, low-light conditions, background interference) and UAV types.
*   Proposes ATMF-Net, an adaptive tri-modal fusion network that dynamically adjusts the contribution of event data based on its real-time reliability.
*   Provides implementations for ATMF-Net, LW-MoESGF (dual-modal baseline), and other models.
*   Detailed evaluation metrics and results, establishing strong baselines.

## 📸 Dataset: Tri-Modal Anti-UAV

### Data Overview
The **Tri-Modal Anti-UAV** dataset is a novel tri-modal benchmark specifically curated for anti-UAV research. It comprises synchronized Visible (RGB), Thermal Infrared (T), and Event-based (E) data streams.

### Data Collection and Processing
**Data Collection**:
*   The dataset contains a total of 1,060 annotated image sets (855 for training, 205 for testing).
*   Data for each modality were collected using dedicated sensors: a conventional RGB camera, a thermal infrared camera, and a DAVIS346 event sensor.
*   UAV platforms include DJI Mini 3 and DJI Mavic 3 Pro models, with the number of UAVs per scene ranging from 1 to 3.
*   Data collection covers a wide range of environmental conditions: weather variations (sunny to cloudy), different times of day. UAV flight profiles are diverse, with altitudes from near-ground to hundreds of meters, captured from multiple camera perspectives.
*   Operating environments are varied, including complex urban settings, open grasslands, dense forests, lake surfaces, unobstructed high altitudes, and foothill terrains.
*   Special attention was paid to preserving event data of varying quality levels, from information-rich, clear events to sparse, noisy ones.

**Data Processing and Annotation**:
*   **Event Data Processing**: Raw asynchronous event streams from the event camera are accumulated over fixed 20ms time windows to generate event frames, balancing motion blur and information density.
*   **Multi-modal Alignment**: Given differences in sensor specifications (resolution, field of view) and physical arrangement, a feature-point-based registration technique is employed to align RGB and event frames to the coordinate system of the thermal modality (which has the highest native resolution). An affine transformation matrix is applied to ensure that regions containing UAVs are spatially consistent across the three modalities. A "weak alignment" strategy is adopted, meaning it does not enforce strict pixel-level correspondence of UAVs within bounding boxes and also allows for weak alignment of background regions, aiming to encourage the development of robust fusion mechanisms insensitive to slight spatial inconsistencies.
*   **Data Annotation**: All data were annotated in YOLO format using LabelImg. Annotations were made on images created by pixel-level fusion of the aligned tri-modal data, and these annotations can be directly transferred to the registered RGB, IR, and event data frames.

### Dataset Statistics
**Table: Key Statistics of the Tri-Modal Anti-UAV Dataset**
| Attribute                             | Proportion (Instances) |
|---------------------------------------|------------------------|
| Small Targets (e.g., area < 8x8 pixels) | 7.96%                  |
| Low-light/Extreme Light Scenarios     | 8.92%                  |
| High-Quality Event Data               | 6.94%                  |
| Complex Background Interference       | 23.2%                  |

### Scene Showcase
The dataset includes a variety of challenging scenarios.

**Dataset Sample Overview:**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-49-42.png?raw=true" width="1000" alt="Dataset Samples">
  <br/><em>Figure: Sample images from the Tri-Modal Anti-UAV dataset. Targets are marked with red boxes. Top row: RGB modality; Middle row: Thermal Infrared modality; Bottom row: Event modality, showcasing the diversity of event data quality (Corresponds to Figure 1 in the paper).</em>
</div>

**Examples of Various Quality Event Data:**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/1415f24196421d9e56c68916e871d3a260d8debc/images/Screenshot2025-06-01_18-39-12.png?raw=true" width="1000" alt="Event Data Quality Examples">
  <br/><em>Figure: Demonstration of the diversity in event modality data quality. The dataset includes data of various qualities, including noisy and uninformative cases.</em>
</div>

**Diverse Capture Scenes (Shown in Infrared Modality):**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/7821aac26e95be05f95116f40abd9d082f66017c/images/Screenshot2025-06-01_18-57-35.png?raw=true" width="1000" alt="Diverse Scenes IR">
  <br/><em>Figure: Diverse capture scenes in the dataset (some examples shown in the infrared modality).</em>
</div>

**Other key scenario types include:**
*   High-altitude small targets
*   UAVs in low-light environments
*   UAVs with complex background interference (e.g., tree branches, buildings)
*   Fast-moving UAVs

### Data Download
You can download the complete dataset from the following link:
*   **Google Drive**: [https://drive.google.com/drive/folders/1t_oaJZuSyBd7W4oW93-T_WvN4_0cdBuE?usp=drive_link](https://drive.google.com/drive/folders/1t_oaJZuSyBd7W4oW93-T_WvN4_0cdBuE?usp=drive_link)

The dataset uses YOLO annotation format.

## 🔧 Image Alignment

Due to inherent differences in sensors (such as resolution and field of view) and their fixed, non-co-located physical arrangement, we employ a feature-point-based registration technique to align RGB and event frames with the coordinate system of the thermal infrared modality. An estimated affine transformation matrix is applied, primarily ensuring that regions containing UAVs are spatially consistent across the three modalities. We adopt a "weak alignment" strategy, intentionally not enforcing strict pixel-level correspondence of UAVs within annotated bounding boxes, to encourage the development of fusion mechanisms more robust to minor spatial inconsistencies.

**Example of Images Before Alignment (RGB aligned to Infrared, green dots are feature points):**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-57-22.png?raw=true" width="800" alt="Image Alignment Before">
  <br/><em>Figure: Images before alignment (green dots represent corresponding feature points, shown here for RGB to Infrared alignment as an example).</em>
</div>

**Example of Images After Alignment:**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-55-17.png?raw=true" width="800" alt="Image Alignment After">
  <br/><em>Figure: Example of aligned images, demonstrating spatial consistency of the target region across different modalities.</em>
</div>

Aligned images ensure spatial consistency of target regions across different modalities, laying the foundation for subsequent unified annotation and effective multi-modal fusion.

## 🚀 Models and Weights

### ATMF-Net
Our proposed ATMF-Net (Adaptive Tri-Modal Fusion Network) is an effective network structure for fusing tri-modal information, specifically designed for UAV detection. Its core idea is to dynamically evaluate and adjust the contribution of the event modality in the fusion process based on its real-time reliability, thereby maintaining detection robustness when event data quality fluctuates.
*   **Code**: `./models/ATMF_Net/` (Please replace with the actual path if different)
*   **Best Trained Weights (`best.pt`) GDrive**: [https://drive.google.com/file/d/1xsx8g-1wAIUPylxw0jj6pXMck-VM_JX7/view?usp=drive_link](https://drive.google.com/file/d/1xsx8g-1wAIUPylxw0jj6pXMck-VM_JX7/view?usp=drive_link)
*   **Initial Backbone Weights**: Training often starts from standard pre-trained backbone weights (e.g., `yolov5l.pt` for a YOLOv5-Large based model). These are typically obtained from the original YOLOv5 repository or similar sources.

**ATMF-Net Network Architecture Diagram:**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-48-55.png?raw=true" width="1200" alt="ATMF-Net Architecture">
  <br/><em>Figure: ATMF-Net network architecture. Right: Overall fusion path (using RGB features as an example). Left: Details of the Tri-Modal Fusion Expert. Key components include the Event Reliability Estimator (ERE) and the MoE router for dynamic expert weight adjustment. $\oplus$: Feature addition, $\otimes$: Weighted fusion (Corresponds to Figure 2 in the paper).</em>
</div>

**Self-Guided Fusion (SGF) Module Structure Diagram (ATMF-Net Component):**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_17-04-09.png?raw=true" width="400" alt="Self-Guided Fusion Architecture">
  <br/><em>Figure: Detailed structure of the Self-Guided Fusion (SGF) module, a key component in ATMF-Net.</em>
</div>


### Other Experimental Resources
*   **Other Paper-related Experimental Code and Weights GDrive**: [https://drive.google.com/file/d/1WDaYFGmbvIM_oK0p7rGdpjXrhATFc6l2/view?usp=drive_link](https://drive.google.com/file/d/1WDaYFGmbvIM_oK0p7rGdpjXrhATFc6l2/view?usp=drive_link)
    *   This link contains other model codes and/or pre-trained weights used for comparative experiments or ablation studies in the paper. Please follow the instructions within the archive.

## 📊 Main Results

### Tables

**Table 1: Effectiveness of adaptive tri-modal fusion**
| Method                                      | mAP$_{50}$ (%) |
|-------------------------------------------|----------------|
| LW-MoESGF (RGB+IR)                        | 87.4           |
| Tri-Modal (Non-adaptive)                  | 87.8           |
| **ATMF-Net (Adaptive)**                   | **89.9**       |

**Table 2: Performance of single-modality detection on the Tri-Modal Anti-UAV test set**
| Modality                   | mAP$_{50}$ (%) | mAP (%) |
|------------------------|----------------|---------|
| YOLOv5l (RGB-only)     | 65.5           | 20.2    |
| YOLOv5l (IR-only)      | **78.8**       | **27.2**|
| YOLOv5l (Event-only)   | 9.76           | 3.57    |

**Table 3: Performance and efficiency comparison of RGB-IR dual-modal fusion methods**
| Method                      | Parameters (M) | FLOPs (G) | mAP$_{50}$ (%) |
|---------------------------|----------------|-----------|----------------|
| Best Single Modality (IR) | **46.5**       | **109**   | 78.8           |
| CFT (RGB+IR) [Li et al., 2021] | 206        | 224       | 86.6           |
| LW-MoESGF (RGB+IR)        | 76.2       | 192       | **87.4**       |


### Figures

#### Qualitative Comparison of Detection Results
**Comparison of Event Modality Effects and Detection Results:**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-49-24.png?raw=true" width="600" alt="Qualitative Results Comparison">
  <br/><em>Figure: Qualitative comparison on Tri-Modal Anti-UAV. From left to right: RGB, IR, and Event inputs. Overlaid on images: Ground Truth (red box), detection results from LW-MoESGF (RGB+IR, green box), and our ATMF-Net (blue box) (Corresponds to Figure 4 in the paper).</em>
</div>

### Dataset Supplementary Notes
It should be noted that in the dataset, a small number of images, due to factors such as being captured in close-range scenarios, inherent viewing angle differences between sensors, and pixel resolution disparities, may have areas without pixel information (uniformly filled with black or white) in their peripheral regions after affine transformation alignment. Coupled with the inherent limitations of the affine transformation alignment method itself, there might be some visual discrepancies between pre- and post-alignment images.

We chose to retain these samples rather than discard them because:
1.  **Real-world Scenario Simulation**: Such situations can occur in practical multi-sensor fusion applications; retaining them helps models learn to cope with such imperfect alignments.
2.  **Robustness Enhancement**: These visual differences, introduced by alignment but with largely unchanged content information, can be considered a form of data augmentation or interference. We believe this can, in turn, encourage the model to learn more essential and robust features, improving its generalization capabilities in complex real-world environments.

The following image shows an example of such an aligned image with edge padding:
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/34fabfae1f61173924738877dea5e85addc5423b/images/Screenshot2025-06-01_19-31-01.png?raw=true" width="400" alt="Alignment Artifact Example">
  <br/><em>Figure: Example of an aligned image where edge regions might lack pixel information (black padding).</em>
</div>

## 🛠️ Installation

### Environment Requirements
*   Python >= 3.8
*   PyTorch >= 1.7.0
*   CUDA [11.3] (if using GPU)
*   Other dependencies are listed in `requirements.txt`

### Installation Steps

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/eulerbaby123/Tri-Modal-Anti-UAV.git
    cd Tri-Modal-Anti-UAV
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Windows
    # venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    If you encounter issues installing `pycocotools`, please refer to its official documentation. Windows users might need to install Microsoft C++ Build Tools beforehand.



## ⚙️ Usage

### Data Preparation
1.  Download the Tri-Modal Anti-UAV dataset (link in the [Data Download](#data-download) section).
2.  Extract the dataset and organize it as follows (or adjust according to your configuration files):
    ```
    Tri-Modal-Anti-UAV/
    ├── images/       # Contains subdirectories for rgb, ir, event images for train/val splits
    │   ├── train/
    │   │   ├── rgb/
    │   │   ├── ir/
    │   │   └── event/
    │   └── val/
    │       ├── rgb/
    │       ├── ir/
    │       └── event/
    ├── labels/       # Contains corresponding YOLO format label .txt files for train/val splits
    │   ├── train/
    │   └── val/
    └── train_rgb.txt   # List of training RGB image paths
    └── train_ir.txt    # List of training IR image paths
    └── train_event.txt # List of training Event image paths
    └── val_rgb.txt     # List of validation RGB image paths
    └── val_ir.txt      # List of validation IR image paths
    └── val_event.txt   # List of validation Event image paths
    ```
    

### Testing  or Training
Use the following command to start testing and training (parameters can be modified within the script files as per actual requirements):
```bash
python test.py
python train.py 
# Check train.py or associated config files for parameters like model config, data config, weights, batch size, epochs, device, etc.
