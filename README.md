# uavRGBTE: Tri-Modal Anti-UAV Dataset and ATMF-Net

[![Paper](https://img.shields.io/badge/Paper-ACMMM'25_PLACEHOLDER-B31B1B.svg)](YOUR_PAPER_LINK_HERE)
[![Dataset](https://img.shields.io/badge/Dataset-Download-blue.svg)](https://drive.google.com/drive/folders/1t_oaJZuSyBd7W4oW93-T_WvN4_0cdBuE?usp=drive_link)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue.svg)](https://github.com/eulerbaby123/Tri-Modal-Anti-UAV)

欢迎来到 **uavRGBTE** 项目！这是论文 **"Adaptive Tri-Modal Fusion for Robust Anti-UAV Detection with Event Fluctuation Awareness"** (暂定名，请根据实际论文题目修改) 的官方代码和数据集。

本项目旨在提供一个全面的三模态（RGB、Thermal、Event）反无人机检测基准数据集 **Tri-Modal Anti-UAV**，并提出了一种有效的自适应融合网络 **ATMF-Net** 用于无人机目标检测。

## 目录

- [uavRGBTE: Tri-Modal Anti-UAV Dataset and ATMF-Net](#uavrgbte-tri-modal-anti-uav-dataset-and-atmf-net)
  - [目录](#目录)
  - [📝 简介](#-简介)
  - [📸 数据集: Tri-Modal Anti-UAV](#-数据集-tri-modal-anti-uav)
    - [数据概览](#数据概览)
    - [数据采集与处理](#数据采集与处理)
    - [数据集统计](#数据集统计)
    - [场景展示](#场景展示)
    - [数据下载](#数据下载)
  - [🔧 图像对齐方式](#-图像对齐方式)
  - [🚀 模型与权重](#-模型与权重)
    - [ATMF-Net](#atmf-net)
    - [LW-MoESGF (RGB+IR)](#lw-moesgf-rgbir)
  - [📊 主要结果](#-主要结果)
    - [表格](#表格)
    - [图示](#图示)
  - [🛠️ 安装](#️-安装)
    - [环境要求](#环境要求)
    - [安装步骤](#安装步骤)
    - [依赖库](#依赖库)
  - [⚙️ 使用](#️-使用)
    - [数据准备](#数据准备)
    - [训练](#训练)
    - [评估](#评估)
  - [📜 引用](#-引用)
  - [📄 许可证](#-许可证)
  - [🙏 致谢](#-致谢)
  - [📞 联系方式](#-联系方式)

## 📝 简介

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

**主要特性:**
*   首个针对反无人机检测的三模态数据集（RGB、红外热成像、事件相机数据）。
*   系统性地捕获并保留了从密集清晰到稀疏嘈杂的各种质量的事件数据流，更贴近真实应用场景，为开发事件感知和鲁棒的检测算法提供了关键资源。
*   包含1,060组同步图像三元组，覆盖多种复杂场景（如高空小目标、弱光照、背景干扰）和无人机类型。
*   提出了 ATMF-Net，一种根据事件数据实时可靠性动态调整其贡献的自适应三模态融合网络。
*   提供了 ATMF-Net 及 LW-MoESGF (双模态基线) 等模型的实现。
*   详细的评估指标和结果，建立了强大的基线。

## 📸 数据集: Tri-Modal Anti-UAV

### 数据概览
**Tri-Modal Anti-UAV** 数据集是专门为反无人机研究策划的新型三模态基准。它包含同步的可见光（RGB）、热红外（T）和基于事件（E）的数据流。

### 数据采集与处理
**数据采集**:
*   数据集共包含1,060组标注图像集（855组用于训练，205组用于测试）。
*   使用专用传感器采集各模态数据：传统RGB相机、热红外相机和DAVIS346事件传感器。
*   无人机平台包括大疆Mini 3和大疆Mavic 3 Pro型号，每场景无人机数量从1到3不等。
*   数据采集覆盖广泛的环境条件：天气变化（晴天到阴天）、一天中的不同时段。无人机飞行剖面多样，高度从近地面到数百米，并从多个相机视角捕获。
*   操作环境多样化，包括复杂的城市环境、开阔草地、茂密森林、湖面、无遮挡高空和山麓地形。
*   特别关注并保留了从信息丰富、清晰的事件到稀疏、噪声大的事件等各种质量水平的事件数据。

**数据处理与标注**:
*   **事件数据处理**: 将事件相机产生的原始异步事件流在20毫秒的固定时间窗口内累积，生成事件帧，以平衡运动模糊和信息密度。
*   **多模态对齐**: 鉴于传感器规格（分辨率、视场角）和物理布置的差异，采用基于特征点的配准技术将RGB和事件帧对齐到热红外模态的坐标系（热红外具有最高原始分辨率）。应用仿射变换矩阵，确保包含无人机的区域在三个模态间空间一致。采用“弱对齐”策略，即不强制标注边界框内无人机的严格像素级对应，旨在鼓励开发对轻微空间不一致性不敏感的鲁棒融合机制。
*   **数据标注**: 所有数据均使用LabelImg以YOLO格式进行标注。标注在通过对齐后的三模态数据进行像素级融合创建的图像上进行，这些标注可直接转移到配准后的RGB、IR和事件数据帧。

### 数据集统计
**表：Tri-Modal Anti-UAV 数据集关键统计**
| 属性                                  | 占比 (实例) |
|---------------------------------------|-----------------|
| 小目标 (例如，面积 < 8x8 像素)         | 7.96%           |
| 低光照/极端光照场景                     | 8.92%           |
| 高质量事件数据                          | 6.94%           |
| 复杂背景干扰                          | 23.2%           |

### 场景展示
数据集中包含了多种具有挑战性的场景。

**数据集样本概览:**
[查看数据集样本图像](https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/blob/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-49-42.png)
*图注：Tri-Modal Anti-UAV 数据集样本图像。目标用红色框标出。顶行：RGB模态；中间行：红外热成像模态；底行：事件模态，展示了多样性的事件数据质量 (对应论文 Figure 1)。*

**各种质量的事件数据示例:**
[查看各种质量的事件模态图](https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/blob/1415f24196421d9e56c68916e871d3a260d8debc/images/Screenshot2025-06-01_18-39-12.png)
*图注：事件模态数据质量的多样性展示，从左到右质量递减。*

**多样化拍摄场景 (以红外模态展示):**
[查看多种拍摄场景图（红外模态）](https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/blob/7821aac26e95be05f95116f40abd9d082f66017c/images/Screenshot2025-06-01_18-57-35.png)
*图注：数据集中多样化的拍摄场景（以红外模态展示部分样例）。*

**其他关键场景类型包括:**
*   高空小目标
*   弱光照环境下的无人机
*   复杂背景（如树枝、建筑物）干扰下的无人机
*   快速移动的无人机

### 数据下载
您可以从以下链接下载完整的数据集：
*   **Google Drive**: [https://drive.google.com/drive/folders/1t_oaJZuSyBd7W4oW93-T_WvN4_0cdBuE?usp=drive_link](https://drive.google.com/drive/folders/1t_oaJZuSyBd7W4oW93-T_WvN4_0cdBuE?usp=drive_link)

数据集采用YOLO标注格式。

## 🔧 图像对齐方式

由于不同传感器的固有差异（如分辨率和视场角）及其固定的非共处物理排列，我们采用基于特征点的配准技术，将RGB和事件帧与热红外模态的坐标系对齐。估算并应用仿射变换矩阵，主要确保包含无人机的区域在三个模态中空间一致。我们采用“弱对齐”策略，有意不强制标注边界框内无人机的严格像素级对应，以鼓励开发对微小空间不一致性更鲁棒的融合机制。

**对齐前图像示例 (RGB对齐红外，绿点为特征点):**
[查看对齐前图像](https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/blob/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-57-22.png)
*图注：对齐前图像（绿点表示对应特征点，这里以RGB对齐红外图像为例）。*

**对齐后图像示例:**
[查看对齐后图像](https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/blob/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-55-17.png)
*图注：对齐后的图像示例，展示了目标区域在不同模态间的空间一致性。*

对齐后的图像确保了目标区域在不同模态间的空间一致性，为后续的统一标注和有效多模态融合奠定了基础。

## 🚀 模型与预训练权重

### ATMF-Net
我们提出的 ATMF-Net (Adaptive Tri-Modal Fusion Network) 是一种有效融合三模态信息的网络结构，专为无人机检测设计。其核心思想是根据事件模态的实时可靠性动态评估和调整其在融合过程中的贡献，从而在事件数据质量波动时保持检测的鲁棒性。
*   **代码**: `./models/ATMF_Net/` (请替换为实际路径)
*   **预训练权重**: `[ATMF-Net权重下载链接或说明]` (例如: 可在 Release 页面找到)

**ATMF-Net 网络架构图:**
[查看ATMF-Net结构图](https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/blob/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-48-55.png)
*图注：ATMF-Net 网络架构。右侧：整体融合路径（以RGB特征为例）。左侧：三模态融合专家（Tri-Modal Fusion Expert）的详细信息。关键组件包括事件可靠性评估器（ERE）和用于动态专家权重调整的MoE路由器。$\oplus$: 特征相加, $\otimes$: 加权融合 (对应论文 Figure 2)。*

### LW-MoESGF (RGB+IR)
作为对比基线，我们还提供了 LW-MoESGF (Lightweight Mixture-of-Experts with Self-Guided Fusion) 模型的实现，这是一个高效的RGB-IR双模态融合模型。
*   **代码**: `./models/LW_MoESGF/` (请替换为实际路径)
*   **预训练权重**: `[LW-MoESGF权重下载链接或说明]`

**LW-MoESGF 网络架构图:**
[查看LW-MoESGF结构图](https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/blob/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-49-09.png)
*图注：LW-MoESGF 网络架构。左侧：双模态融合专家（Dual-Modal Fusion Expert）的详细信息，包括自引导融合（SGF）和细化模块。右侧：整体结构。$\oplus$: 特征相加, $\otimes$: 加权融合 (对应论文 Figure 3)。*

**Self-Guided Fusion (SGF) 模块结构图 (LW-MoESGF组件):**
[查看Self-Guided Fusion结构图](https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/blob/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_17-04-09.png)
*图注：Self-Guided Fusion (SGF) 模块的详细结构，它是 LW-MoESGF 中的一个关键组件。*


## 📊 主要结果

### 表格

**表1: 自适应三模态融合的有效性 (Effectiveness of adaptive tri-modal fusion)**
| 方法                                      | mAP$_{50}$ (%) |
|-------------------------------------------|----------------|
| LW-MoESGF (RGB+IR)                        | 87.4           |
| Tri-Modal (Non-adaptive)                  | 87.8           |
| **ATMF-Net (Adaptive)**                   | **89.9**       |

**表2: Tri-Modal Anti-UAV 测试集上单模态检测性能 (Performance of single-modality detection)**
| 模态                   | mAP$_{50}$ (%) | mAP (%) |
|------------------------|----------------|---------|
| YOLOv5l (RGB-only)     | 65.5           | 20.2    |
| YOLOv5l (IR-only)      | **78.8**       | **27.2**|
| YOLOv5l (Event-only)   | 9.76           | 3.57    |

**表3: RGB-IR 双模态融合方法的性能和效率比较 (Performance and efficiency comparison of RGB-IR dual-modal fusion methods)**
| 方法                      | 参数量 (M) | FLOPs (G) | mAP$_{50}$ (%) |
|---------------------------|------------|-----------|----------------|
| 最佳单模态 (IR)           | **46.5**   | **109**   | 78.8           |
| CFT (RGB+IR) [Li et al., 2021] | 206        | 224       | 86.6           |
| LW-MoESGF (RGB+IR)        | 76.2       | 192       | **87.4**       |
*CFT citation: Qingyun Li, Filepe R. C. Encarnacao, and Aljosa Osep. 2021. Cross-modality Feature Transformer for Unsupervised Object Tracking. arXiv:2112.02009.*

### 图示

#### 检测结果定性对比
**事件模态效果与检测结果对比图:**
[查看定性结果比较图](https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/blob/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-49-24.png)
*图注：Tri-Modal Anti-UAV 上的定性比较。从左到右：RGB、IR 和事件输入。图像上叠加显示：真实标签 (Ground Truth, 红色框)，LW-MoESGF (RGB+IR, 绿色框) 的检测结果，以及我们的 ATMF-Net (蓝色框) 的检测结果 (对应论文 Figure 4)。*

## 🛠️ 安装

### 环境要求
*   Python >= 3.8
*   PyTorch >= 1.7.0
*   CUDA [例如：10.2 / 11.1 / 11.3] (如果使用GPU)
*   其他依赖请参见 `requirements.txt`

### 安装步骤

1.  **克隆本仓库:**
    ```bash
    git clone https://github.com/eulerbaby123/Tri-Modal-Anti-UAV.git
    cd Tri-Modal-Anti-UAV
    ```

2.  **创建并激活虚拟环境 (推荐):**
    ```bash
    python -m venv venv
    # Windows
    # venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```
    如果遇到 `pycocotools` 安装问题，请参考其官方文档进行安装。对于 Windows 用户，可能需要预先安装 Microsoft C++ Build Tools。

### 依赖库
本项目主要依赖以下库 (完整列表请见 `requirements.txt`):
*   `torch`
*   `torchvision`
*   `numpy`
*   `opencv-python`
*   `matplotlib`
*   `pyyaml`
*   `tqdm`
*   `pycocotools` (用于评估)

## ⚙️ 使用

### 数据准备
1.  下载 Tri-Modal Anti-UAV 数据集 (链接见 [数据下载](#数据下载) 部分)。
2.  将数据集解压并组织成如下结构 (或根据您的配置文件进行调整):
    ```
    Tri-Modal-Anti-UAV/
    ├── images/
    │   ├── train/
    │   │   ├── rgb/      # RGB 图像
    │   │   ├── ir/       # 红外图像
    │   │   └── event/    # 事件帧图像
    │   └── val/
    │       ├── rgb/
    │       ├── ir/
    │       └── event/
    ├── labels/
    │   ├── train/      # YOLO 格式标签 (.txt)
    │   └── val/
    └── dataset.yaml    # 数据集配置文件
    ```
3.  确保 `dataset.yaml` (或类似配置文件) 中的路径正确指向您的数据集位置。

### 训练
使用以下命令开始训练 (请根据您的实际训练脚本和参数进行调整):
```bash
python train.py --cfg models/yolov5l_atmf.yaml --data data/uav_rgbte.yaml --weights yolov5l.pt --batch-size 8 --epochs 100 --device 0
