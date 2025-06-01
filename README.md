# uavRGBTE: Tri-Modal Anti-UAV Dataset and ATMF-Net

[![Paper](https://img.shields.io/badge/Paper-ARXIV_LINK_OR_DOI-B31B1B.svg)]([请在此处插入论文链接])
[![Dataset](https://img.shields.io/badge/Dataset-Download-blue.svg)](https://drive.google.com/drive/folders/1t_oaJZuSyBd7W4oW93-T_WvN4_0cdBuE?usp=drive_link)

欢迎来到 **uavRGBTE** 项目！这是论文 **"Tri-Modal Anti-UAV: A Comprehensive Benchmarking Dataset for UAV-Targeted Detection"** 的官方代码和数据集。

本项目旨在提供一个全面的三模态（RGB、Thermal、Event）反无人机检测基准数据集，并提出了一种有效的融合网络 ATMF-Net 用于无人机目标检测。

## 目录

- [uavRGBTE: Tri-Modal Anti-UAV Dataset and ATMF-Net](#uavrgbte-tri-modal-anti-uav-dataset-and-atmf-net)
  - [目录](#目录)
  - [📝 简介](#-简介)
  - [📸 数据集](#-数据集)
    - [数据概览](#数据概览)
    - [数据集统计](#数据集统计)
    - [场景展示](#场景展示)
    - [数据下载](#数据下载)
  - [🔧 图像对齐方式](#-图像对齐方式)
  - [🚀 模型与权重](#-模型与权重)
    - [ATMF-Net](#atmf-net)
    - [其他模型 (可选)](#其他模型-可选)
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
*   保留各种质量的事件模态，更加贴近实际，也为事件模态平衡方法的开发提供了基础。
*   包含多种复杂场景和无人机类型。
*   提供了 ATMF-Net 等基线模型的实现。
*   详细的评估指标和结果。

## 📸 数据集

### 数据概览
我们的数据集 **"Tri-Modal Anti-UAV Dataset"** 是专门为无人机目标检测任务构建的。它包含了同步采集的 RGB 图像、红外热成像图像以及事件相机数据，共计 1,060 组同步图像三元组。

### 数据集统计
**表：Tri-Modal Anti-UAV 数据集关键统计**
| 属性                                  | 占比 (实例) |
|---------------------------------------|-----------------|
| 小目标 (例如，面积 < 8x8 像素)         | 7.96%           |
| 低光照/极端光照场景                     | 8.92%           |
| 高质量事件数据                          | 6.94%           |
| 复杂背景干扰                          | 23.2%           |

### 场景展示
数据集中包含了多种具有挑战性的场景。下图展示了部分样本：

<!-- 请将 datademo.pdf 转换为 .png 或 .jpg格式, 存放到例如 assets/images/ 目录下, 并更新下面的路径 -->
![Dataset Samples](assets/images/datademo.png)
**图注：** Tri-Modal Anti-UAV 数据集样本图像。目标用红色框标出。顶行：RGB模态；中间行：红外热成像模态；底行：事件模态，展示了多样性的事件数据质量。

[您可以继续列出并描述其他关键场景，例如：]
*   场景1：高空小目标
*   场景2：弱光照环境
*   场景3：复杂背景干扰下的无人机
*   ...

### 数据下载
您可以从以下链接下载完整的数据集：
*   **Google Drive**: [https://drive.google.com/drive/folders/1t_oaJZuSyBd7W4oW93-T_WvN4_0cdBuE?usp=drive_link](https://drive.google.com/drive/folders/1t_oaJZuSyBd7W4oW93-T_WvN4_0cdBuE?usp=drive_link)

数据集采用YOLO标注格式。
[如果数据集有特定的组织结构或更详细的标注格式说明，请在此处添加。]

## 🔧 图像对齐方式

我们采用了 [基于特征点的仿射变换] 的方式来确保不同模态图像之间的目标弱对齐。下图展示了我们的对齐流程/效果：

<!-- 请在此处插入图像对齐的示意图，例如：将其保存为 alignment_diagram.png 到 assets/images/ 目录 -->
<!-- ![Image Alignment](assets/images/alignment_diagram.png) -->
**图注：** [请描述图像对齐示意图内容，例如：RGB、热红外和事件数据帧的对齐示例。]

## 🚀 模型与预训练权重

### ATMF-Net
我们提出的 ATMF-Net 是一种有效融合三模态信息的网络结构，专为无人机检测设计。其核心思想是根据事件模态的实时可靠性动态调整其贡献。
*   **代码**: [请提供 ATMF-Net 代码的链接，例如：`./models/ATMF-Net/`]
*   **预训练权重**: [请提供 ATMF-Net 权重文件的下载链接1]

下图展示了 ATMF-Net 的网络架构：
<!-- 请将 trimodal.pdf 转换为 .png 或 .jpg格式, 存放到例如 assets/images/ 目录下, 并更新下面的路径 -->
![ATMF-Net Architecture](assets/images/trimodal_arch.png)
**图注：** ATMF-Net 网络架构。右侧：整体融合路径（以RGB特征为例）。左侧：三模态融合专家（Tri-Modal Fusion Expert）的详细信息。关键组件包括事件可靠性评估器（ERE）和用于动态专家权重调整的MoE路由器。$\oplus$: 特征相加, $\otimes$: 加权融合。

### 其他模型 (可选)
[如果您在论文中对比了其他模型，或者提供了其他模型的实现，请在此处列出。例如 LW-MoESGF 模型。]

**模型名称**: LW-MoESGF (RGB+IR)
*   **代码**: [请提供 LW-MoESGF 模型代码的链接]
*   **预训练权重**: [请提供 LW-MoESGF 模型权重文件的下载链接2]

下图展示了 LW-MoESGF 的网络架构：
<!-- 请将 dualmodal.pdf 转换为 .png 或 .jpg格式, 存放到例如 assets/images/ 目录下, 并更新下面的路径 -->
![LW-MoESGF Architecture](assets/images/dualmodal_arch.png)
**图注：** LW-MoESGF 网络架构。左侧：双模态融合专家（Dual-Modal Fusion Expert）的详细信息，包括自引导融合（SGF）和细化模块。右侧：整体结构。$\oplus$: 特征相加, $\otimes$: 加权融合。

## 📊 主要结果

### 表格

**表1: 自适应三模态融合的有效性**
| 方法                                      | mAP$_{50}$ (%) |
|-------------------------------------------|----------------|
| LW-MoESGF (RGB+IR)                        | 87.4           |
| Tri-Modal (Non-adaptive)                  | 87.8           |
| **ATMF-Net (Adaptive)**                   | **89.9**       |

**表2: Tri-Modal Anti-UAV 测试集上单模态检测性能**
| 模态                   | mAP$_{50}$ (%) | mAP (%) |
|------------------------|----------------|---------|
| YOLOv5l (RGB-only)     | 65.5           | 20.2    |
| YOLOv5l (IR-only)      | **78.8**       | **27.2**|
| YOLOv5l (Event-only)   | 9.76           | 3.57    |

**表3: RGB-IR 双模态融合方法的性能和效率比较**
| 方法                      | 参数量 (M) | FLOPs (G) | mAP$_{50}$ (%) |
|---------------------------|------------|-----------|----------------|
| 最佳单模态 (IR)           | **46.5**   | **109**   | 78.8           |
| CFT (RGB+IR) [qingyun2021cross] | 206        | 224       | 86.6           |
| LW-MoESGF (RGB+IR)        | 76.2       | 192       | **87.4**       |

### 图示

#### 事件模态效果与检测结果对比
下图定性比较了不同方法在 Tri-Modal Anti-UAV 数据集上的检测效果。
<!-- 请确保 result.png 存放到例如 assets/images/ 目录下, 并更新下面的路径 -->
![Qualitative Results](assets/images/result.png)
**图注：** Tri-Modal Anti-UAV 上的定性比较。从左到右：RGB、IR 和事件输入。图像上叠加显示：真实标签 (Ground Truth, 红色框)，LW-MoESGF (RGB+IR, 绿色框) 的检测结果，以及我们的 ATMF-Net (蓝色框) 的检测结果。

[可以添加更多图示，例如 PR 曲线等。]

## 🛠️ 安装

### 环境要求
*   Python >= 3.8
*   PyTorch >= 1.7.0
*   CUDA [您的CUDA版本，例如：10.2 / 11.1] (如果使用GPU)
*   其他依赖请参见 `requirements.txt`

### 安装步骤

1.  **克隆本仓库:**
    ```bash
    git clone [您的仓库HTTPS链接]
    cd uavRGBTE
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
