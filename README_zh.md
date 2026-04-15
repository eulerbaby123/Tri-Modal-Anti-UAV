<p align="right">
  <a href="./README.md">English</a> | <a href="./README_zh.md">中文</a>
</p>

# Tri-Modal Anti-UAV 数据集与 ATMF-Net（多模态无人机检测系统，可见光+红外+Event）

[![Dataset](https://img.shields.io/badge/Dataset-Download-blue.svg)](https://drive.google.com/drive/folders/1t_oaJZuSyBd7W4oW93-T_WvN4_0cdBuE?usp=drive_link)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue.svg)](https://github.com/eulerbaby123/Tri-Modal-Anti-UAV)

欢迎使用 **Tri-Modal Anti-UAV** 项目。本仓库包含论文 **“Adaptive Tri-Modal Fusion for Robust Anti-UAV Detection with Event Fluctuation Awareness”** 的官方代码与数据集。

本项目旨在提供一个完整的三模态（RGB、Thermal、Event）反无人机检测基准数据集 **Tri-Modal Anti-UAV**，并提出一种有效的无人机目标检测自适应融合网络 **ATMF-Net**。

## ⚖️ 伦理、隐私与许可说明

本数据集中的全部数据均采集于公共区域或受控区域，不涉及个人隐私信息。图像中如出现人员，均已进行模糊化或其他去标识化处理。项目中提供的所有代码、数据集及相关资源，默认允许用于科学研究、教育或个人实验等非商业用途，无需额外获得作者授权。商业用途请联系作者。

## 目录

- [Tri-Modal Anti-UAV 数据集与 ATMF-Net](#tri-modal-anti-uav-数据集与-atmf-net多模态无人机检测系统可见光红外event)
  - [⚖️ 伦理、隐私与许可说明](#️-伦理隐私与许可说明)
  - [目录](#目录)
  - [📝 项目简介](#-项目简介)
  - [📸 数据集：Tri-Modal Anti-UAV](#-数据集tri-modal-anti-uav)
    - [数据概览](#数据概览)
    - [数据采集与处理](#数据采集与处理)
    - [数据集统计](#数据集统计)
    - [场景展示](#场景展示)
    - [数据下载](#数据下载)
  - [🔧 图像对齐](#-图像对齐)
  - [🚀 模型与权重](#-模型与权重)
    - [ATMF-Net](#atmf-net)
    - [LW-MoESGF（RGB+IR）](#lw-moesgfrgbir)
    - [其他实验资源](#其他实验资源)
  - [📊 主要结果](#-主要结果)
    - [表格](#表格)
    - [图示](#图示)
    - [数据集补充说明](#数据集补充说明)
  - [🛠️ 安装](#️-安装)
    - [环境要求](#环境要求)
    - [安装步骤](#安装步骤)
    - [关键依赖](#关键依赖)
  - [⚙️ 使用方法](#️-使用方法)
    - [数据准备](#数据准备)
    - [训练](#训练)
    - [评估](#评估)

## 📝 项目简介

无人机（UAV）的广泛应用使得高鲁棒性的反无人机检测系统变得必要。尽管多模态融合（例如 RGB-Thermal）能够提升系统鲁棒性，但在运动模糊、低对比度等极端场景下，性能瓶颈仍然存在。事件相机具有高动态范围和高时间分辨率，但其数据质量会天然波动，而现有数据集尚未对这种现象进行系统性刻画。为弥补这一空白，我们提出 **Tri-Modal Anti-UAV**：首个专门面向反无人机研究的三模态（RGB、Thermal、Event）数据集。该数据集包含 1,060 组同步图像三元组，覆盖多种复杂场景（例如高空微小目标、弱光、环境背景干扰），并特别强调保留从稠密清晰到稀疏/噪声的完整事件数据质量谱系。

基于该基准数据集，我们提出 **ATMF-Net**，一种自适应三模态融合网络，可根据事件模态的实时可靠性动态调节其在融合过程中的贡献。该轻量级架构集成了 Mixture-of-Experts 框架与 Self-Guided Fusion，在保持高效率的同时优于非自适应融合方法。系统性基准测试验证了 **Tri-Modal Anti-UAV** 的挑战性：仅使用事件模态进行检测时性能很差（9.76% mAP50），而自适应三模态融合可将精度提升至 89.9% mAP50。该数据集为开发具备事件感知能力、更加鲁棒的反无人机检测器提供了关键资源。

**关键特性：**
*   首个专门面向反无人机检测的三模态数据集（RGB、热红外、事件相机）。
*   系统性采集并保留不同质量水平的事件数据流，从稠密清晰到稀疏噪声，能够更真实地反映实际应用场景，并为开发事件感知型、鲁棒型检测算法提供关键资源。
*   包含 1,060 组同步图像三元组，覆盖多种复杂场景（如高空小目标、低照度、背景干扰）和多种无人机类型。
*   提出 ATMF-Net，一种可根据事件数据实时可靠性动态调整其贡献的自适应三模态融合网络。
*   提供 ATMF-Net、LW-MoESGF（双模态基线）以及其他模型的实现。
*   提供详细的评估指标与实验结果，建立强基线。

## 📸 数据集：Tri-Modal Anti-UAV

### 数据概览
**Tri-Modal Anti-UAV** 数据集是一个专为反无人机研究构建的新型三模态基准，包含同步的可见光（RGB）、热红外（T）和事件（E）数据流。

### 数据采集与处理
**数据采集：**
*   数据集共包含 1,060 组带标注的图像样本（855 组训练，205 组测试）。
*   各模态数据分别由专用传感器采集：传统 RGB 相机、热红外相机和 DAVIS346 事件传感器。
*   无人机平台包括 DJI Mini 3 和 DJI Mavic 3 Pro，每个场景中的无人机数量为 1 到 3 架。
*   数据采集覆盖多种环境条件：天气变化（晴天到阴天）、一天中的不同时段。无人机飞行状态多样，飞行高度从近地面到数百米不等，并由多个相机视角进行采集。
*   采集环境多样，包括复杂城市环境、开阔草地、茂密森林、湖面、无遮挡高空以及山脚地形等。
*   特别关注保留不同质量水平的事件数据，从信息丰富、清晰的事件数据到稀疏、噪声较大的数据。

**数据处理与标注：**
*   **事件数据处理**：将事件相机输出的原始异步事件流按固定的 20ms 时间窗进行累积，生成事件帧，以平衡运动模糊和信息密度。
*   **多模态对齐**：不同传感器的原始分辨率不同：RGB 相机为 40x360 像素，热红外相机为 640x512 像素，事件相机（DAVIS346）为 346x260 像素。考虑到这些传感器在分辨率、视场角及物理安装位置上的差异，项目采用基于特征点的配准方法，将 RGB 与事件帧对齐到热模态坐标系下（热模态在文中给出的分辨率中最高，但仍建议再次核对 RGB 的 40x360 是否为笔误或特定裁剪结果）。通过仿射变换矩阵，确保包含 UAV 的区域在三个模态下尽可能保持空间一致。项目采用“弱对齐（weak alignment）”策略，即不强制要求 UAV 在边界框内部达到严格像素级对应，同时背景区域也允许弱对齐，以鼓励开发对轻微空间不一致不敏感的鲁棒融合机制。
*   **数据标注**：所有数据均使用 LabelImg 以 YOLO 格式进行标注。标注是在对齐后的三模态像素级融合图像上完成的，这些标注可以直接迁移到已配准的 RGB、IR 和 Event 帧中。

### 数据集统计
**表：Tri-Modal Anti-UAV 数据集关键统计**
| 属性 | 占比（实例） |
|---------------------------------------|------------------------|
| 小目标（例如面积 < 8x8 像素） | 7.96% |
| 弱光/极端光照场景 | 8.92% |
| 高质量事件数据 | 6.94% |
| 复杂背景干扰 | 23.2% |

### 场景展示
该数据集包含多种具有挑战性的场景。

**数据集样例概览：**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-49-42.png?raw=true" width="1000" alt="Dataset Samples">
  <br/><em>图：Tri-Modal Anti-UAV 数据集样例图像。目标以红框标出。第一行：RGB 模态；第二行：热红外模态；第三行：事件模态，展示了事件数据质量的多样性（对应论文图 1）。</em>
</div>

**不同质量事件数据示例：**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/1415f24196421d9e56c68916e871d3a260d8debc/images/Screenshot2025-06-01_18-39-12.png?raw=true" width="1000" alt="Event Data Quality Examples">
  <br/><em>图：事件模态数据质量多样性的展示。数据集中包含不同质量的数据，包括噪声较大和信息不足的情况。</em>
</div>

**多样化采集场景（以下以红外模态展示）：**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/7821aac26e95be05f95116f40abd9d082f66017c/images/Screenshot2025-06-01_18-57-35.png?raw=true" width="1000" alt="Diverse Scenes IR">
  <br/><em>图：数据集中的多样化采集场景（部分示例以红外模态展示）。</em>
</div>

**其他关键场景类型包括：**
*   高空小目标
*   弱光环境下的无人机
*   复杂背景干扰下的无人机（例如树枝、建筑物）
*   高速运动无人机

### 数据下载
可通过以下链接下载完整数据集：
*   **Google Drive**: [https://drive.google.com/drive/folders/1t_oaJZuSyBd7W4oW93-T_WvN4_0cdBuE?usp=drive_link](https://drive.google.com/drive/folders/1t_oaJZuSyBd7W4oW93-T_WvN4_0cdBuE?usp=drive_link)

数据集采用 YOLO 标注格式。

## 🔧 图像对齐

由于不同传感器在分辨率、视场角以及固定但非共址的物理安装位置上存在天然差异，我们采用基于特征点的配准技术，将 RGB 和事件帧对齐到热红外模态的坐标系中。通过估计得到的仿射变换矩阵，主要保证包含 UAV 的区域在三个模态之间具备空间一致性。我们采用“弱对齐”策略，即有意不强制要求已标注边界框内的 UAV 严格像素级对应，以鼓励开发对轻微空间不一致更鲁棒的融合机制。

**对齐前图像示例（RGB 对齐到红外，绿色点为特征点）：**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-57-22.png?raw=true" width="800" alt="Image Alignment Before">
  <br/><em>图：对齐前图像示例（绿色点表示对应特征点，此处以 RGB 到红外对齐为例）。</em>
</div>

**对齐后图像示例：**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-55-17.png?raw=true" width="800" alt="Image Alignment After">
  <br/><em>图：对齐后图像示例，展示了目标区域在不同模态间的空间一致性。</em>
</div>

对齐后的图像保证了不同模态间目标区域的空间一致性，为后续统一标注和有效多模态融合奠定基础。

## 🚀 模型与权重

### ATMF-Net
我们提出的 **ATMF-Net（Adaptive Tri-Modal Fusion Network）** 是一种用于三模态信息融合的有效网络结构，专门面向 UAV 检测设计。其核心思想是在融合过程中根据事件模态的实时可靠性，动态评估并调整事件模态的贡献，从而在事件数据质量波动时仍保持检测鲁棒性。
*   **代码**：`./models/ATMF_Net/`（如实际路径不同，请替换为正确路径）
*   **最佳训练权重（`best.pt`）GDrive**： [https://drive.google.com/file/d/1xsx8g-1wAIUPylxw0jj6pXMck-VM_JX7/view?usp=drive_link](https://drive.google.com/file/d/1xsx8g-1wAIUPylxw0jj6pXMck-VM_JX7/view?usp=drive_link)
*   **初始骨干网络权重**：训练通常从标准预训练骨干权重开始（例如基于 YOLOv5-Large 的模型可使用 `yolov5l.pt`）。这些权重通常来自原始 YOLOv5 仓库或类似来源。

**ATMF-Net 网络结构图：**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-48-55.png?raw=true" width="1200" alt="ATMF-Net Architecture">
  <br/><em>图：ATMF-Net 网络结构。右侧：整体融合路径（以 RGB 特征为例）。左侧：三模态融合专家模块细节。关键组件包括事件可靠性估计器（ERE）和用于动态专家权重调节的 MoE 路由器。$\oplus$：特征相加，$\otimes$：加权融合（对应论文图 2）。</em>
</div>

**Self-Guided Fusion（SGF）模块结构图（ATMF-Net 组成部分）：**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_17-04-09.png?raw=true" width="400" alt="Self-Guided Fusion Architecture">
  <br/><em>图：Self-Guided Fusion（SGF）模块的详细结构，这是 ATMF-Net 的关键组成部分。</em>
</div>

### Other Experimental Resources
*   **Other Paper-related Experimental Code and Weights GDrive**: [https://drive.google.com/file/d/1WDaYFGmbvIM_oK0p7rGdpjXrhATFc6l2/view?usp=drive_link](https://drive.google.com/file/d/1WDaYFGmbvIM_oK0p7rGdpjXrhATFc6l2/view?usp=drive_link)
    *   This link contains other model codes and/or pre-trained weights used for comparative experiments or ablation studies in the paper. Please follow the instructions within the archive.

## 📊 主要结果

### 表格

**表 1：自适应三模态融合的有效性**
| 方法 | mAP<sub>50</sub> (%) |
|-------------------------------------------|----------------|
| LW-MoESGF (RGB+IR) | 87.4 |
| Tri-Modal（非自适应） | 87.8 |
| **ATMF-Net（自适应）** | **89.9** |

**表 2：Tri-Modal Anti-UAV 测试集上单模态检测性能**
| 模态 | mAP<sub>50</sub> (%) | mAP (%) |
|------------------------|----------------|---------|
| YOLOv5l（仅 RGB） | 65.5 | 20.2 |
| YOLOv5l（仅 IR） | **78.8** | **27.2** |
| YOLOv5l（仅 Event） | 9.76 | 3.57 |

**表 3：RGB-IR 双模态融合方法的性能与效率比较**
| 方法 | 参数量 (M) | FLOPs (G) | mAP<sub>50</sub> (%) |
|---------------------------|----------------|-----------|----------------|
| 最佳单模态（IR） | **46.5** | **109** | 78.8 |
| CFT (RGB+IR) | 206 | 224 | 86.6 |
| LW-MoESGF (RGB+IR) | 76.2 | 192 | **87.4** |

### 图示

#### 检测结果定性对比
**事件模态效果与检测结果对比：**
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/2de41951e962a9cff3eb1c2849c3d051f70fc087/images/Screenshot2025-06-01_16-49-24.png?raw=true" width="600" alt="Qualitative Results Comparison">
  <br/><em>图：Tri-Modal Anti-UAV 上的定性比较。从左到右分别为 RGB、IR 和 Event 输入。图中叠加显示：真实标注（红框）、LW-MoESGF（RGB+IR，绿框）的检测结果，以及我们的 ATMF-Net（蓝框）检测结果（对应论文图 4）。</em>
</div>

### 数据集补充说明
需要说明的是，在该数据集中，少量图像由于近距离采集场景、传感器固有视角差异以及像素分辨率差异等因素，在经过仿射变换对齐后，其边缘区域可能会出现无像素信息区域（统一填充为黑色或白色）。同时，仿射变换对齐方法本身也存在固有限制，因此对齐前后图像在视觉上可能存在一定差异。

我们选择保留这些样本而不是将其丢弃，原因如下：
1.  **真实场景模拟**：这种情况在实际多传感器融合应用中可能出现，保留它们有助于模型学习如何处理这种不完美对齐。
2.  **鲁棒性增强**：这些由对齐过程引入、但内容信息基本不变的视觉差异，可以视为一种数据增强或干扰。我们认为这反而能够促使模型学习更本质、更鲁棒的特征，从而提升其在复杂真实环境中的泛化能力。

下图展示了一个带有边缘填充的对齐图像示例：
<div align="center">
  <img src="https://github.com/eulerbaby123/Tri-Modal-Anti-UAV/raw/34fabfae1f61173924738877dea5e85addc5423b/images/Screenshot2025-06-01_19-31-01.png?raw=true" width="400" alt="Alignment Artifact Example">
  <br/><em>图：对齐图像示例，其边缘区域可能缺失像素信息（黑色填充）。</em>
</div>

## 🛠️ 安装

### 环境要求
*   Python >= 3.8
*   PyTorch >= 1.7.0
*   CUDA [11.3]（如使用 GPU）
*   其他依赖见 `requirements.txt`

### 安装步骤

1.  **克隆仓库：**
    ```bash
    git clone https://github.com/eulerbaby123/Tri-Modal-Anti-UAV.git
    cd Tri-Modal-Anti-UAV
    ```

2.  **创建并激活虚拟环境：**
    ```bash
    python -m venv venv
    # Windows
    # venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **安装依赖：**
    ```bash
    pip install -r requirements.txt
    ```
    如果安装 `pycocotools` 时遇到问题，请参考其官方文档。Windows 用户可能需要先安装 Microsoft C++ Build Tools。

## ⚙️ 使用方法

### 数据准备
1. 下载 Tri-Modal Anti-UAV 数据集（链接见 [数据下载](#数据下载) 部分）。
2. 解压数据集，并按如下结构组织（或根据你的配置文件进行调整）：
    ```

    Tri-Modal-Anti-UAV/
    ├── images/
    │  
    ├── labels/
    │  
    └──train_rgb.txt
    └──train_ir.txt
    └──train_event.txt
    └──val_rgb.txt
    └──val_ir.txt
    └──val_event.txt
    ```

### 测试或训练
使用以下命令开始测试或训练（参数可根据实际需求在脚本文件中修改）：
```bash
python test.py
python train.py 
# Check train.py or associated config files for parameters like model config, data config, weights, batch size, epochs, device, etc.
