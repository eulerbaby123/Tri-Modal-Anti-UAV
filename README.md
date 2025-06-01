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
    - [场景展示](#场景展示)
    - [数据下载](#数据下载)
  - [🔧 图像对齐方式](#-图像对齐方式)
  - [🚀 模型与预训练权重](#-模型与预训练权重)
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

[请在此处插入论文的摘要或项目简介，简要介绍研究背景、目的、方法和主要贡献。]

**主要特性:**
*   首个针对反无人机检测的大型三模态数据集（RGB、红外热成像、事件相机数据）。
*   包含多种复杂场景和无人机类型。
*   提供了 ATMF-Net 等基线模型的实现。
*   详细的评估指标和结果。

## 📸 数据集

### 数据概览
我们的数据集 **"Tri-Modal Anti-UAV Dataset"** 是专门为无人机目标检测任务构建的。它包含了同步采集的 RGB 图像、红外热成像图像以及事件相机数据。

### 场景展示
数据集中包含了多种具有挑战性的场景，例如：
*   场景1：[请描述场景1，例如：城市背景下的低空飞行无人机]
*   场景2：[请描述场景2，例如：复杂天空背景下的高速移动无人机]
*   场景3：[请描述场景3，例如：夜间或弱光条件下的无人机]
*   ... [请列出并描述所有关键场景，可以考虑使用小型示例图片或GIF]

### 数据下载
您可以从以下链接下载完整的数据集：
*   **Google Drive**: [https://drive.google.com/drive/folders/1t_oaJZuSyBd7W4oW93-T_WvN4_0cdBuE?usp=drive_link](https://drive.google.com/drive/folders/1t_oaJZuSyBd7W4oW93-T_WvN4_0cdBuE?usp=drive_link)

[如果数据集有特定的组织结构或标注格式说明，请在此处添加。]

## 🔧 图像对齐方式

我们采用了 [请简要描述对齐方法] 的方式来确保不同模态图像之间的空间对齐。下图展示了我们的对齐流程/效果：

[请在此处插入图像对齐的示意图，例如：]
<!-- ![Image Alignment](path/to/your/alignment_figure.png) -->
**图注：** [请描述上图内容，例如：RGB、热红外和事件数据帧的对齐示例。]

## 🚀 模型与预训练权重

### ATMF-Net
我们提出的 ATMF-Net 是一种有效融合三模态信息的网络结构，专为无人机检测设计。
*   **代码**: [请提供 ATMF-Net 代码的链接，例如：`./models/ATMF-Net/`]
*   **预训练权重**: [请提供 ATMF-Net 权重文件的下载链接1]

### 其他模型 (可选)
[如果您在论文中对比了其他模型，或者提供了其他模型的实现，请在此处列出。]
*   **模型名称**: [例如：Baseline CNN]
    *   **代码**: [请提供该模型代码的链接]
    *   **预训练权重**: [请提供该模型权重文件的下载链接2]

## 📊 主要结果

[请在此处重新展示您论文中的主要数据表格和图。您可以使用 Markdown 表格，或者截图后插入图片。]

### 表格

**表1: [表格标题，例如：不同模型在 uavRGBTE 数据集上的性能对比]**
| 模型        | mAP@0.5 | Precision | Recall | F1-Score | FPS | FLOPs (G) |
|-------------|---------|-----------|--------|----------|-----|-----------|
| ATMF-Net    | [值]    | [值]      | [值]   | [值]     | [值]| [值]      |
| 模型 A      | [值]    | [值]      | [值]   | [值]     | [值]| [值]      |
| ...         | ...     | ...       | ...    | ...      | ... | ...       |

[可以添加更多表格]

### 图示

**图1: [图标题，例如：PR 曲线对比]**
<!-- ![PR Curve](path/to/your/pr_curve_figure.png) -->
**图注：** [请描述上图内容]

[可以添加更多图示，例如检测结果的可视化示例]

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
