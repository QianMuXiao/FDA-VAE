# Flip Distribution Alignment VAE for Multi-Phase MRI Synthesis

## 概述

这个项目是论文《Flip Distribution Alignment VAE for Multi-Phase MRI Synthesis》的官方实现代码，该论文已被MICCAI 2025提前接收。FDA-VAE是一个基于变分自编码器的跨模态医学影像合成框架，用于多期对比增强MRI影像的合成任务。该框架通过在潜在空间中执行分布翻转对齐（Flipped Distribution Alignment）约束，实现了高质量的跨期MRI影像合成，在获得更佳的合成质量的同时参数量仅为现有先进方法的约十分之一。

## 特点

- **轻量化的编解码器结构**：相比层数和宽度相对较低所需推理参数极少
- **分布翻转机制**：在潜空间通过对分布进行翻转实现跨模态转换
- **双解码器架构**：用于实现多模态影像的双向合成确保训练稳定性
- **共享编码器**：对不同期的影像使用一个共享编码器，减少参数量并获得相对统一的表征

## 安装要求

```bash
# 安装必要的库
pip install torch torchvision
pip install monai
pip install monai-generative
pip install SimpleITK
pip install scipy
pip install tqdm
pip install numpy
pip install tensorboardX
pip install pytorch_msssim
pip install lpips
# 可选：安装xformers以加速注意力机制计算（需要CUDA支持）
```

## 数据准备

原始数据集链接：https://github.com/LMMMEng/LLD-MMRI-Dataset

数据集应具有以下结构：

```
data_root/
├── phase_1/   
│   ├── patient1_1.nii.gz
│   ├── patient1_2.nii.gz
│   └── ...
├── phase_2/   
│   ├── patient1_1.nii.gz
│   ├── patient1_2.nii.gz
│   └── ...
├── phase_3/ 
│   ├── patient1_1.nii.gz
│   ├── patient1_2.nii.gz
│   └── ...
```

注意：patient的id需要和提供的lesion_patient_list.txt文件中患者的id对应，因为需要根据数据集中多种病灶的比例进行划分训练验证集。

PS：你也可以随便按照你的喜好和数据集处理以后的结构写一个Dataset类来用于训练。

## 使用方法

### 训练模型

```bash
PYTHONWARNINGS="ignore" torchrun --nproc_per_node=your_gpu_number train.py
```

## 方法概述

### 变分自编码器结构

生成器基于MONAI generative的AutoencoderKL框架，进行了以下修改以适应跨模态翻译任务：

1. 使用单个编码器将不同模态的图像编码到共享潜在空间
2. 实现了基于分布反转的模态转换机制
3. 为每个模态设计专用解码器

### 分布翻转机制

FDA-VAE核心创新是在潜在空间中的分布翻转操作。当图像被编码到潜在空间后，我们通过对均值向量取反，保持标准差不变的方式，得到另一个模态的潜在表示。这种简单而有效的方法构建了一个结构化的潜在空间，使得成对的切片的潜在分布特征在保留重叠部分的前提下最大化非重叠区域，且相对关系可控。

## 引用

如果您使用了本代码，请引用我们的工作：

```
@inproceedings{kui2025fdavae,
  title={Flip Distribution Alignment VAE for Multi-Phase MRI Synthesis},
  author={Kui, Xiaoyan and Xiao, Qianmu and Li, Qinsong and Ji, Zexin and Zhang, Jielin and Zou, Beiji},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2025},
  publisher={Springer},
  pages={},  % 页码将在出版后确定
  doi={},    % DOI将在出版后确定
  url={https://github.com/QianMuXiao/FDA-VAE}
}
```


