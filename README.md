# 口腔鳞状细胞癌（OSCCC）检测系统

基于《基于摄影图像的口腔鳞状细胞癌检测深度学习算法：一项回顾性研究》的核心思路，开发的一套口腔鳞状细胞癌（OCSCC）图像检测 AI 程序。

## 项目概述

本项目实现了一个级联卷积神经网络，用于检测和分类口腔鳞状细胞癌。模型采用两阶段方法：
1. 检测网络：用于粗定位疑似病变区域
2. 分类网络：基于检测结果进行精细分类

## 功能特点

- 图像预处理和数据增强
- 级联网络架构（检测+分类）
- 支持迁移学习（使用ImageNet预训练的DenseNet121）
- 权重平衡的损失函数
- 完整的训练和验证流程
- 性能评估（准确率、敏感性、特异性、AUC）
- ROC曲线可视化
- 推理脚本支持单张图像预测

## 环境依赖

- Python 3.8+
- PyTorch 1.10+
- OpenCV-Python
- Pandas
- Matplotlib
- Scikit-learn
- Numpy

## 安装说明

```bash
pip install -r requirements.txt
```

## 数据格式

需要准备以下CSV文件：

1. `train_data.csv` - 训练数据
2. `val_data.csv` - 验证数据

CSV文件格式：
```
image_path,label
./data/ocsccc_001.jpg,1
./data/ocsccc_002.jpg,0
...
```

字段说明：
- `image_path`: 图像文件路径
- `label`: 标签（1表示患病，0表示正常）

## 使用方法

### 训练模型

```bash
python main.py
```

训练完成后将生成以下文件：
- `best_ocsccc_model.pth` - 最佳模型权重
- `train_log.csv` - 训练日志
- `val_roc_curve.png` - ROC曲线图

### 推理预测

```bash
python infer.py --image_path path/to/your/image.jpg
```

## 模型架构

### 检测网络
- 轻量级目标检测网络简化版
- 输入原始图像
- 输出疑似病变区域的边界框

### 分类网络
- 预训练DenseNet121作为骨干网络
- 移除原网络最后一层全连接层
- 替换为适配二分类的输出层

## 训练策略

1. 先冻结骨干网络前80%层权重训练分类头
2. 解冻所有层进行微调
3. 使用加权交叉熵损失处理类别不平衡
4. AdamW优化器配合学习率调度
5. 早停机制防止过拟合

## 性能评估

在验证集上计算以下指标：
- 准确率（Accuracy）
- 灵敏度（Sensitivity）
- 特异度（Specificity）
- AUC（ROC曲线下面积）