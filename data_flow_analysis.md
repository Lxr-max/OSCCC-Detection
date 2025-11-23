# OSCC Detection 数据流分析

本文档详细描述了在运行程序时数据经过函数的顺序与变化，包括数据维度的变化情况。

## 1. 整体数据流程

```
原始图像数据 → 数据加载 → 预处理与增强 → 神经网络模型 → 输出结果 → 评估指标
```

## 2. 详细数据流与变化

### 2.1 数据加载阶段

#### OCSCCDataset 类
- 输入：CSV文件中的图像路径和标签
- 输出：单个图像数据和对应标签
- 流程：
  1. 从CSV文件中读取图像路径和标签
  2. 使用OpenCV读取图像（BGR格式）
  3. 转换颜色空间为RGB格式
  4. 返回图像和标签

### 2.2 数据预处理与增强阶段

#### 训练集变换 (train_transform)
- 输入：原始图像（任意尺寸，3通道）
- 输出：224×224×3的张量
- 变换序列：
  1. `transforms.ToPILImage()` - 转换为PIL图像
  2. `transforms.Resize((224, 224))` - 调整大小为224×224
  3. `transforms.RandomHorizontalFlip(p=0.5)` - 50%概率水平翻转
  4. `transforms.RandomRotation(degrees=15)` - 随机旋转±15度
  5. `transforms.ColorJitter(saturation=(0.8, 1.2), brightness=(0.8, 1.2))` - 色彩抖动
  6. `transforms.ToTensor()` - 转换为张量，形状变为[3, 224, 224]，像素值范围[0,1]
  7. `transforms.Normalize(...)` - 标准化处理，保持形状不变[3, 224, 224]

#### 验证集变换 (val_transform)
- 输入：原始图像（任意尺寸，3通道）
- 输出：224×224×3的张量
- 变换序列：
  1. `transforms.ToPILImage()` - 转换为PIL图像
  2. `transforms.Resize((224, 224))` - 调整大小为224×224
  3. `transforms.ToTensor()` - 转换为张量，形状变为[3, 224, 224]，像素值范围[0,1]
  4. `transforms.Normalize(...)` - 标准化处理，保持形状不变[3, 224, 224]

### 2.3 神经网络模型阶段

#### CascadeNetwork 模型
整个模型由两个子网络组成：DetectionNetwork（检测网络）和ClassificationNetwork（分类网络）。

##### DetectionNetwork（检测网络）
- 输入：一批图像张量，形状为 [batch_size, 3, 224, 224]
- 输出：边界框坐标，形状为 [batch_size, 4]

网络结构：
1. Conv2d(3, 32, kernel_size=3, padding=1) + ReLU
   - 输入：[batch_size, 3, 224, 224]
   - 输出：[batch_size, 32, 224, 224]
   
2. MaxPool2d(2, 2)
   - 输入：[batch_size, 32, 224, 224]
   - 输出：[batch_size, 32, 112, 112]
   
3. Conv2d(32, 64, kernel_size=3, padding=1) + ReLU
   - 输入：[batch_size, 32, 112, 112]
   - 输出：[batch_size, 64, 112, 112]
   
4. MaxPool2d(2, 2)
   - 输入：[batch_size, 64, 112, 112]
   - 输出：[batch_size, 64, 56, 56]
   
5. Conv2d(64, 128, kernel_size=3, padding=1) + ReLU
   - 输入：[batch_size, 64, 56, 56]
   - 输出：[batch_size, 128, 56, 56]
   
6. MaxPool2d(2, 2)
   - 输入：[batch_size, 128, 56, 56]
   - 输出：[batch_size, 128, 28, 28]
   
7. AdaptiveAvgPool2d((7, 7))
   - 输入：[batch_size, 128, 28, 28]
   - 输出：[batch_size, 128, 7, 7]
   
8. Flatten()
   - 输入：[batch_size, 128, 7, 7]
   - 输出：[batch_size, 128*7*7] = [batch_size, 6272]
   
9. Linear(6272, 512) + ReLU
   - 输入：[batch_size, 6272]
   - 输出：[batch_size, 512]
   
10. Linear(512, 4)
    - 输入：[batch_size, 512]
    - 输出：[batch_size, 4] （边界框坐标 x, y, w, h）

注意：在当前实现中，检测网络的输出并未被用于裁剪图像，而是直接将原始图像传递给分类网络。

##### ClassificationNetwork（分类网络）
- 输入：一批图像张量，形状为 [batch_size, 3, 224, 224]
- 输出：分类结果，形状为 [batch_size, 2]

网络结构：
1. 使用预训练的 DenseNet121 作为骨干网络
   - 输入：[batch_size, 3, 224, 224]
   - 输出：[batch_size, 1000]（DenseNet121默认输出）
   
2. 自定义分类头 Linear(1000, 2)
   - 输入：[batch_size, 1000]
   - 输出：[batch_size, 2]（二分类：正常/癌症）

### 2.4 训练和验证阶段

#### train_one_epoch 函数
- 输入：模型、数据加载器、损失函数、优化器、设备
- 输出：平均训练损失和训练准确率
- 流程：
  1. 遍历训练数据加载器中的每个批次
  2. 将数据移动到指定设备（CPU/GPU）
  3. 前向传播得到输出 [batch_size, 2]
  4. 计算损失（CrossEntropyLoss）
  5. 反向传播并更新参数
  6. 计算准确率

#### validate 函数
- 输入：模型、数据加载器、损失函数、设备
- 输出：验证损失、准确率、敏感性、特异性、AUC以及标签和概率列表
- 流程：
  1. 遍历验证数据加载器中的每个批次
  2. 将数据移动到指定设备（CPU/GPU）
  3. 前向传播得到输出 [batch_size, 2]
  4. 计算损失（CrossEntropyLoss）
  5. 使用softmax计算各类别概率
  6. 计算各项评估指标

### 2.5 主程序执行流程

1. 设置设备（CUDA或CPU）
2. 定义训练和验证数据变换
3. 创建数据集和数据加载器
4. 构建级联网络模型
5. 冻结DenseNet121前80%的层
6. 计算类别权重
7. 定义损失函数（带权重的交叉熵）
8. 定义优化器和学习率调度器
9. 执行训练循环（50个epochs）
   - 每个epoch调用[train_one_epoch](file:///c%3A/Users/34475/Desktop/VScode/AIcode/OSCC_Detection_WH/main.py#L166-L197)
   - 每个epoch结束后调用[validate](file:///c%3A/Users/34475/Desktop/VScode/AIcode/OSCC_Detection_WH/main.py#L200-L257)
   - 根据验证损失调整学习率
   - 保存最佳模型和ROC曲线
10. 解冻所有层进行微调（5个epochs）
11. 保存训练日志到CSV文件

## 3. 关键数据维度总结

| 阶段 | 数据形状 | 说明 |
|------|----------|------|
| 原始图像 | 任意尺寸×任意尺寸×3 | RGB彩色图像 |
| 预处理后 | [3, 224, 224] | 标准化张量 |
| 检测网络输入 | [batch_size, 3, 224, 224] | 一批图像 |
| 检测网络输出 | [batch_size, 4] | 边界框坐标 |
| 分类网络输入 | [batch_size, 3, 224, 224] | 一批图像 |
| 分类网络输出 | [batch_size, 2] | 二分类结果 |
| 最终输出 | [batch_size, 2] | 概率分布 |

注：虽然设计了级联网络结构，但目前实现中检测网络的输出并未用于裁剪图像，而是直接使用原始图像进行分类。