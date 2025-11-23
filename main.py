import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import os
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from collections import Counter
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCSCCDataset(Dataset):
    """实现数据读取、预处理逻辑，继承 PyTorch 的 Dataset"""
    
    def __init__(self, csv_file, transform=None, is_train=True):
        """
        初始化数据集
        
        Args:
            csv_file (string): 包含图像路径和标签的csv文件路径
            transform (callable, optional): 可选的转换函数
            is_train (bool): 是否为训练集，决定是否应用数据增强
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取图像路径和标签
        img_path = self.data.iloc[idx]['image_path']
        label = self.data.iloc[idx]['label']
        
        # 处理图像路径不存在或图像损坏的情况
        try:
            # 读取图像
            image = cv2.imread(img_path)
            if image is None:
                logger.warning(f"图像路径 {img_path} 无法读取，已跳过该样本")
                # 尝试下一个样本，避免无限递归
                next_idx = (idx + 1) % len(self.data)
                if next_idx != idx:  # 防止只有一个样本的情况
                    return self.__getitem__(next_idx)
                else:
                    # 如果数据集中只有一个无效样本，则返回一个零图像
                    image = np.zeros((224, 224, 3), dtype=np.uint8)
            # 转换颜色空间 BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"读取图像 {img_path} 时出错: {str(e)}，已跳过该样本")
            return self.__getitem__((idx + 1) % len(self.data))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
            
        return image, label

class DetectionNetwork(nn.Module):
    """
    检测网络（粗定位）
    采用轻量级目标检测网络的简化版，输入原始图像，输出疑似病变区域的边界框
    """
    def __init__(self):
        super(DetectionNetwork, self).__init__()
        # 简化的检测网络，用于演示级联结构
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # 回归头部，预测边界框 (x, y, w, h)
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),  # 边界框坐标 (x, y, w, h)
        )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        bbox = self.regressor(x)
        return bbox

class ClassificationNetwork(nn.Module):
    """
    分类网络（细分类）
    以检测网络输出的候选 patch 为输入，采用预训练的 DenseNet121 作为骨干网络
    """
    def __init__(self, num_classes=2):
        super(ClassificationNetwork, self).__init__()
        # 使用预训练的DenseNet121作为骨干网络
        self.backbone = models.densenet121(pretrained=True)
        
        # 替换分类头以适应二分类任务
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class CascadeNetwork(nn.Module):
    """
    级联网络：检测网络 + 分类网络
    参考论文 2.2 节级联网络结构，采用 SSD 简化版进行病变区域粗定位，
    然后使用 DenseNet121 进行细分类
    """
    def __init__(self, num_classes=2):
        super(CascadeNetwork, self).__init__()
        self.detection_net = DetectionNetwork()
        self.classification_net = ClassificationNetwork(num_classes)
        
    def forward(self, x):
        # 第一阶段：检测疑似区域
        bbox = self.detection_net(x)
        # 在实际应用中，我们会根据bbox裁剪图像，这里简化处理直接使用原图
        # 第二阶段：分类
        output = self.classification_net(x)
        return output

def build_model():
    """
    构建级联检测-分类网络，返回可训练的模型实例
    参考论文 2.2 节级联网络结构，采用 SSD 简化版进行病变区域粗定位
    """
    model = CascadeNetwork(num_classes=2)
    return model

def calculate_weights(csv_file):
    """
    根据训练集正负样本比例计算加权损失权重
    """
    data = pd.read_csv(csv_file)
    labels = data['label'].values
    counter = Counter(labels)
    
    total_samples = len(labels)
    pos_samples = counter[1]
    neg_samples = counter[0]
    
    # 计算权重
    pos_weight = total_samples / (2 * pos_samples)
    neg_weight = total_samples / (2 * neg_samples)
    
    return [neg_weight, pos_weight]

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    实现单个 epoch 的训练逻辑，返回该 epoch 的平均 train_loss 和 train_acc
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images.float())
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计信息
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """
    实现验证集评估逻辑，返回 val_loss、val_acc、val_sensitivity、val_specificity、val_auc
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_probs = []
    
    tp, tn, fp, fn = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images.float())
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # 计算概率
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 类别1的概率
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集所有标签和预测概率用于计算AUC
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # 计算混淆矩阵元素
            for i in range(labels.size(0)):
                if labels[i] == 1 and predicted[i] == 1:
                    tp += 1
                elif labels[i] == 0 and predicted[i] == 0:
                    tn += 1
                elif labels[i] == 0 and predicted[i] == 1:
                    fp += 1
                else:  # labels[i] == 1 and predicted[i] == 0
                    fn += 1
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # 计算敏感性和特异性
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # 计算AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    
    return avg_loss, accuracy, sensitivity, specificity, auc, all_labels, all_probs

def plot_roc_curve(labels, probs, auc, save_path='val_roc_curve.png'):
    """
    绘制ROC曲线
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def main():
    """
    串联数据加载、模型构建、训练、评估流程，处理日志与结果保存
    """
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 定义数据增强和预处理
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(saturation=(0.8, 1.2), brightness=(0.8, 1.2)),
        transforms.ToTensor(),
        # 使用ImageNet的均值和标准差进行标准化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 创建数据集和数据加载器
    try:
        # 更新为正确的数据文件路径
        train_dataset = OCSCCDataset('C:/Users/34475/Desktop/VScode/AIcode/OSCC_Detection_new/train_data_new/train_data_new.csv', transform=train_transform, is_train=True)
        val_dataset = OCSCCDataset('C:/Users/34475/Desktop/VScode/AIcode/OSCC_Detection_new/val_data_new/val_data_new.csv', transform=val_transform, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    except FileNotFoundError as e:
        logger.error(f"找不到数据文件: {e}")
        logger.info("请确保数据文件路径正确")
        return
    except Exception as e:
        logger.error(f"创建数据集时发生错误: {e}")
        return
    
    # 构建模型
    model = build_model().to(device)
    
    # 冻结DenseNet121骨干网络的前80%层权重，只训练分类头
    backbone = model.classification_net.backbone.features
    backbone_layers = list(backbone.children())
    freeze_layers_count = int(len(backbone_layers) * 0.8)
    
    for i, child in enumerate(backbone_layers):
        if i < freeze_layers_count:
            for param in child.parameters():
                param.requires_grad = False
    
    logger.info(f"已冻结DenseNet121前{freeze_layers_count}层")
    
    # 计算类别权重
    weights = calculate_weights('C:/Users/34475/Desktop/VScode/AIcode/OSCC_Detection_new/train_data_new/train_data_new.csv')
    class_weights = torch.FloatTensor(weights).to(device)
    logger.info(f"类别权重: {weights}")
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 定义优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-6)
    
    # 训练参数
    num_epochs = 50
    best_auc = 0.0
    train_logs = []
    
    logger.info("开始训练...")
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 在验证集上评估
        val_loss, val_acc, val_sensitivity, val_specificity, val_auc, labels, probs = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录日志
        train_logs.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_sensitivity': val_sensitivity,
            'val_specificity': val_specificity,
            'val_auc': val_auc
        })
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_ocsccc_model.pth')
            # 保存ROC曲线
            plot_roc_curve(labels, probs, val_auc, 'val_roc_curve.png')
        
        # 每5个epoch打印一次日志
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                        f"Val Sensitivity: {val_sensitivity:.4f}, Val Specificity: {val_specificity:.4f}, "
                        f"Val AUC: {val_auc:.4f}")
    
    # 解冻所有层进行微调
    for param in model.parameters():
        param.requires_grad = True
    
    logger.info("开始微调所有层...")
    
    # 创建新的优化器用于微调
    fine_tune_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    
    # 微调几个epoch
    for epoch in range(5):
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, fine_tune_optimizer, device)
        
        # 在验证集上评估
        val_loss, val_acc, val_sensitivity, val_specificity, val_auc, labels, probs = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录日志
        train_logs.append({
            'epoch': num_epochs + epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_sensitivity': val_sensitivity,
            'val_specificity': val_specificity,
            'val_auc': val_auc
        })
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_ocsccc_model.pth')
            # 保存ROC曲线
            plot_roc_curve(labels, probs, val_auc, 'val_roc_curve.png')
        
        logger.info(f"Fine-tune Epoch [{epoch+1}/5], "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                    f"Val Sensitivity: {val_sensitivity:.4f}, Val Specificity: {val_specificity:.4f}, "
                    f"Val AUC: {val_auc:.4f}")
    
    # 保存训练日志
    log_df = pd.DataFrame(train_logs)
    log_df.to_csv('train_log.csv', index=False)
    
    logger.info("训练完成!")
    logger.info(f"最佳验证集AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()