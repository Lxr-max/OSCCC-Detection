import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import argparse
import numpy as np

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

def load_model(model_path, device):
    """加载训练好的模型"""
    model = CascadeNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path):
    """预处理单张图像"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换颜色空间 BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 定义预处理步骤
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 应用预处理
    image = transform(image)
    image = image.unsqueeze(0)  # 添加batch维度
    
    return image

def predict(model, image, device):
    """对单张图像进行预测"""
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
    return predicted.item(), probabilities.cpu().numpy()[0]

def main():
    parser = argparse.ArgumentParser(description='口腔鳞状细胞癌检测推理脚本')
    parser.add_argument('--image_path', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model_path', type=str, default='best_ocsccc_model.pth', help='模型权重文件路径')
    
    args = parser.parse_args()
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        # 加载模型
        model = load_model(args.model_path, device)
        print("模型加载成功")
        
        # 预处理图像
        image = preprocess_image(args.image_path)
        print("图像预处理完成")
        
        # 进行预测
        prediction, probabilities = predict(model, image, device)
        
        # 输出结果
        print(f"\n预测结果:")
        print(f"图像路径: {args.image_path}")
        print(f"预测类别: {'患病(1)' if prediction == 1 else '正常(0)'}")
        print(f"预测概率: 正常={probabilities[0]:.4f}, 患病={probabilities[1]:.4f}")
        
    except Exception as e:
        print(f"推理过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()