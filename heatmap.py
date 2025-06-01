import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import select_device
import os
from pathlib import Path

class FeatureExtractor:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.features = {layer: None for layer in target_layers}
        self.hooks = []
        
        # 注册钩子函数
        for layer_name in target_layers:
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                self.hooks.append(
                    layer.register_forward_hook(self._make_hook(layer_name))
                )
            else:
                print(f"找不到层: {layer_name}")
    
    def _get_layer_by_name(self, name):
        # 解析层名称并获取相应的模块
        parts = name.split('.')
        module = self.model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def _make_hook(self, layer_name):
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def __call__(self, x):
        # 进行前向传播，同时收集特征
        with torch.no_grad():
            _ =self.model(*x)
        return self.features

def generate_heatmap(feature_map, img_size):
    """生成热力图"""
    # 合并所有通道为一个特征图
    feature_map = feature_map.sum(dim=1, keepdim=True)
    
    # 归一化
    min_val = feature_map.min()
    max_val = feature_map.max()
    feature_map = (feature_map - min_val) / (max_val - min_val + 1e-10)
    
    # 转换为numpy并调整大小
    heatmap = feature_map[0, 0].cpu().numpy()
    heatmap = cv2.resize(heatmap, (img_size, img_size))
    
    # 应用颜色映射
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap

def overlay_heatmap(image, heatmap, alpha=0.5):
    """将热力图叠加到原始图像上"""
    # 确保图像是3通道
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # 调整热力图大小
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # 叠加
    overlay = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
    return overlay

def main():
    # 配置参数
    weights = 'runs/train/exp11/weights/best.pt'  # 模型权重路径
    device = select_device('0')  # GPU设备
    img_size = 640  # 图像大小
    save_dir = Path('runs/feature_maps')  # 保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试图像路径 - 替换为你的实际图像路径
    test_img_rgb = 'dataset/uav_online/images/rgb_0336_1129.jpg'  # RGB样本图像
    test_img_ir = 'dataset/uav_online/images/ir_0336_1129.jpg'    # 红外样本图像
    test_img_event = 'dataset/uav_online/images/event_0336_1129.jpg'  # 事件样本图像
    
    # 加载模型
    model = attempt_load(weights, map_location=device)
    model.eval()
    stride = int(model.stride.max())  # 模型步长
    img_size = check_img_size(img_size, s=stride)  # 检查图像大小
    
    # 我们关注的是第一次融合前的特征图 - 在模型结构中查看相应的层索引
    target_layers = ['model.4', 'model.9', 'model.11']  # RGB, IR, Event P3特征层索引
    
    # 创建特征提取器
    extractor = FeatureExtractor(model, target_layers)
    
    # 加载测试图像并预处理
    rgb_img_orig = cv2.imread(test_img_rgb)
    ir_img_orig = cv2.imread(test_img_ir)
    event_img_orig = cv2.imread(test_img_event)
    
    # 预处理图像
    rgb_img = cv2.resize(rgb_img_orig, (img_size, img_size))
    ir_img = cv2.resize(ir_img_orig, (img_size, img_size))
    event_img = cv2.resize(event_img_orig, (img_size, img_size))
    
    # 转换为PyTorch张量
    rgb_tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float().to(device) / 255.0
    ir_tensor = torch.from_numpy(ir_img.transpose(2, 0, 1)).float().to(device) / 255.0
    event_tensor = torch.from_numpy(event_img.transpose(2, 0, 1)).float().to(device) / 255.0
    
    # 添加批次维度
    rgb_tensor = rgb_tensor.unsqueeze(0)
    ir_tensor = ir_tensor.unsqueeze(0)
    event_tensor = event_tensor.unsqueeze(0)
    
    # 提取特征
    features = extractor([rgb_tensor, ir_tensor, event_tensor])
    
    # 生成并保存热力图
    plt.figure(figsize=(18, 6))
    
    # RGB特征热力图
    rgb_feature = features['model.4']
    rgb_heatmap = generate_heatmap(rgb_feature, img_size)
    rgb_overlay = overlay_heatmap(rgb_img, rgb_heatmap)
    plt.subplot(1, 3, 1)
    plt.title('RGB特征热力图')
    plt.imshow(cv2.cvtColor(rgb_overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # IR特征热力图
    ir_feature = features['model.9']
    ir_heatmap = generate_heatmap(ir_feature, img_size)
    ir_overlay = overlay_heatmap(ir_img, ir_heatmap)
    plt.subplot(1, 3, 2)
    plt.title('红外特征热力图')
    plt.imshow(cv2.cvtColor(ir_overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Event特征热力图
    event_feature = features['model.11']
    event_heatmap = generate_heatmap(event_feature, img_size)
    event_overlay = overlay_heatmap(event_img, event_heatmap)
    plt.subplot(1, 3, 3)
    plt.title('事件特征热力图')
    plt.imshow(cv2.cvtColor(event_overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(str(save_dir / 'modality_features_before_fusion.png'))
    plt.close()
    
    # 单独保存每个热力图
    cv2.imwrite(str(save_dir / 'rgb_heatmap.jpg'), rgb_overlay)
    cv2.imwrite(str(save_dir / 'ir_heatmap.jpg'), ir_overlay)
    cv2.imwrite(str(save_dir / 'event_heatmap.jpg'), event_overlay)
    
    print(f"热力图已保存到 {save_dir}")
    
    # 释放钩子
    extractor.remove_hooks()

if __name__ == '__main__':
    main()
