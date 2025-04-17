from ultralytics import YOLO
import yaml
from pathlib import Path
import os

def train():
    # 获取当前脚本所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载配置
    config_path = os.path.join(current_dir, 'configs', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 加载YOLOv5模型
    model = YOLO('yolov5su.pt') 
    
    # 训练模型
    data_yaml_path = os.path.join(current_dir, 'configs', 'data.yaml')
    model.train(
        data=data_yaml_path,  # 数据集配置文件
        epochs=cfg['train']['epochs'],
        batch=cfg['train']['batch_size'],
        imgsz=cfg['train']['img_size'],
        device=0,  # 使用GPU
        pretrained=True,
        resume=False
    )

if __name__ == '__main__':
    train()