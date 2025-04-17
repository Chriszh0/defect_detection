import os
import cv2
import numpy as np
from pathlib import Path

def verify_dataset(data_dir):
    """
    验证数据集的完整性和正确性
    
    Args:
        data_dir: 数据集根目录
    """
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f'\n检查{split}集...')
        
        # 图像目录和标签目录
        img_dir = os.path.join(data_dir, 'images', split)
        label_dir = os.path.join(data_dir, 'labels', split)
        
        # 获取所有图像文件
        img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f'找到{len(img_files)}张图片')
        
        for img_file in img_files:
            # 检查图像
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f'警告：无法读取图像 {img_path}')
                continue
                
            # 检查对应的标签文件
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            
            if not os.path.exists(label_path):
                print(f'警告：找不到标签文件 {label_path}')
                continue
                
            # 验证标签格式
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                try:
                    values = list(map(float, line.strip().split()))
                    if len(values) != 5:  # YOLO格式：class x y w h
                        print(f'警告：标签格式错误 {label_path} 第{i+1}行')
                    if values[0] != 0:  # 类别ID应该为0
                        print(f'警告：类别ID错误 {label_path} 第{i+1}行')
                except:
                    print(f'警告：标签数据无效 {label_path} 第{i+1}行')
        
        print(f'{split}集检查完成')

if __name__ == '__main__':
    data_dir = '../data'
    verify_dataset(data_dir)