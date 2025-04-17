import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import albumentations as A

def create_augmentation():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(p=1),
            A.MultiplicativeNoise(p=1),
        ], p=0.3),
    ])

def augment_dataset(source_dir, target_dir, num_aug=2):
    """
    对数据集进行增强
    source_dir: 原始数据目录
    target_dir: 增强后的数据目录
    num_aug: 每张图像增强的次数
    """
    transform = create_augmentation()
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    images_dir = os.path.join(target_dir, 'images')
    labels_dir = os.path.join(target_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # 复制原始数据
    src_images = os.path.join(source_dir, 'images')
    src_labels = os.path.join(source_dir, 'labels')
    
    # 复制原始文件
    for img_file in os.listdir(src_images):
        if img_file.endswith('.jpg'):
            # 复制图像
            shutil.copy2(
                os.path.join(src_images, img_file),
                os.path.join(images_dir, img_file)
            )
            # 复制标签
            label_file = img_file.replace('.jpg', '.txt')
            if os.path.exists(os.path.join(src_labels, label_file)):
                shutil.copy2(
                    os.path.join(src_labels, label_file),
                    os.path.join(labels_dir, label_file)
                )
    
    # 对性能较差的类别进行额外增强
    poor_classes = ['crazing', 'rolled-in_scale']
    for img_file in os.listdir(src_images):
        if any(cls_name in img_file for cls_name in poor_classes):
            img_path = os.path.join(src_images, img_file)
            label_path = os.path.join(src_labels, img_file.replace('.jpg', '.txt'))
            
            if not os.path.exists(label_path):
                continue
                
            image = cv2.imread(img_path)
            
            for i in range(num_aug):
                # 应用增强
                augmented = transform(image=image)
                aug_image = augmented['image']
                
                # 保存增强后的图像
                aug_name = f"{os.path.splitext(img_file)[0]}_aug{i}.jpg"
                cv2.imwrite(os.path.join(images_dir, aug_name), aug_image)
                
                # 复制对应的标签文件
                aug_label = aug_name.replace('.jpg', '.txt')
                shutil.copy2(label_path, os.path.join(labels_dir, aug_label))

if __name__ == '__main__':
    source_dir = "D:\\Program\\defect_detection\\data\\NEU-DET\\train"
    target_dir = "D:\\Program\\defect_detection\\data\\NEU-DET\\train_augmented"
    augment_dataset(source_dir, target_dir)