import os
import random
import shutil
from pathlib import Path

def split_dataset(image_dir, label_dir, output_dir, split_ratio=(0.7, 0.2, 0.1)):
    """
    划分数据集为训练集、验证集和测试集
    
    Args:
        image_dir: 原始图像目录
        label_dir: 原始标签目录
        output_dir: 输出目录
        split_ratio: (训练集比例, 验证集比例, 测试集比例)
    """
    # 确保比例和为1
    assert sum(split_ratio) == 1.0
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(image_files)
    
    # 计算每个集合的大小
    n_total = len(image_files)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    
    # 划分数据集
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # 创建目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # 复制文件
    def copy_files(files, split):
        for f in files:
            # 复制图像
            src_img = os.path.join(image_dir, f)
            dst_img = os.path.join(output_dir, 'images', split, f)
            shutil.copy2(src_img, dst_img)
            
            # 复制标签（如果存在）
            label_file = f.rsplit('.', 1)[0] + '.txt'
            src_label = os.path.join(label_dir, label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(output_dir, 'labels', split, label_file)
                shutil.copy2(src_label, dst_label)
    
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    
    # 打印统计信息
    print(f'数据集划分完成：')
    print(f'训练集：{len(train_files)}张图片')
    print(f'验证集：{len(val_files)}张图片')
    print(f'测试集：{len(test_files)}张图片')

if __name__ == '__main__':
    # 设置路径
    image_dir = '../raw_data/images'  # 原始图像目录
    label_dir = '../raw_data/labels'  # 原始标签目录
    output_dir = '../data'            # 输出目录
    
    # 执行划分
    split_dataset(image_dir, label_dir, output_dir)