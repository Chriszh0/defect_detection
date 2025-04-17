import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path

def convert_bbox_to_yolo(size, box):
    """将VOC格式的边界框转换为YOLO格式
    
    Args:
        size: 图像尺寸 (width, height)
        box: VOC格式边界框 (xmin, ymin, xmax, ymax)
    Returns:
        tuple: YOLO格式 (x_center, y_center, width, height)
    """
    dw = 1./size[0]
    dh = 1./size[1]
    
    xmin, ymin, xmax, ymax = box
    w = xmax - xmin
    h = ymax - ymin
    x = xmin + w/2
    y = ymin + h/2
    
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    return (x, y, w, h)

def convert_neu_to_yolo(src_dir, dst_dir):
    """转换NEU-DET数据集为YOLO格式
    
    Args:
        src_dir: NEU-DET数据集根目录
        dst_dir: 输出目录
    """
    # 创建目标目录
    os.makedirs(os.path.join(dst_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'labels', 'val'), exist_ok=True)
    
    # 类别映射
    classes = {
        'crazing': 0,
        'inclusion': 1,
        'patches': 2,
        'pitted_surface': 3,
        'rolled-in_scale': 4,
        'scratches': 5
    }
    
    # 处理训练集和验证集
    for split in ['train', 'validation']:
        src_images = os.path.join(src_dir, split, 'images')
        src_annots = os.path.join(src_dir, split, 'annotations')
        
        dst_split = 'train' if split == 'train' else 'val'
        dst_images = os.path.join(dst_dir, 'images', dst_split)
        dst_labels = os.path.join(dst_dir, 'labels', dst_split)
        
        # 遍历所有XML文件
        for xml_file in os.listdir(src_annots):
            if not xml_file.endswith('.xml'):
                continue
                
            # 解析XML
            tree = ET.parse(os.path.join(src_annots, xml_file))
            root = tree.getroot()
            
            # 获取图像尺寸
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # 创建YOLO格式标签文件
            txt_file = xml_file.replace('.xml', '.txt')
            txt_path = os.path.join(dst_labels, txt_file)
            
            with open(txt_path, 'w') as f:
                for obj in root.findall('object'):
                    cls = obj.find('name').text
                    if cls not in classes:
                        continue
                        
                    cls_id = classes[cls]
                    
                    # 获取边界框
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # 转换为YOLO格式
                    x, y, w, h = convert_bbox_to_yolo((width, height), (xmin, ymin, xmax, ymax))
                    
                    # 写入标签文件
                    f.write(f'{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')
            
            # 复制对应的图像文件
            img_file = xml_file.replace('.xml', '.jpg')
            shutil.copy2(
                os.path.join(src_images, img_file),
                os.path.join(dst_images, img_file)
            )
    
    # 创建数据集配置文件
    with open(os.path.join(dst_dir, 'dataset.yaml'), 'w') as f:
        f.write(f"""path: {dst_dir}  # 数据集根目录
train: images/train  # 训练图像相对路径
val: images/val  # 验证图像相对路径

nc: {len(classes)}  # 类别数量
names: {list(classes.keys())}  # 类别名称

author: 'NEU-DET'
date: '2024-03'
description: 'NEU表面缺陷检测数据集'""")
    
    print('数据集转换完成！')
    print(f'类别: {classes}')
    
    # 统计数据集大小
    train_images = len(os.listdir(os.path.join(dst_dir, 'images', 'train')))
    val_images = len(os.listdir(os.path.join(dst_dir, 'images', 'val')))
    print(f'训练集: {train_images}张图片')
    print(f'验证集: {val_images}张图片')

if __name__ == '__main__':
    src_dir = '../archive/NEU-DET'
    dst_dir = '../data'
    convert_neu_to_yolo(src_dir, dst_dir)