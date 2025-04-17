import xml.etree.ElementTree as ET
import glob
import os
import shutil
from pathlib import Path

def convert_box(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_voc_to_yolo(data_dir, phase):
    classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    
    # 创建目标目录
    new_images_dir = os.path.join(data_dir, phase, 'images')
    new_labels_dir = os.path.join(data_dir, phase, 'labels')
    os.makedirs(new_images_dir, exist_ok=True)
    os.makedirs(new_labels_dir, exist_ok=True)
    
    # 处理每个类别
    for idx, cls in enumerate(classes):
        # 移动图片
        images_path = os.path.join(data_dir, phase, 'images', cls, '*.jpg')
        for img_path in glob.glob(images_path):
            img_name = os.path.basename(img_path)
            shutil.copy2(img_path, os.path.join(new_images_dir, img_name))
            
            # 转换对应的XML标注
            xml_path = os.path.join(data_dir, phase, 'annotations', 
                                  os.path.splitext(img_name)[0] + '.xml')
            
            # 如果找不到XML文件，跳过
            if not os.path.exists(xml_path):
                print(f"Warning: Cannot find annotation for {img_name}")
                continue
                
            # 解析XML
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 获取图片尺寸
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            
            # 处理每个目标
            txt_path = os.path.join(new_labels_dir, 
                                  os.path.splitext(img_name)[0] + '.txt')
            with open(txt_path, 'w') as txt_file:
                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    cls_name = obj.find('name').text
                    if cls_name not in classes or int(difficult) == 1:
                        continue
                    cls_id = classes.index(cls_name)
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                         float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
                    bb = convert_box((w,h), b)
                    txt_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")

if __name__ == '__main__':
    data_dir = 'D:\\Program\\defect_detection\\data\\NEU-DET'
    for phase in ['train', 'validation']:
        convert_voc_to_yolo(data_dir, phase)
        print(f"Converted {phase} dataset")