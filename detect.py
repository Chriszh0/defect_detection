import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from utils.preprocess import preprocess_image, detect_defects
import os

def detect_defects_yolo(image_path, model_path, conf_thres=0.25, save_path=None):
    """使用训练好的YOLOv5模型检测缺陷
    
    Args:
        image_path: 输入图像路径
        model_path: 模型权重路径
        conf_thres: 置信度阈值
        save_path: 保存结果的路径
    """
    # 加载配置
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 加载模型
    model = YOLO(model_path)
    
    # 读取并预处理图像
    image = cv2.imread(image_path)
    print("原始图像形状:", image.shape)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 使用预处理增强特征
    enhanced_3ch, enhanced = preprocess_image(image, cfg['preprocess'])
    print("预处理后图像形状:", enhanced_3ch.shape)
    
    # 在模型推理之前打印
    print("输入到模型的图像类型:", type(enhanced_3ch))
    
    # 模型推理 - 使用更高的置信度阈值和NMS阈值
    results = model(enhanced_3ch, 
                   conf=max(conf_thres, cfg['detect']['conf_thres']),  # 使用较高的置信度阈值
                   iou=cfg['detect']['iou_thres'],                     # 使用NMS阈值
                   max_det=cfg['detect']['max_det'])[0]                # 限制最大检测数
    
    # 对检测结果按置信度排序并只保留前N个最高置信度的结果
    max_boxes = 5  # 最多显示5个检测框
    boxes = results.boxes
    if len(boxes) > max_boxes:
        # 按置信度排序并只保留前max_boxes个
        conf_sorted_idx = boxes.conf.argsort(descending=True)[:max_boxes]
        results.boxes = boxes[conf_sorted_idx]
    
    # 绘制YOLO结果
    for r in results.boxes:
        box = r.xyxy[0]
        conf = r.conf[0]
        cls = int(r.cls[0])
        cls_name = model.names[cls]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{cls_name}: {conf:.2f}', 
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)
    
    # 传统方法检测 - 也限制检测框数量
    boxes, types = detect_defects(image, cfg['preprocess'])
    if len(boxes) > max_boxes:
        boxes = boxes[:max_boxes]
        types = types[:max_boxes]
    
    # 绘制传统方法结果
    for (box, defect_type) in zip(boxes, types):
        x1, y1, x2, y2 = box
        color = (0, 0, 255) if defect_type == 'crack' else (255, 0, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'Traditional: {defect_type}', 
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)
    
    # 保存结果
    if save_path:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image)
        print(f"结果已保存到: {save_path}")
    
    return image

def run_live_detection(weights='models/best.pt', save_dir=None):
    """实时摄像头检测
    
    Args:
        weights: 模型权重路径
        save_dir: 保存视频的目录
    """
    # 加载配置
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 加载模型
    model = YOLO(weights)
    
    cap = cv2.VideoCapture(0)
    
    # 设置视频保存
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'detection.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, 20.0, 
                            (int(cap.get(3)), int(cap.get(4))))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 预处理
        enhanced_3ch, enhanced = preprocess_image(frame, cfg['preprocess'])
        
        # YOLO检测
        results = model(enhanced_3ch, conf=cfg['detect']['conf_thres'])[0]
        
        # 传统方法检测
        boxes, types = detect_defects(frame, cfg['preprocess'])
        
        # 绘制YOLO结果
        for r in results.boxes:
            box = r.xyxy[0]
            conf = r.conf[0]
            cls = int(r.cls[0])
            cls_name = model.names[cls]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{cls_name}: {conf:.2f}', 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)
        
        # 绘制传统方法结果
        for (box, defect_type) in zip(boxes, types):
            x1, y1, x2, y2 = box
            color = (0, 0, 255) if defect_type == 'crack' else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'Traditional: {defect_type}', 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 2)
        
        # 保存视频帧
        if save_dir:
            out.write(frame)
            
        cv2.imshow('Defect Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    if save_dir:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='图像路径或"0"表示摄像头')
    parser.add_argument('--weights', type=str, default='models/best.pt', help='模型权重路径')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--save', type=str, help='保存结果的路径')
    args = parser.parse_args()
    
    if args.source == '0':
        run_live_detection(weights=args.weights, save_dir=args.save)
    else:
        result = detect_defects_yolo(args.source, args.weights, args.conf, args.save)
        cv2.imshow('Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()