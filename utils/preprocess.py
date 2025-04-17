import cv2
import numpy as np

def preprocess_image(image, params=None):
    """根据文档要求的预处理流程处理图像
    
    Args:
        image: 输入的BGR格式图像
        params: 预处理参数字典
    Returns:
        tuple: (enhanced_3ch, enhanced)
            - enhanced_3ch: 处理后的3通道BGR格式图像（用于YOLO）
            - enhanced: 处理后的单通道图像（用于传统检测）
    """
    if params is None:
        params = {
            'median_size': 3,
            'sobel_mode': 'sqrt',  # 'x', 'y', 'xy', 'sqrt'
            'thresh_low': 64,
            'thresh_high': 192,
            'edge_scale': 4,
            'edge_threshold': 20
        }
    
    # 1. 转换为灰度图（使用红通道）
    gray = cv2.split(image)[2]
    
    # 2. 中值滤波
    median = cv2.medianBlur(gray, params['median_size'])
    
    # 3. Sobel滤波
    sobelx = cv2.Sobel(median, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(median, cv2.CV_64F, 0, 1, ksize=3)
    
    if params['sobel_mode'] == 'x':
        gradient = np.abs(sobelx)
    elif params['sobel_mode'] == 'y':
        gradient = np.abs(sobely)
    elif params['sobel_mode'] == 'xy':
        gradient = np.abs(sobelx) + np.abs(sobely)
    else:  # sqrt
        gradient = np.sqrt(sobelx**2 + sobely**2)
    
    gradient = np.uint8(255 * gradient / gradient.max())
    
    # 4. 双阈值分割
    mask = np.zeros_like(gradient)
    mask[(gradient >= params['thresh_low']) & (gradient <= params['thresh_high'])] = 128
    mask[gradient > params['thresh_high']] = 255
    
    # 5. 边缘增强
    kernel = np.ones((params['edge_scale'], params['edge_scale']), np.uint8)
    enhanced = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 6. 转换回3通道BGR格式（用于YOLO）
    enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced_3ch, enhanced  # 返回两种格式

def detect_defects(image, params=None):
    """检测图像中的缺陷
    
    Args:
        image: 输入的BGR格式图像
        params: 检测参数
    Returns:
        boxes: 检测到的缺陷边界框列表 [(x1,y1,x2,y2),...]
        types: 缺陷类型列表
    """
    # 获取预处理结果，使用单通道版本
    _, processed = preprocess_image(image, params)
    
    # Hough变换检测直线型缺陷
    lines = cv2.HoughLinesP(processed, 1, np.pi/180, 50, 
                           minLineLength=100, maxLineGap=10)
    
    # 轮廓检测其他类型缺陷
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    types = []
    
    # 处理直线型缺陷
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            boxes.append([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)])
            types.append('crack')
            
    # 处理斑点型缺陷
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # 面积阈值
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, x+w, y+h])
            types.append('spot')
            
    return boxes, types