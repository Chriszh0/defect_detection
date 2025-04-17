# 仅供天津科技大学“机器视觉检测技术”课程学习交流使用

# 工业缺陷检测系统

基于YOLOv5和传统图像处理技术的工业表面缺陷检测系统。该项目可以检测金属表面的多种缺陷类型，包括裂纹、夹杂物、斑块、凹坑表面、轧制鳞片和划痕等。

## 项目结构

```
defect_detection/
├── configs/                  # 配置文件目录
│   ├── config.yaml           # 主配置文件（训练参数、预处理参数等）
│   └── data.yaml             # 数据集配置文件
├── data/                     # 数据集目录
│   ├── NEU-DET/              # NEU-DET数据集
│   └── dataset.yaml          # 数据集描述
├── models/                   # 模型目录
│   └── weights/              # 存放训练好的权重
├── results/                  # 检测结果输出目录
├── runs/                     # 训练过程输出目录
├── utils/                    # 工具函数目录
│   ├── augment.py            # 数据增强
│   ├── convert_neu_to_yolo.py # NEU-DET数据集转YOLO格式
│   ├── convert_voc_to_yolo.py # VOC数据集转YOLO格式
│   ├── preprocess.py         # 图像预处理
│   ├── split_dataset.py      # 数据集分割
│   └── verify_dataset.py     # 数据集验证
├── detect.py                 # 缺陷检测主程序
├── train.py                  # 训练脚本
├── train_tutorial.ipynb      # 训练教程(Jupyter notebook)
├── defect_tutorial.ipynb     # 缺陷检测教程(Jupyter notebook)
├── model_evaluation_analysis.ipynb # 模型评估分析
├── requirements.txt          # 项目依赖
├── yolov5su.pt               # YOLOv5预训练模型(small)
├── yolov5mu.pt               # YOLOv5预训练模型(medium)
└── yolo11n.pt                # YOLOv11预训练模型
```

## 功能特点

- 支持多种工业表面缺陷类型检测
- 结合深度学习(YOLOv5/YOLOv11)和传统图像处理方法
- 实时摄像头检测
- 图像预处理增强特征
- 批量处理图像文件
- 训练自定义数据集
- 模型评估和结果分析

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/chriszh0/defect_detection.git
cd defect_detection
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据集

本项目默认使用NEU-DET数据集，包含六种表面缺陷类型：
- crazing（龟裂）
- inclusion（夹杂物）
- patches（斑块）
- pitted_surface（凹坑表面）
- rolled-in_scale（轧制鳞片）
- scratches（划痕）

如果要使用自己的数据集，需要按照YOLO格式组织数据，并修改`configs/data.yaml`文件。

## 使用方法

### 训练模型

使用默认配置训练模型：

```bash
python train.py
```

训练结果将保存在`runs/train/`目录下。

### 检测缺陷

1. 使用图像文件：

```bash
python detect.py --source path/to/image.jpg --weights models/best.pt --conf 0.25 --save results/output.jpg
```

2. 使用摄像头：

```bash
python detect.py --source 0 --weights models/best.pt --save results/
```

### 参数说明

- `--source`：输入源，可以是图像路径或`0`（表示摄像头）
- `--weights`：模型权重路径
- `--conf`：置信度阈值
- `--save`：保存结果的路径

## 配置文件

配置文件位于`configs/`目录下：

- `config.yaml`：包含训练参数、预处理参数和检测配置
- `data.yaml`：数据集配置

可以根据需要修改这些配置文件以适应不同的场景。

## 工具函数

`utils/`目录包含一系列实用工具：

- 数据增强：`augment.py`
- 数据集格式转换：`convert_neu_to_yolo.py`, `convert_voc_to_yolo.py`
- 图像预处理：`preprocess.py`
- 数据集分割：`split_dataset.py`
- 数据集验证：`verify_dataset.py`

## 教程

项目提供了两个教程Jupyter Notebook：

- `train_tutorial.ipynb`：模型训练教程
- `defect_tutorial.ipynb`：缺陷检测教程

## 项目依赖

主要依赖包括：

- torch >= 1.7.0
- torchvision >= 0.8.1
- opencv-python >= 4.1.2
- numpy >= 1.18.5
- PyYAML >= 5.3.1
- tqdm >= 4.41.0
- matplotlib >= 3.2.2
- seaborn >= 0.11.0
- pandas >= 1.1.4

详细依赖请参考`requirements.txt`文件。

## 联系方式

如有问题或建议，请联系：jiuzhu590@gmail.com
