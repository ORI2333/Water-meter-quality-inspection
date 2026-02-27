# 说明文档

## 版本说明

### V0.1.1（20260227）
- 训练脚本
- 冒烟测试

### V0.1.0（20260203）
- 有效数据集559张（图片总数：967；标注总数：559），408张尚未匹配标签
- 验证了有效数据集
- 测试了常见cv算法判断指针，效果不佳受场景影响大

## 当前状态
- 原始数据集图片：`967` 张
- 已有标注：`559` 张（其余待补标）
- 当前主目标：先完成 `10^-1 ~ 10^-4` 四类指针检测，再进入角度估计

## 项目计划
### 阶段 1（进行中）
- 使用现有框标注训练 `YOLO11s` 检测模型
- 目标：稳定定位四个指针表盘区域

### 阶段 2（下一步）
- 为每个指针补充关键点：`center`（旋转中心）+ `tip`（针尖）
- 训练 `YOLO11s-pose`
- 用关键点计算旋转角度并做零位校准

## 脚本说明

### 1) 标注质量检查
文件：`script/validate_pointer_labels.py`

用途：
- 检查图片/标签配对情况
- 校验标注格式（点合法性、越界、零面积）
- 导出可视化预览图

示例：
```bash
# 常规验证
python script/validate_pointer_labels.py

# 只检查指定图片
python script/validate_pointer_labels.py --check-stems 00201 00202 00203

# 从文本文件读取要检查的图片
python script/validate_pointer_labels.py --check-stems-file script/my_check_list.txt

# 只导出问题样本，且放大标签文字
python script/validate_pointer_labels.py --vis-problems-only --label-font-size 48 --samples 100
```

### 2) 纯算法角度 Demo（传统视觉）
文件：`script/pointer_angle_demo.py`

用途：
- 在已有框标注基础上，使用 OpenCV 估计指针方向并计算角度

示例：
```bash
python script/pointer_angle_demo.py --stem 00201
```

### 3) YOLO11s 检测训练脚本
文件：`script/train_yolo11s_pointer.py`

用途：
- 将 Labelme 数据转换为 YOLO 检测格式
- 训练 `YOLO11s`（支持 CPU/CUDA 开关）

默认类别：
- `10^-1`
- `10^-2`
- `10^-3`
- `10^-4`

示例：
```bash
# 仅准备数据（不训练）
python script/train_yolo11s_pointer.py --prepare-only --device cpu --overwrite

# CPU 快速冒烟训练
python script/train_yolo11s_pointer.py --device cpu --epochs 5 --imgsz 640 --batch 4

# CPU 正式训练（较慢）
python script/train_yolo11s_pointer.py --device cpu --epochs 100 --imgsz 960 --batch 8

# 如有 NVIDIA GPU，可切换 CUDA
python script/train_yolo11s_pointer.py --device cuda --epochs 100 --imgsz 960 --batch 16
```

输出目录：
- 数据准备结果：`data/yolo11s_pointer_detect/`
- 训练结果：`data/yolo11s_pointer_detect/runs/yolo11s_pointer_detect/`

## 环境建议
- Python 3.10+
- 主要依赖：`ultralytics`、`torch`、`Pillow`、`opencv-python`

安装示例：
```bash
pip install ultralytics torch torchvision pillow opencv-python
```
