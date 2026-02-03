# 说明文档

## 修订记录

### V0.1
- 新建工程
- 检验当前标签质量

## 当前
原始数据集图片有967张
标签待完善

数据集尚未划分

## 计划

### 1阶段
用当前数据集训练 YOLO11s，把 10^-1 ~ 10^-4 四类位置稳定检出。

### 2阶段
- 为每个指针补 2个关键点：center（旋转中心）+ tip（针尖）。
- 用关键点数据训练 YOLO11s-pose模型
- 直接算角度
- 矫正

## 脚本说明
### script\validate_pointer_labels.py
**只检查指定图片**
python script/validate_pointer_labels.py --check-stems 00201 00202 00203

**从文件读取要检查的图片**
python script/validate_pointer_labels.py --check-stems-file script/my_check_list.txt

**只导出有问题样本 + 放大标签字**
python script/validate_pointer_labels.py --vis-problems-only --label-font-size 48 --samples 100

### script/pointer_angle_demo.py
python script/pointer_angle_demo.py --stem 00201
