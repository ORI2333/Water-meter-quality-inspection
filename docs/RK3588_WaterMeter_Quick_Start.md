# RK3588 水表检测快速使用方法

版本：V1.1

日期：2026-07-07

## 1. 当前默认链路

```text
SC1336 相机 -> FPGA ISP -> HDMI -> RK3588 /dev/video73 -> YOLO11-pose -> Web 页面
```

RK3588 地址：

```text
<RK3588-IP>
```

SSH：

```bash
ssh root@<RK3588-IP>
```

密码请使用现场设备实际配置，不建议写入公开仓库。

## 2. FPGA 侧推荐操作

上板后串口参数：

```text
115200, 8N1, no flow control
```

推荐测试顺序：

```text
L  -> HDMI 基础测试图
D  -> DDR 动态测试图
R  -> 相机 RAW 灰度
I  -> RGB 去马赛克
C  -> 推荐检测模式，RGB + 白平衡/亮度 + crop
```

如果颜色不对，尝试：

```text
0 / 1 / 2 / 3
```

常用调亮：

```text
a  全局增益增加
z  全局增益减少
+  亮度增加
-  亮度减少
w  恢复默认白平衡和亮度
```

常用白平衡：

```text
r/f  红色增加/减少
g/v  绿色增加/减少
b/n  蓝色增加/减少
```

## 3. 启动 RK Web 检测

```bash
cd /home/demo/water_meter
./run_hdmi_yolo11_pose_web.sh
```

电脑浏览器打开：

```text
http://<RK3588-IP>:6008/
```

当前默认模型为精度优先 FP：

```text
/home/demo/water_meter/module/water_meter_yolo11n_pose_fp.rknn
```

当前快速模型为 INT8 hybrid：

```text
/home/demo/water_meter/module/int8_variants/water_meter_yolo11n_pose_int8_headrs_float_normal.rknn
```

默认启动使用 FP 模型，关键点更稳，适合看指针角度和圈数：

```text
Web 实时流：约 16 到 18 FPS
RKNN 推理耗时：约 48 ms
检测数量：正常应为 4 个小表盘
```

如果只追求帧率，可启动快速模式：

```bash
cd /home/demo/water_meter
WM_MODEL_MODE=fast ./run_hdmi_yolo11_pose_web.sh
```

快速 INT8 参考：

```text
Web 实时流：约 27 到 28 FPS
RKNN 推理耗时：约 25 到 28 ms
```

注意：快速模式关键点可能不如 FP 稳，正式观察指针角度时建议使用默认 FP。

## 4. Web 页面常用功能

页面地址：

```text
http://<RK3588-IP>:6008/
```

左侧为视频区，下方为功能区，右侧为两列信息栏。

常用功能：

| 功能 | 说明 |
| --- | --- |
| 暂停 / 继续 | 暂停或恢复采集和推理 |
| 刷新视频 | 重新打开 MJPEG 视频流 |
| 打开截图 | 打开当前 Web 输出帧 |
| 显示检测标注 | 开关检测框、中心点、指针线和短标签 |
| 精度 FP / 快速 INT8 | 切换模型模式，切换后服务会自动重启 |
| 小窗 / 中窗 / 影院 | 调整浏览器视频框大小 |

画质滑条：

| 滑条 | 建议 |
| --- | --- |
| 视频框宽度 | 只改变显示大小，不影响推理和码率 |
| 推流宽度 | 默认 1280，卡顿时可降到 960 或 800 |
| JPEG 质量 | 默认 90，画面糊可升高，卡顿可降低 |
| 网页帧率 | 默认 12 fps，卡顿时先降低这个 |

视频内只保留 `10^-1` 这类短标签，详细原始角度、稳定角度、圈数、每圈水量和本表估算水量在右侧“表盘检测列表”中查看。

水量计算：

```text
点击“开始测量/清零”后，从当前角度开始累计经过水量。
```

四个小表盘是十进制进位关系，不能把四个表盘的读数直接相加。当前 Web 主显示默认使用最灵敏的 `10^-4` 表盘作为测量来源：

| 表盘 | 每小格 | 每圈水量 |
| --- | --- | --- |
| `10^-1` | 0.1 m³ | 1.0 m³ |
| `10^-2` | 0.01 m³ | 0.1 m³ |
| `10^-3` | 0.001 m³ | 0.01 m³ |
| `10^-4` | 0.0001 m³ | 0.001 m³ |

右侧“表盘检测列表”会显示每个表盘独立估算的经过水量，主显示“本次经过水量”取默认测量来源。若要显示绝对读数，在“起始/基准 m³”中填入开始测量时的机械读数。

如果页面布局没有变化，按 `Ctrl+F5` 强制刷新浏览器缓存。

## 5. 停止和重启

查看进程：

```bash
ps -ef | grep hdmi_yolo11_pose_web.py | grep -v grep
```

重启：

```bash
cd /home/demo/water_meter
./run_hdmi_yolo11_pose_web.sh
```

查看日志：

```bash
tail -f /home/demo/water_meter/hdmi_yolo11_pose_web.log
```

如果 `/dev/video73` 被占用：

```bash
fuser /dev/video73
fuser -k /dev/video73
cd /home/demo/water_meter
./run_hdmi_yolo11_pose_web.sh
```

## 6. 已知图片验证模型

```bash
cd /home/demo/water_meter/code
python3 -u ./hdmi_yolo11_pose_detect.py \
  --model /home/demo/water_meter/module/water_meter_yolo11n_pose_fp.rknn \
  --image /home/demo/water_meter/pose_known_val.jpg \
  --save-output /home/demo/water_meter/fp_known.jpg \
  --conf 0.05 \
  --input-layout nhwc \
  --input-dtype float32 \
  --color rgb
```

正常结果：

```text
detections=4
```

## 7. 常见问题

### 7.1 Web 页面打不开

检查：

```bash
ss -lntp | grep 6008
tail -n 80 /home/demo/water_meter/hdmi_yolo11_pose_web.log
```

确认电脑能 ping 通：

```bash
ping <RK3588-IP>
```

### 7.2 检测不到表盘

按顺序检查：

1. FPGA 串口是否在 `C` 模式。
2. HDMI 采集是否为 `/dev/video73`。
3. Web 日志里是否有 `det=4`。
4. 画面是否过暗、过曝或颜色严重偏移。
5. 是否误用了普通 INT8 模型。
6. 先用默认 FP 模式确认关键点是否稳定，再切换快速 INT8。

### 7.3 暗光下画面延时

当前 FPGA 已限制 SC1336 最大曝光时间。暗光不要通过放开长曝光解决，否则会出现拖影和检测延迟。优先使用：

```text
a  增大全局增益
+  增加亮度偏置
外部补光
```

### 7.4 角度轻微抖动

启动时提高死区：

```bash
cd /home/demo/water_meter
./run_hdmi_yolo11_pose_web.sh --angle-deadband 4.0 --angle-alpha 0.20 --angle-confirm-frames 5 --turn-deadband 5.0
```

如果显示响应太慢：

```bash
./run_hdmi_yolo11_pose_web.sh --angle-deadband 2.0 --angle-alpha 0.35 --angle-confirm-frames 2 --turn-deadband 3.0
```

## 8. C++ 性能测试

目录：

```bash
cd /home/demo/water_meter/cpp_bench
```

运行：

```bash
LD_LIBRARY_PATH=/usr/lib:/home/demo/water_meter/module \
./rknn_pose_bench \
  /home/demo/water_meter/module/int8_variants/water_meter_yolo11n_pose_int8_headrs_float_normal.rknn \
  300 20 1
```

注意：测试 C++ 极限性能前先停止 Web，否则 Web 和 benchmark 会争用 NPU。

参考结果：

```text
want_float=1: 约 36.8 FPS
want_float=0: 约 37.7 FPS
```
