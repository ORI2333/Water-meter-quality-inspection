#!/bin/bash
set -e

PORT=6008
LOG=/home/demo/water_meter/hdmi_yolo11_pose_web.log
PID=/home/demo/water_meter/hdmi_yolo11_pose_web.pid
MODEL_MODE=${WM_MODEL_MODE:-accuracy}
FP_MODEL=/home/demo/water_meter/module/water_meter_yolo11n_pose_fp.rknn
FAST_MODEL=/home/demo/water_meter/module/int8_variants/water_meter_yolo11n_pose_int8_headrs_float_normal.rknn
if [ -z "${MODEL:-}" ]; then
  if [ "$MODEL_MODE" = "fast" ]; then
    MODEL="$FAST_MODEL"
  else
    MODEL="$FP_MODEL"
  fi
fi

cd /home/demo/water_meter/code

if [ -f "$PID" ] && kill -0 "$(cat "$PID")" >/dev/null 2>&1; then
  OLD_PID=$(cat "$PID")
  echo "[INFO] stopping old web dashboard pid=$OLD_PID"
  kill "$OLD_PID" || true
  sleep 1
  if kill -0 "$OLD_PID" >/dev/null 2>&1; then
    echo "[INFO] old web dashboard still alive, force killing pid=$OLD_PID"
    kill -9 "$OLD_PID" || true
    sleep 1
  fi
fi

if fuser /dev/video73 >/dev/null 2>&1; then
  echo "[INFO] /dev/video73 is busy, stopping old users..."
  fuser -k /dev/video73 || true
  sleep 1
fi

nohup python3 -u ./hdmi_yolo11_pose_web.py \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL" \
  --device /dev/video73 \
  --width 1280 \
  --height 720 \
  --fps 60 \
  --conf 0.25 \
  --core-mask all \
  --stream-width 1280 \
  --stream-fps 12 \
  --jpeg-quality 90 \
  --angle-deadband 3.0 \
  --angle-alpha 0.20 \
  --angle-confirm-frames 4 \
  --angle-confirm-band 2.0 \
  --turn-deadband 4.0 \
  "$@" > "$LOG" 2>&1 &

echo $! > "$PID"
echo "[INFO] web dashboard started pid=$(cat "$PID")"
echo "[INFO] model mode: $MODEL_MODE"
echo "[INFO] model: $MODEL"
echo "[INFO] log: $LOG"
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$HOST_IP" ]; then
  HOST_IP="<RK3588-IP>"
fi
echo "[INFO] open: http://${HOST_IP}:${PORT}/"
