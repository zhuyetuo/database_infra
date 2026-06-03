#!/bin/bash
# =============================================================
#  宠物项圈模拟器启动脚本
#  配置统一在 sim_config.env 修改
# =============================================================

cd "$(dirname "$0")"

# 加载配置
set -a
source sim_config.env
set +a

echo "================================================="
echo " 配置摘要"
echo "  窗口时长     : ${WINDOW_MINUTES} min"
echo "  每窗口等待   : ${WINDOW_SEC} s (真实时间)"
echo "  每天窗口数   : ${WINDOWS_PER_DAY}"
echo "  IMU 采样率   : ${IMU_SAMPLE_HZ} Hz"
echo "  环境采样间隔 : ${ENV_SAMPLE_INTERVAL} s"
echo "  颈温采样间隔 : ${NECK_SAMPLE_INTERVAL} s"
echo "  发病天数     : day ${SICK_START_DAY}"
echo "================================================="
echo ""

python sim_devices.py
