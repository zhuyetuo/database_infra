#!/bin/bash
# =============================================================
#  宠物项圈模拟器启动脚本
#  修改下方配置后直接运行：bash run_sim.sh
# =============================================================

# ── 数据库连接 ─────────────────────────────────────────────
export TD_HOST="127.0.0.1"
export TD_PORT="6041"
export TD_USER="root"
export TD_PASS="taosdata"
export TD_DB="pet_collar_raw"

export PG_HOST="127.0.0.1"
export PG_PORT="5432"
export PG_USER="postgres"
export PG_PASSWORD="123456"
export PG_DB="pet_collar"

# ── 窗口设置 ───────────────────────────────────────────────
#
#  WINDOW_MINUTES : 每批数据窗口的时长（分钟）
#                   可选值：15 / 30 / 60
#
export WINDOW_MINUTES=15

#  WINDOW_SEC : 测试加速——每个模拟窗口等待多少真实秒
#               例：WINDOW_MINUTES=15，WINDOW_SEC=15
#                   → 每 15 真实秒 = 1 个 15 分钟模拟窗口
#               生产环境（不加速）：WINDOW_SEC=$((WINDOW_MINUTES * 60))
#
export WINDOW_SEC=15

#  WINDOWS_PER_DAY : 多少个窗口合计为 1 个模拟天
#                    实际值 = 1440 / WINDOW_MINUTES
#                      15min 窗口 -> 96
#                      30min 窗口 -> 48
#                      60min 窗口 -> 24
#                    测试时可设小一些（如 12）以快速看到每日评估结果
#
export WINDOWS_PER_DAY=12

# ── IMU 采样率 ─────────────────────────────────────────────
#
#  IMU_SAMPLE_HZ : 设备端 IMU 采样频率（Hz）
#                  可选值：25 / 50
#                  决定每个窗口原始数据量：
#                    25Hz × 15min = 22,500 pts/window/device
#                    50Hz × 15min = 45,000 pts/window/device
#
export IMU_SAMPLE_HZ=50

# ── 环境传感器采样间隔 ─────────────────────────────────────
#
#  ENV_SAMPLE_INTERVAL : 环境温度/湿度 & 脖颈温度的采样间隔（秒）
#                        例：60 → 每分钟一个读数
#                        15min 窗口内采样点数 = 900 / ENV_SAMPLE_INTERVAL
#
export ENV_SAMPLE_INTERVAL=60

# ── 场景设置 ───────────────────────────────────────────────
#
#  SICK_START_DAY : sim_device_sick 在第几模拟天开始出现皮肤异常
#
export SICK_START_DAY=5

# =============================================================
#  启动
# =============================================================
cd "$(dirname "$0")"

echo "================================================="
echo " 配置摘要"
echo "  窗口时长     : ${WINDOW_MINUTES} min"
echo "  每窗口等待   : ${WINDOW_SEC} s (真实时间)"
echo "  每天窗口数   : ${WINDOWS_PER_DAY}"
echo "  IMU 采样率   : ${IMU_SAMPLE_HZ} Hz"
echo "  环境采样间隔 : ${ENV_SAMPLE_INTERVAL} s"
echo "  颈温采样间隔 : ${ENV_SAMPLE_INTERVAL} s (同环境传感器)"
echo "  发病天数     : day ${SICK_START_DAY}"
echo "================================================="
echo ""

python sim_devices.py
