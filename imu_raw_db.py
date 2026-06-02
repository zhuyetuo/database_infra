"""
IMU 原始采样点批量加载（TDengine）
====================================
数据库: pet_collar_raw
超级表: imu_raw
子表:   {device_sn}_imu

每行是一个原始 IMU 采样点：
  ts         采样时间戳 UTC ms（TDengine 主时间戳）
  ax/ay/az   加速度 m/s²，范围 ±78.46
  gx/gy/gz   角速度 rad/s，范围 ±17.87
"""

import os
import requests
import numpy as np
from datetime import date, timedelta, datetime, timezone

# ══════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════
TD_HOST      = os.environ.get("TD_HOST", "127.0.0.1")
TD_PORT      = int(os.environ.get("TD_PORT", "6041"))
TD_USER      = os.environ.get("TD_USER", "root")
TD_PASS      = os.environ.get("TD_PASS", "taosdata")
TD_DB        = os.environ.get("TD_DB",   "pet_collar_raw")
IMU_SAMPLE_HZ = int(os.environ.get("IMU_SAMPLE_HZ", "50"))

DAYS       = 180
WARMUP     = 3
START_DATE = date(2024, 1, 1)

ACC_MAX  = 78.46   # m/s²
GYRO_MAX = 17.87   # rad/s

BEHAVIOR_MOVE    = 1
BEHAVIOR_SLEEP   = 2
BEHAVIOR_SCRATCH = 3

# ══════════════════════════════════════════════════════
#  全局时间序列（seed=42）
# ══════════════════════════════════════════════════════
np.random.seed(42)

_temperature = (22 + 13 * np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, DAYS))
                + np.random.normal(0, 1.5, DAYS))

# 信号丢失缺口（seed=7）
np.random.seed(7)
_signal_gap_days: set = set()
_idx = 20
while _idx < 80:
    if np.random.random() < 0.25:
        _glen = np.random.randint(1, 4)
        for _j in range(_idx, min(_idx + _glen, 80)):
            _signal_gap_days.add(_j)
        _idx += _glen + np.random.randint(2, 6)
    else:
        _idx += 1

np.random.seed(42)

# ══════════════════════════════════════════════════════
#  场景定义（24 个）
# ══════════════════════════════════════════════════════
SCENARIOS = [
    {'sn': 'device_sn_1',  'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_2',  'phases': [(0, 60, 10.0, 2.0), (60, 80, 30.0, 4.0), (80, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': (60, 80)},
    {'sn': 'device_sn_3',  'phases': [(0, 60, 10.0, 2.0), (60, 180, 28.0, 4.0)], 'tc': 0.10, 'gaps': [], 'sick': (60, 180)},
    {'sn': 'device_sn_4',  'phases': [(0, 40, 10.0, 2.0), (40, 55, 28.0, 4.0), (55, 120, 10.0, 2.0), (120, 135, 30.0, 4.0), (135, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None, 'sick_episodes': [(40, 55), (120, 135)]},
    {'sn': 'device_sn_5',  'phases': [(0, 60, 10.0, 2.0), (60, 120, 15.0, 2.0), (120, 180, 22.0, 3.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_6',  'phases': [(0, 90, 10.0, 2.0), (90, 180, 25.0, 3.0)], 'tc': 0.10, 'gaps': [], 'sick': (90, 180)},
    {'sn': 'device_sn_7',  'phases': [(0, 50, 10.0, 2.0), (50, 80, 45.0, 6.0), (80, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': (50, 80)},
    {'sn': 'device_sn_8',  'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.35, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_9',  'phases': [(0, 30, 10.0, 2.0), (30, 90, 3.0, 1.0), (90, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_10', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(35, 38, 'unworn')], 'sick': None},
    {'sn': 'device_sn_11', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(40, 45, 'battery')], 'sick': None},
    {'sn': 'device_sn_12', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(30, 65, 'battery')], 'sick': None},
    {'sn': 'device_sn_13', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(d, d + 1, 'signal') for d in sorted(_signal_gap_days)], 'sick': None},
    {'sn': 'device_sn_14', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(50, 58, 'loose')], 'sick': None},
    {'sn': 'device_sn_15', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(88, 92, 'battery')], 'sick': None},
    {'sn': 'device_sn_16', 'phases': [(0, 70, 10.0, 2.0), (70, 90, 35.0, 5.0), (90, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None, 'drift_range': (70, 90)},
    {'sn': 'device_sn_17', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.30, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_18', 'phases': [(0, 60, 10.0, 2.0), (60, 180, 13.0, 2.0)], 'tc': 0.15, 'gaps': [], 'sick': None, 'temp_shift': (60, 5.0)},
    {'sn': 'device_sn_19', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(80, 90, 'unworn')], 'sick': None},
    {'sn': 'device_sn_20', 'phases': [(0, 180, 14.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_21', 'phases': [(0, 180, 15.0, 4.0)], 'tc': 0.10, 'gaps': [], 'sick': None, 'warmup': 7},
    {'sn': 'device_sn_22', 'phases': [(0, 180, 5.0, 1.0)],  'tc': 0.05, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_23', 'phases': [(0, 180, 20.0, 3.0)], 'tc': 0.12, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_24', 'phases': [(0, 180, 4.0, 1.0)],  'tc': 0.08, 'gaps': [], 'sick': None},
]


# ══════════════════════════════════════════════════════
#  TDengine REST API
# ══════════════════════════════════════════════════════

def td_exec(sql: str) -> dict:
    url  = f"http://{TD_HOST}:{TD_PORT}/rest/sql"
    resp = requests.post(url, data=sql.encode('utf-8'),
                         auth=(TD_USER, TD_PASS), timeout=60)
    resp.raise_for_status()
    result = resp.json()
    if result.get('code', 0) != 0:
        raise RuntimeError(f"TDengine error: {result.get('desc', result)}")
    return result


# ══════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════

def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def is_sick_day(day_idx: int, sc: dict) -> bool:
    episodes = sc.get('sick_episodes')
    if episodes:
        return any(s <= day_idx < e for s, e in episodes)
    sick = sc.get('sick')
    if sick:
        return sick[0] <= day_idx < sick[1]
    return False


def scratch_count_for_day(day_idx: int, phases: list, temp: float, tc: float) -> int:
    for s, e, mean, std in phases:
        if s <= day_idx < e:
            return max(0, int(np.random.normal(mean + tc * (temp - 20), std)))
    return 0


def build_gap_map(gaps: list) -> dict:
    gm = {}
    for start, end, reason in gaps:
        for d in range(start, min(end, DAYS)):
            gm[d] = reason
    return gm


# ══════════════════════════════════════════════════════
#  IMU 原始采样点生成
# ══════════════════════════════════════════════════════

def _clip(v, m):
    return round(float(np.clip(v, -m, m)), 2)


def gen_imu_sample(behavior: int, sick_intensity: float = 1.0) -> tuple:
    """生成单个原始采样点 (ax, ay, az, gx, gy, gz)。单位: m/s², rad/s。"""
    si = sick_intensity
    if behavior == BEHAVIOR_SLEEP:
        return (
            _clip(np.random.normal(0,    0.3),  ACC_MAX),
            _clip(np.random.normal(0,    0.3),  ACC_MAX),
            _clip(np.random.normal(9.8,  0.4),  ACC_MAX),
            _clip(np.random.normal(0,    0.05), GYRO_MAX),
            _clip(np.random.normal(0,    0.05), GYRO_MAX),
            _clip(np.random.normal(0,    0.04), GYRO_MAX),
        )
    elif behavior == BEHAVIOR_MOVE:
        return (
            _clip(np.random.normal(0,   4.0),  ACC_MAX),
            _clip(np.random.normal(0,   3.5),  ACC_MAX),
            _clip(np.random.normal(9.8, 5.0),  ACC_MAX),
            _clip(np.random.normal(0,   1.5),  GYRO_MAX),
            _clip(np.random.normal(0,   1.2),  GYRO_MAX),
            _clip(np.random.normal(0,   1.0),  GYRO_MAX),
        )
    else:  # SCRATCH
        return (
            _clip(np.random.normal(0,  12.0 * si), ACC_MAX),
            _clip(np.random.normal(0,   8.0 * si), ACC_MAX),
            _clip(np.random.normal(9.8, 6.0 * si), ACC_MAX),
            _clip(np.random.normal(0,   5.0 * si), GYRO_MAX),
            _clip(np.random.normal(0,   4.0 * si), GYRO_MAX),
            _clip(np.random.normal(0,   3.5 * si), GYRO_MAX),
        )


def gen_day_raw_samples(day_idx: int, n_scratch: int, sick_intensity: float) -> list:
    """
    生成一天的原始 IMU 采样点。
    返回 list of (ts_ms, ax, ay, az, gx, gy, gz)。
    """
    d        = START_DATE + timedelta(days=day_idx)
    day_ts   = to_ts(d)
    step_ms  = int(1000 / IMU_SAMPLE_HZ)
    rows     = []

    s_morn = n_scratch // 3
    s_aftn = n_scratch - s_morn
    segments = [
        (0,         7 * 3600,  BEHAVIOR_SLEEP, BEHAVIOR_MOVE,  0),
        (7 * 3600,  12 * 3600, BEHAVIOR_SLEEP, BEHAVIOR_MOVE,  s_morn),
        (12 * 3600, 14 * 3600, BEHAVIOR_SLEEP, BEHAVIOR_MOVE,  0),
        (14 * 3600, 20 * 3600, BEHAVIOR_SLEEP, BEHAVIOR_MOVE,  s_aftn),
        (20 * 3600, 24 * 3600, BEHAVIOR_SLEEP, BEHAVIOR_MOVE,  0),
    ]

    # 睡眠权重对应列表
    sleep_weights = [0.85, 0.10, 0.80, 0.10, 0.75]

    for (seg_s, seg_e, _, _, n_sc), sleep_w in zip(segments, sleep_weights):
        scratch_times = (
            sorted(np.random.randint(seg_s, seg_e, n_sc).tolist())
            if n_sc > 0 else []
        )
        sc_idx  = 0
        cursor  = seg_s  # seconds within day

        while cursor < seg_e:
            # 判断当前是否进入抓挠片段
            if sc_idx < len(scratch_times) and cursor >= scratch_times[sc_idx]:
                dur_sec = int(np.random.uniform(1, 8))
                btype   = BEHAVIOR_SCRATCH
                si      = sick_intensity
            else:
                btype   = BEHAVIOR_SLEEP if np.random.random() < sleep_w else BEHAVIOR_MOVE
                dur_sec = (int(np.random.uniform(600, 3600))
                           if btype == BEHAVIOR_SLEEP
                           else int(np.random.uniform(60, 900)))
                si      = 1.0

            dur_sec = min(dur_sec, seg_e - cursor)
            if dur_sec <= 0:
                break

            # 展开成逐帧原始采样点
            n_samples = dur_sec * IMU_SAMPLE_HZ
            ts_ms     = day_ts + cursor * 1000
            for i in range(n_samples):
                sample = gen_imu_sample(btype, si)
                rows.append((ts_ms + i * step_ms,) + sample)

            if btype == BEHAVIOR_SCRATCH and sc_idx < len(scratch_times):
                sc_idx += 1
            cursor += dur_sec

    return rows


def build_scenario_rows(sc: dict, seed: int = 42) -> list:
    np.random.seed(seed)
    gap_map  = build_gap_map(sc['gaps'])
    all_rows = []

    for i in range(DAYS):
        if i in gap_map:
            continue
        temp        = float(_temperature[i])
        n_scratch   = scratch_count_for_day(i, sc['phases'], temp, sc['tc'])
        sick_intens = 1.8 if is_sick_day(i, sc) else 1.0
        day_rows    = gen_day_raw_samples(i, n_scratch, sick_intens)
        all_rows.extend(day_rows)

    return all_rows


# ══════════════════════════════════════════════════════
#  数据库操作
# ══════════════════════════════════════════════════════

def create_database():
    td_exec(f"CREATE DATABASE IF NOT EXISTS {TD_DB} KEEP 3650 DURATION 10 COMP 2")
    print(f"[OK] TDengine 数据库 {TD_DB} 已就绪")


def create_stable():
    td_exec(f"""
        CREATE STABLE IF NOT EXISTS {TD_DB}.imu_raw (
            ts  TIMESTAMP,
            ax  FLOAT,
            ay  FLOAT,
            az  FLOAT,
            gx  FLOAT,
            gy  FLOAT,
            gz  FLOAT
        ) TAGS (device_sn BINARY(64))
    """)
    print(f"[OK] 超级表 {TD_DB}.imu_raw 已就绪")


def create_child_table(sn: str):
    td_exec(
        f"CREATE TABLE IF NOT EXISTS {TD_DB}.{sn}_imu "
        f"USING {TD_DB}.imu_raw TAGS ('{sn}')"
    )
    print(f"  [OK] 子表 {sn}_imu 已就绪")


def insert_rows(sn: str, rows: list):
    if not rows:
        print(f"  [{sn}] 无数据，跳过")
        return

    CHUNK = 1000
    total = 0
    for i in range(0, len(rows), CHUNK):
        batch = rows[i: i + CHUNK]
        vals  = " ".join(
            f"({r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]})"
            for r in batch
        )
        td_exec(f"INSERT INTO {TD_DB}.{sn}_imu VALUES {vals}")
        total += len(batch)

    print(f"  [{sn}] 插入 {total:,} 条原始采样点")


# ══════════════════════════════════════════════════════
#  查询验证
# ══════════════════════════════════════════════════════

def query_summary():
    print("\n======= IMU 原始数据概况 =======")
    for sc in SCENARIOS:
        sn = sc['sn']
        try:
            result = td_exec(
                f"SELECT COUNT(*) AS total, "
                f"FIRST(ts) AS earliest, LAST(ts) AS latest "
                f"FROM {TD_DB}.{sn}_imu"
            )
            row = result['data'][0]
            print(f"  {sn:20s}  总={int(row[0]):>12,}  "
                  f"最早={row[1]}  最晚={row[2]}")
        except Exception as e:
            print(f"  {sn}: 查询失败 {e}")


# ══════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════

def main():
    print(f"IMU 采样率: {IMU_SAMPLE_HZ} Hz  |  天数: {DAYS}  |  设备: {len(SCENARIOS)}")
    print(f"预估每设备数据量: {IMU_SAMPLE_HZ * 86400 * DAYS:,} 条\n")

    print("=== 第一步：创建数据库 ===")
    create_database()

    print("\n=== 第二步：创建超级表 ===")
    create_stable()

    print("\n=== 第三步：创建子表 ===")
    for sc in SCENARIOS:
        create_child_table(sc['sn'])

    print("\n=== 第四步：生成并插入数据 ===")
    for idx, sc in enumerate(SCENARIOS):
        print(f"  [{sc['sn']}] 生成中...")
        rows = build_scenario_rows(sc, seed=42 + idx)
        print(f"  [{sc['sn']}] 生成 {len(rows):,} 条，开始插入...")
        insert_rows(sc['sn'], rows)

    print("\n=== 第五步：查询验证 ===")
    query_summary()

    print("\n[完成] IMU 原始数据写入完毕！")


if __name__ == "__main__":
    main()
