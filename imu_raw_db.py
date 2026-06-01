"""
IMU 原始数据库（TDengine）
=====================
数据库: pet_dog_imu
超级表: imu_events
每个设备独立子表: {device_sn}

表结构：每行是一个连续行为片段的 IMU 特征摘要
  ts       行为开始时间 UTC ms（TDengine 主时间戳）
  ts_end   行为结束时间 UTC ms
  ax/ay/az 加速度均值 mg
  gx/gy/gz 陀螺仪均值 deg/s
"""

import requests
import numpy as np
from datetime import date, timedelta, datetime, timezone

# ══════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════
TD_HOST     = "127.0.0.1"
TD_PORT     = 6041
TD_USER     = "root"
TD_PASS     = "taosdata"
IMU_DB      = "pet_dog_imu"

DAYS       = 180
WARMUP     = 3
START_DATE = date(2024, 1, 1)

BEHAVIOR_MOVE    = 1
BEHAVIOR_SLEEP   = 2
BEHAVIOR_SCRATCH = 3

# ══════════════════════════════════════════════════════
#  全局时间序列（seed=42）
# ══════════════════════════════════════════════════════
np.random.seed(42)

_temperature = (22 + 13 * np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, DAYS))
                + np.random.normal(0, 1.5, DAYS))
_humidity    = (65 + 15 * np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, DAYS))
                + np.random.normal(0, 3.0, DAYS))

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
#  IMU 特征生成
# ══════════════════════════════════════════════════════

def gen_imu(behavior: int, sick_intensity: float = 1.0) -> tuple:
    if behavior == BEHAVIOR_SLEEP:
        ax = round(float(np.random.normal(0,   20)),  2)
        ay = round(float(np.random.normal(0,   15)),  2)
        az = round(float(np.random.normal(980, 25)),  2)
        gx = round(float(np.random.normal(0,   2)),   2)
        gy = round(float(np.random.normal(0,   2)),   2)
        gz = round(float(np.random.normal(0,   1.5)), 2)
    elif behavior == BEHAVIOR_MOVE:
        ax = round(float(np.random.normal(40,  150)), 2)
        ay = round(float(np.random.normal(20,  120)), 2)
        az = round(float(np.random.normal(650, 280)), 2)
        gx = round(float(np.random.normal(0,   90)),  2)
        gy = round(float(np.random.normal(0,   70)),  2)
        gz = round(float(np.random.normal(0,   55)),  2)
    else:
        si = sick_intensity
        ax = round(float(np.random.normal(180 * si, 80)),  2)
        ay = round(float(np.random.normal(40,        60)),  2)
        az = round(float(np.random.normal(800,       180)), 2)
        gx = round(float(np.random.normal(0, 130 * si)),    2)
        gy = round(float(np.random.normal(0,  90)),          2)
        gz = round(float(np.random.normal(0,  70)),          2)
    return ax, ay, az, gx, gy, gz


def gen_day_events(day_idx: int, n_scratch: int, sick_intensity: float) -> list:
    d      = START_DATE + timedelta(days=day_idx)
    day_ts = to_ts(d)
    rows   = []

    s_morn = n_scratch // 3
    s_aftn = n_scratch - s_morn
    segments = [
        (0,         7 * 3600,  0.85, 0.15, 0),
        (7 * 3600,  12 * 3600, 0.10, 0.85, s_morn),
        (12 * 3600, 14 * 3600, 0.80, 0.20, 0),
        (14 * 3600, 20 * 3600, 0.10, 0.80, s_aftn),
        (20 * 3600, 24 * 3600, 0.75, 0.25, 0),
    ]

    for seg_s, seg_e, sleep_w, _, n_sc in segments:
        cursor = seg_s
        scratch_times = (
            sorted(np.random.randint(seg_s, seg_e, n_sc).tolist())
            if n_sc > 0 else []
        )
        sc_idx = 0

        while cursor < seg_e:
            if sc_idx < len(scratch_times) and cursor >= scratch_times[sc_idx]:
                dur_ms = int(np.random.uniform(1000, 8000))
                s_ts   = day_ts + scratch_times[sc_idx] * 1000
                e_ts   = s_ts + dur_ms
                imu    = gen_imu(BEHAVIOR_SCRATCH, sick_intensity)
                rows.append((s_ts, e_ts) + imu)
                cursor  = scratch_times[sc_idx] + dur_ms // 1000 + 1
                sc_idx += 1
            else:
                btype   = BEHAVIOR_SLEEP if np.random.random() < sleep_w else BEHAVIOR_MOVE
                dur_sec = (int(np.random.uniform(600, 3600))
                           if btype == BEHAVIOR_SLEEP
                           else int(np.random.uniform(60, 900)))
                dur_sec = min(dur_sec, seg_e - cursor)
                if dur_sec <= 0:
                    break
                s_ts = day_ts + cursor * 1000
                e_ts = s_ts + dur_sec * 1000
                imu  = gen_imu(btype, 1.0)
                rows.append((s_ts, e_ts) + imu)
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
        day_rows    = gen_day_events(i, n_scratch, sick_intens)
        all_rows.extend(day_rows)

    return all_rows


# ══════════════════════════════════════════════════════
#  数据库操作
# ══════════════════════════════════════════════════════

def create_database():
    td_exec(f"CREATE DATABASE IF NOT EXISTS {IMU_DB} KEEP 3650 DURATION 10 COMP 2")
    print(f"[OK] TDengine 数据库 {IMU_DB} 已就绪")


def create_stable():
    td_exec(f"""
        CREATE STABLE IF NOT EXISTS {IMU_DB}.imu_events (
            ts     TIMESTAMP,
            ts_end BIGINT,
            ax     FLOAT,
            ay     FLOAT,
            az     FLOAT,
            gx     FLOAT,
            gy     FLOAT,
            gz     FLOAT
        ) TAGS (device_sn BINARY(32))
    """)
    print(f"[OK] 超级表 {IMU_DB}.imu_events 已就绪")


def create_child_table(sn: str):
    td_exec(
        f"CREATE TABLE IF NOT EXISTS {IMU_DB}.{sn} "
        f"USING {IMU_DB}.imu_events TAGS ('{sn}')"
    )
    print(f"  [OK] 子表 {sn} 已就绪")


def insert_rows(sn: str, rows: list):
    if not rows:
        print(f"  [{sn}] 无数据，跳过")
        return

    batch_size = 500
    total      = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i: i + batch_size]
        vals  = " ".join(
            f"({r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},{r[7]})"
            for r in batch
        )
        td_exec(f"INSERT INTO {IMU_DB}.{sn} VALUES {vals}")
        total += len(batch)

    print(f"  [{sn}] 插入 {total} 条 IMU 事件记录")


# ══════════════════════════════════════════════════════
#  查询验证
# ══════════════════════════════════════════════════════

def query_summary():
    print("\n======= IMU 数据概况 =======")
    for sc in SCENARIOS:
        sn = sc['sn']
        try:
            result = td_exec(
                f"SELECT COUNT(*) AS total, "
                f"FIRST(ts) AS earliest, LAST(ts) AS latest "
                f"FROM {IMU_DB}.{sn}"
            )
            row = result['data'][0]
            print(f"  {sn:20s}  总={int(row[0]):6d}  "
                  f"最早={row[1]}  最晚={row[2]}")
        except Exception as e:
            print(f"  {sn}: 查询失败 {e}")


# ══════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════

def main():
    print("=== 第一步：创建数据库 ===")
    create_database()

    print("\n=== 第二步：创建超级表 ===")
    create_stable()

    print("\n=== 第三步：创建子表 ===")
    for sc in SCENARIOS:
        create_child_table(sc['sn'])

    print("\n=== 第四步：生成并插入数据 ===")
    for idx, sc in enumerate(SCENARIOS):
        rows = build_scenario_rows(sc, seed=42 + idx)
        print(f"  [{sc['sn']}] 生成 {len(rows)} 条事件，开始插入...")
        insert_rows(sc['sn'], rows)

    print("\n=== 第五步：查询验证 ===")
    query_summary()

    print("\n[完成] IMU 数据库写入完毕！")


if __name__ == "__main__":
    main()
