"""
传感器原始数据批量加载（TDengine）
====================================
数据库: pet_collar_raw

超级表          子表命名             内容
imu_raw        d{n}_imu         IMU 6轴原始采样点  50Hz
env_raw        d{n}_env         环境温度 + 湿度    每60s
neck_temp_raw  d{n}_neck        脖颈温度           每300s
"""

import os
import math
import time
import requests
import numpy as np
from datetime import date, timedelta, datetime, timezone

# ══════════════════════════════════════════════════════
#  加载 sim_config.env
# ══════════════════════════════════════════════════════
def _load_config(path: str = "sim_config.env"):
    here = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.join(here, path)
    if not os.path.exists(fpath):
        return
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_config()

# ══════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════
TD_HOST      = os.environ.get("TD_HOST", "127.0.0.1")
TD_PORT      = int(os.environ.get("TD_PORT", "6041"))
TD_USER      = os.environ.get("TD_USER", "root")
TD_PASS      = os.environ.get("TD_PASS", "taosdata")
TD_DB        = os.environ.get("TD_DB",   "pet_collar_raw")
IMU_SAMPLE_HZ        = int(os.environ.get("IMU_SAMPLE_HZ",        "50"))
ENV_SAMPLE_INTERVAL  = int(os.environ.get("ENV_SAMPLE_INTERVAL",  "60"))  # 秒
NECK_SAMPLE_INTERVAL = int(os.environ.get("NECK_SAMPLE_INTERVAL", "60"))  # 秒

DAYS       = 3      # 验证通过后改为 180
START_DATE = date(2024, 1, 1)

ACC_MAX  = 78.46   # m/s²
GYRO_MAX = 17.87   # rad/s

BEHAVIOR_MOVE    = 1
BEHAVIOR_SLEEP   = 2
BEHAVIOR_SCRATCH = 3

# ══════════════════════════════════════════════════════
#  全局时间序列
# ══════════════════════════════════════════════════════
np.random.seed(42)
_N = 180
_temperature = (22 + 13 * np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, _N))
                + np.random.normal(0, 1.5, _N))
_humidity    = (65 + 15 * np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, _N))
                + np.random.normal(0, 3.0, _N))

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
    {'device_id': 1,  'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'device_id': 2,  'phases': [(0, 60, 10.0, 2.0), (60, 80, 30.0, 4.0), (80, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': (60, 80)},
    {'device_id': 3,  'phases': [(0, 60, 10.0, 2.0), (60, 180, 28.0, 4.0)], 'tc': 0.10, 'gaps': [], 'sick': (60, 180)},
    {'device_id': 4,  'phases': [(0, 40, 10.0, 2.0), (40, 55, 28.0, 4.0), (55, 120, 10.0, 2.0), (120, 135, 30.0, 4.0), (135, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None, 'sick_episodes': [(40, 55), (120, 135)]},
    {'device_id': 5,  'phases': [(0, 60, 10.0, 2.0), (60, 120, 15.0, 2.0), (120, 180, 22.0, 3.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'device_id': 6,  'phases': [(0, 90, 10.0, 2.0), (90, 180, 25.0, 3.0)], 'tc': 0.10, 'gaps': [], 'sick': (90, 180)},
    {'device_id': 7,  'phases': [(0, 50, 10.0, 2.0), (50, 80, 45.0, 6.0), (80, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': (50, 80)},
    {'device_id': 8,  'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.35, 'gaps': [], 'sick': None},
    {'device_id': 9,  'phases': [(0, 30, 10.0, 2.0), (30, 90, 3.0, 1.0), (90, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'device_id': 10, 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(35, 38, 'unworn')], 'sick': None},
    {'device_id': 11, 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(40, 45, 'battery')], 'sick': None},
    {'device_id': 12, 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(30, 65, 'battery')], 'sick': None},
    {'device_id': 13, 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(d, d + 1, 'signal') for d in sorted(_signal_gap_days)], 'sick': None},
    {'device_id': 14, 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(50, 58, 'loose')], 'sick': None},
    {'device_id': 15, 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(88, 92, 'battery')], 'sick': None},
    {'device_id': 16, 'phases': [(0, 70, 10.0, 2.0), (70, 90, 35.0, 5.0), (90, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None, 'drift_range': (70, 90)},
    {'device_id': 17, 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.30, 'gaps': [], 'sick': None},
    {'device_id': 18, 'phases': [(0, 60, 10.0, 2.0), (60, 180, 13.0, 2.0)], 'tc': 0.15, 'gaps': [], 'sick': None, 'temp_shift': (60, 5.0)},
    {'device_id': 19, 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(80, 90, 'unworn')], 'sick': None},
    {'device_id': 20, 'phases': [(0, 180, 14.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'device_id': 21, 'phases': [(0, 180, 15.0, 4.0)], 'tc': 0.10, 'gaps': [], 'sick': None, 'warmup': 7},
    {'device_id': 22, 'phases': [(0, 180, 5.0, 1.0)],  'tc': 0.05, 'gaps': [], 'sick': None},
    {'device_id': 23, 'phases': [(0, 180, 20.0, 3.0)], 'tc': 0.12, 'gaps': [], 'sick': None},
    {'device_id': 24, 'phases': [(0, 180, 4.0, 1.0)],  'tc': 0.08, 'gaps': [], 'sick': None},
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
    return bool(sick and sick[0] <= day_idx < sick[1])


def scratch_count_for_day(day_idx: int, phases: list, temp: float, tc: float) -> int:
    for s, e, mean, std in phases:
        if s <= day_idx < e:
            return max(0, int(np.random.normal(mean + tc * (temp - 20), std)))
    return 0


def build_gap_map(gaps: list) -> dict:
    gm = {}
    for start, end, reason in gaps:
        for d in range(start, min(end, _N)):
            gm[d] = reason
    return gm


# ══════════════════════════════════════════════════════
#  IMU 原始采样点生成
# ══════════════════════════════════════════════════════

def _clip(v, m):
    return round(float(np.clip(v, -m, m)), 2)


def gen_imu_sample(behavior: int, sick_intensity: float = 1.0) -> tuple:
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


def gen_imu_day(day_idx: int, n_scratch: int, sick_intensity: float) -> list:
    """返回 list of (ts_ms, ax, ay, az, gx, gy, gz)"""
    d       = START_DATE + timedelta(days=day_idx)
    day_ts  = to_ts(d)
    step_ms = int(1000 / IMU_SAMPLE_HZ)
    rows    = []

    s_morn = n_scratch // 3
    s_aftn = n_scratch - s_morn
    segments = [
        (0,         7 * 3600,  0.85, s_morn * 0),
        (7 * 3600,  12 * 3600, 0.10, s_morn),
        (12 * 3600, 14 * 3600, 0.80, 0),
        (14 * 3600, 20 * 3600, 0.10, s_aftn),
        (20 * 3600, 24 * 3600, 0.75, 0),
    ]

    for seg_s, seg_e, sleep_w, n_sc in segments:
        scratch_times = (
            sorted(np.random.randint(seg_s, seg_e, int(n_sc)).tolist())
            if n_sc > 0 else []
        )
        sc_idx = 0
        cursor = seg_s

        while cursor < seg_e:
            if sc_idx < len(scratch_times) and cursor >= scratch_times[sc_idx]:
                dur_sec = int(np.random.uniform(1, 8))
                btype   = BEHAVIOR_SCRATCH
                si      = sick_intensity
                sc_idx += 1
            else:
                btype   = BEHAVIOR_SLEEP if np.random.random() < sleep_w else BEHAVIOR_MOVE
                dur_sec = (int(np.random.uniform(600, 3600))
                           if btype == BEHAVIOR_SLEEP
                           else int(np.random.uniform(60, 900)))
                si      = 1.0

            dur_sec = min(dur_sec, seg_e - cursor)
            if dur_sec <= 0:
                break

            ts_ms     = day_ts + cursor * 1000
            n_samples = dur_sec * IMU_SAMPLE_HZ
            for i in range(n_samples):
                rows.append((ts_ms + i * step_ms,) + gen_imu_sample(btype, si))

            cursor += dur_sec

    return rows


# ══════════════════════════════════════════════════════
#  环境温湿度 + 脖颈温度生成
# ══════════════════════════════════════════════════════

def gen_env_day(day_idx: int) -> list:
    """返回 list of (ts_ms, env_temp, env_humi)，每 ENV_SAMPLE_INTERVAL 秒一条"""
    d      = START_DATE + timedelta(days=day_idx)
    day_ts = to_ts(d)
    doy    = d.timetuple().tm_yday
    t_base = 22 + 13 * math.sin((doy - 80) / 365 * 2 * math.pi)
    h_base = 65 + 15 * math.sin((doy - 80) / 365 * 2 * math.pi)

    rows = []
    for s in range(0, 86400, ENV_SAMPLE_INTERVAL):
        env_temp = round(t_base + np.random.normal(0, 1.5), 1)
        env_humi = round(h_base + np.random.normal(0, 3.0), 1)
        rows.append((day_ts + s * 1000, env_temp, env_humi))
    return rows


def gen_neck_day(day_idx: int, sick_intensity: float) -> list:
    """返回 list of (ts_ms, neck_temp)，每 NECK_SAMPLE_INTERVAL 秒一条"""
    d      = START_DATE + timedelta(days=day_idx)
    day_ts = to_ts(d)

    rows = []
    for s in range(0, 86400, NECK_SAMPLE_INTERVAL):
        if sick_intensity > 1.3:
            neck_temp = round(38.5 + np.random.uniform(0.0, 0.8) * (sick_intensity - 0.3), 2)
        else:
            neck_temp = round(37.5 + np.random.uniform(-0.3, 0.3), 2)
        rows.append((day_ts + s * 1000, neck_temp))
    return rows


# ══════════════════════════════════════════════════════
#  数据库初始化
# ══════════════════════════════════════════════════════

def init_db():
    td_exec(f"CREATE DATABASE IF NOT EXISTS {TD_DB} KEEP 3650 DURATION 10 COMP 2")

    td_exec(f"""
        CREATE STABLE IF NOT EXISTS {TD_DB}.imu_raw (
            ts  TIMESTAMP,
            ax  FLOAT, ay  FLOAT, az  FLOAT,
            gx  FLOAT, gy  FLOAT, gz  FLOAT
        ) TAGS (device_id BIGINT)
    """)

    td_exec(f"""
        CREATE STABLE IF NOT EXISTS {TD_DB}.env_raw (
            ts        TIMESTAMP,
            env_temp  FLOAT,
            env_humi  FLOAT
        ) TAGS (device_id BIGINT)
    """)

    td_exec(f"""
        CREATE STABLE IF NOT EXISTS {TD_DB}.neck_temp_raw (
            ts        TIMESTAMP,
            neck_temp FLOAT
        ) TAGS (device_id BIGINT)
    """)

    for sc in SCENARIOS:
        device_id = sc['device_id']
        td_exec(f"CREATE TABLE IF NOT EXISTS {TD_DB}.d{device_id}_imu  USING {TD_DB}.imu_raw       TAGS ({device_id})")
        td_exec(f"CREATE TABLE IF NOT EXISTS {TD_DB}.d{device_id}_env  USING {TD_DB}.env_raw        TAGS ({device_id})")
        td_exec(f"CREATE TABLE IF NOT EXISTS {TD_DB}.d{device_id}_neck USING {TD_DB}.neck_temp_raw  TAGS ({device_id})")

    print(f"[OK] 数据库 & 表结构已就绪（{len(SCENARIOS)} 个设备）")


# ══════════════════════════════════════════════════════
#  数据写入
# ══════════════════════════════════════════════════════

IMU_CHUNK  = 16000
ENV_CHUNK  = 2000
NECK_CHUNK = 1000


def insert_imu(device_id: int, rows: list):
    for i in range(0, len(rows), IMU_CHUNK):
        b    = rows[i: i + IMU_CHUNK]
        vals = " ".join(f"({r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]})" for r in b)
        td_exec(f"INSERT INTO {TD_DB}.d{device_id}_imu VALUES {vals}")


def insert_env(device_id: int, rows: list):
    for i in range(0, len(rows), ENV_CHUNK):
        b    = rows[i: i + ENV_CHUNK]
        vals = " ".join(f"({r[0]},{r[1]},{r[2]})" for r in b)
        td_exec(f"INSERT INTO {TD_DB}.d{device_id}_env VALUES {vals}")


def insert_neck(device_id: int, rows: list):
    for i in range(0, len(rows), NECK_CHUNK):
        b    = rows[i: i + NECK_CHUNK]
        vals = " ".join(f"({r[0]},{r[1]})" for r in b)
        td_exec(f"INSERT INTO {TD_DB}.d{device_id}_neck VALUES {vals}")


# ══════════════════════════════════════════════════════
#  场景数据生成
# ══════════════════════════════════════════════════════

def _bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total)
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"


def load_scenario(sc: dict, seed: int = 42, dev_idx: int = 0, dev_total: int = 1):
    device_id = sc['device_id']
    gap_map   = build_gap_map(sc['gaps'])
    np.random.seed(seed)

    imu_total = env_total = neck_total = 0
    t0 = time.time()
    gen_s = ins_s = 0.0

    valid_days = [i for i in range(DAYS) if i not in gap_map]

    print(f"\n  [{dev_idx+1:>2}/{dev_total}] device_id={device_id}")
    for idx, i in enumerate(valid_days):
        temp        = float(_temperature[i])
        n_scratch   = scratch_count_for_day(i, sc['phases'], temp, sc['tc'])
        sick_intens = 1.8 if is_sick_day(i, sc) else 1.0

        t_gen = time.time()
        imu_rows  = gen_imu_day(i, n_scratch, sick_intens)
        env_rows  = gen_env_day(i)
        neck_rows = gen_neck_day(i, sick_intens)
        gen_s += time.time() - t_gen

        t_ins = time.time()
        insert_imu(device_id, imu_rows)
        insert_env(device_id, env_rows)
        insert_neck(device_id, neck_rows)
        ins_s += time.time() - t_ins

        imu_total  += len(imu_rows)
        env_total  += len(env_rows)
        neck_total += len(neck_rows)

        elapsed = time.time() - t0
        done    = idx + 1
        eta     = (elapsed / done) * (len(valid_days) - done)
        bar     = _bar(done, len(valid_days))
        print(f"\r    {bar}  gen={gen_s:.1f}s ins={ins_s:.1f}s  ETA {eta:.0f}s   ",
              end="", flush=True)

    elapsed = time.time() - t0
    print(f"\r    完成  imu={imu_total:>12,}  env={env_total:>6,}  neck={neck_total:>4,}"
          f"  耗时 {elapsed:.1f}s (生成 {gen_s:.1f}s / 插入 {ins_s:.1f}s)")


# ══════════════════════════════════════════════════════
#  查询验证
# ══════════════════════════════════════════════════════

def query_summary():
    print("\n======= 数据概况 =======")
    print(f"  {'设备':<20}  {'IMU条数':>12}  {'ENV条数':>8}  {'NECK条数':>8}")
    for sc in SCENARIOS:
        device_id = sc['device_id']
        try:
            imu  = td_exec(f"SELECT COUNT(*) FROM {TD_DB}.d{device_id}_imu")['data'][0][0]
            env  = td_exec(f"SELECT COUNT(*) FROM {TD_DB}.d{device_id}_env")['data'][0][0]
            neck = td_exec(f"SELECT COUNT(*) FROM {TD_DB}.d{device_id}_neck")['data'][0][0]
            print(f"  device_id={device_id:<14}  {int(imu):>12,}  {int(env):>8,}  {int(neck):>8,}")
        except Exception as e:
            print(f"  device_id={device_id}: 查询失败 {e}")


# ══════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════

def main():
    imu_est  = IMU_SAMPLE_HZ * 86400 * DAYS
    env_est  = 86400 // ENV_SAMPLE_INTERVAL * DAYS
    neck_est = 86400 // NECK_SAMPLE_INTERVAL * DAYS

    print("=" * 60)
    print(f"  设备数    : {len(SCENARIOS)}")
    print(f"  天数      : {DAYS}  （完整跑改为 180）")
    print(f"  IMU       : {IMU_SAMPLE_HZ} Hz  → 预估 {imu_est:,} 条/设备")
    print(f"  ENV       : 每 {ENV_SAMPLE_INTERVAL}s   → 预估 {env_est:,} 条/设备")
    print(f"  NECK      : 每 {NECK_SAMPLE_INTERVAL}s  → 预估 {neck_est:,} 条/设备")
    print("=" * 60)

    print("\n[1] 初始化数据库 & 表结构...")
    init_db()

    print("\n[2] 生成并写入数据...")
    t_all = time.time()
    for idx, sc in enumerate(SCENARIOS):
        load_scenario(sc, seed=42 + idx, dev_idx=idx, dev_total=len(SCENARIOS))
    print(f"\n  全部设备总耗时: {time.time() - t_all:.1f}s")

    print("\n[3] 查询验证...")
    query_summary()

    print("\n[完成]")


if __name__ == "__main__":
    main()
