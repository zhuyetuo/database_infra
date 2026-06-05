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
import threading
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
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

DAYS       = 180
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


def scratch_count_for_day(day_idx: int, phases: list, temp: float, tc: float,
                          rng: np.random.RandomState) -> int:
    for s, e, mean, std in phases:
        if s <= day_idx < e:
            return max(0, int(rng.normal(mean + tc * (temp - 20), std)))
    return 0


def build_gap_map(gaps: list) -> dict:
    gm = {}
    for start, end, reason in gaps:
        for d in range(start, min(end, _N)):
            gm[d] = reason
    return gm


# ══════════════════════════════════════════════════════
#  IMU 原始采样点生成（向量化批量生成）
# ══════════════════════════════════════════════════════

_IMU_CLIPS = np.array([ACC_MAX, ACC_MAX, ACC_MAX, GYRO_MAX, GYRO_MAX, GYRO_MAX])
_IMU_MEANS = {
    BEHAVIOR_SLEEP:   np.array([0.0, 0.0, 9.8, 0.0,  0.0,  0.0]),
    BEHAVIOR_MOVE:    np.array([0.0, 0.0, 9.8, 0.0,  0.0,  0.0]),
    BEHAVIOR_SCRATCH: np.array([0.0, 0.0, 9.8, 0.0,  0.0,  0.0]),
}
_IMU_STDS_BASE = {
    BEHAVIOR_SLEEP:   np.array([0.3,  0.3,  0.4,  0.05, 0.05, 0.04]),
    BEHAVIOR_MOVE:    np.array([4.0,  3.5,  5.0,  1.5,  1.2,  1.0]),
    BEHAVIOR_SCRATCH: np.array([12.0, 8.0,  6.0,  5.0,  4.0,  3.5]),
}


def gen_imu_batch(n: int, behavior: int, si: float, rng: np.random.RandomState) -> np.ndarray:
    """返回 shape (n, 6)，已 clip 并四舍五入到 2 位小数"""
    means = _IMU_MEANS[behavior]
    stds  = _IMU_STDS_BASE[behavior] * (si if behavior == BEHAVIOR_SCRATCH else 1.0)
    data  = rng.normal(means, stds, size=(n, 6))
    data  = np.clip(data, -_IMU_CLIPS, _IMU_CLIPS)
    return np.round(data, 2)


def gen_imu_day(day_idx: int, n_scratch: int, sick_intensity: float,
                rng: np.random.RandomState) -> tuple:
    """返回 (ts_arr: int64 (N,), imu_arr: float64 (N,6))，不转 Python list"""
    d       = START_DATE + timedelta(days=day_idx)
    day_ts  = to_ts(d)
    step_ms = int(1000 / IMU_SAMPLE_HZ)

    s_morn = n_scratch // 3
    s_aftn = n_scratch - s_morn
    segments = [
        (0,         7 * 3600,  0.85, s_morn * 0),
        (7 * 3600,  12 * 3600, 0.10, s_morn),
        (12 * 3600, 14 * 3600, 0.80, 0),
        (14 * 3600, 20 * 3600, 0.10, s_aftn),
        (20 * 3600, 24 * 3600, 0.75, 0),
    ]

    blocks = []
    for seg_s, seg_e, sleep_w, n_sc in segments:
        scratch_times = (
            sorted(rng.randint(seg_s, seg_e, int(n_sc)).tolist())
            if n_sc > 0 else []
        )
        sc_idx = 0
        cursor = seg_s

        while cursor < seg_e:
            if sc_idx < len(scratch_times) and cursor >= scratch_times[sc_idx]:
                dur_sec = int(rng.uniform(1, 8))
                btype   = BEHAVIOR_SCRATCH
                si      = sick_intensity
                sc_idx += 1
            else:
                btype   = BEHAVIOR_SLEEP if rng.random() < sleep_w else BEHAVIOR_MOVE
                dur_sec = (int(rng.uniform(600, 3600))
                           if btype == BEHAVIOR_SLEEP
                           else int(rng.uniform(60, 900)))
                si      = 1.0

            dur_sec = min(dur_sec, seg_e - cursor)
            if dur_sec <= 0:
                break

            blocks.append((day_ts + cursor * 1000, dur_sec * IMU_SAMPLE_HZ, btype, si))
            cursor += dur_sec

    ts_parts  = []
    imu_parts = []
    for ts_start, n_samples, btype, si in blocks:
        imu = gen_imu_batch(n_samples, btype, si, rng)
        ts  = np.arange(n_samples, dtype=np.int64) * step_ms + ts_start
        ts_parts.append(ts)
        imu_parts.append(imu)

    return np.concatenate(ts_parts), np.concatenate(imu_parts, axis=0)


# ══════════════════════════════════════════════════════
#  环境温湿度 + 脖颈温度生成
# ══════════════════════════════════════════════════════

def gen_env_day(day_idx: int, rng: np.random.RandomState) -> list:
    """返回 list of (ts_ms, env_temp, env_humi)，每 ENV_SAMPLE_INTERVAL 秒一条"""
    d      = START_DATE + timedelta(days=day_idx)
    day_ts = to_ts(d)
    doy    = d.timetuple().tm_yday
    t_base = 22 + 13 * math.sin((doy - 80) / 365 * 2 * math.pi)
    h_base = 65 + 15 * math.sin((doy - 80) / 365 * 2 * math.pi)

    secs   = np.arange(0, 86400, ENV_SAMPLE_INTERVAL)
    n      = len(secs)
    temps  = np.round(t_base + rng.normal(0, 1.5, n), 1)
    humis  = np.round(h_base + rng.normal(0, 3.0, n), 1)
    ts_arr = (day_ts + secs * 1000).astype(np.int64)
    return list(zip(ts_arr.tolist(), temps.tolist(), humis.tolist()))


def gen_neck_day(day_idx: int, sick_intensity: float, rng: np.random.RandomState) -> list:
    """返回 list of (ts_ms, neck_temp)，每 NECK_SAMPLE_INTERVAL 秒一条"""
    d      = START_DATE + timedelta(days=day_idx)
    day_ts = to_ts(d)

    secs   = np.arange(0, 86400, NECK_SAMPLE_INTERVAL)
    n      = len(secs)
    ts_arr = (day_ts + secs * 1000).astype(np.int64)
    if sick_intensity > 1.3:
        neck = np.round(38.5 + rng.uniform(0.0, 0.8, n) * (sick_intensity - 0.3), 2)
    else:
        neck = np.round(37.5 + rng.uniform(-0.3, 0.3, n), 2)
    return list(zip(ts_arr.tolist(), neck.tolist()))


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

IMU_CHUNK      = 16000
ENV_CHUNK      = 2000
NECK_CHUNK     = 1000
PARALLEL_DEVS  = int(os.environ.get("PARALLEL_DEVS", "4"))  # 并行设备数


def _imu_vals_str(ts_chunk: np.ndarray, imu_chunk: np.ndarray) -> str:
    """numpy 数组直接转 SQL VALUES 字符串，避免中间 tuple list"""
    ts_list  = ts_chunk.tolist()
    ax = imu_chunk[:, 0].tolist(); ay = imu_chunk[:, 1].tolist()
    az = imu_chunk[:, 2].tolist(); gx = imu_chunk[:, 3].tolist()
    gy = imu_chunk[:, 4].tolist(); gz = imu_chunk[:, 5].tolist()
    return " ".join(
        f"({ts_list[j]},{ax[j]},{ay[j]},{az[j]},{gx[j]},{gy[j]},{gz[j]})"
        for j in range(len(ts_list))
    )


def insert_imu(device_id: int, ts_arr: np.ndarray, imu_arr: np.ndarray):
    n = len(ts_arr)
    for i in range(0, n, IMU_CHUNK):
        vals = _imu_vals_str(ts_arr[i:i + IMU_CHUNK], imu_arr[i:i + IMU_CHUNK])
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

_print_lock = threading.Lock()


def _bar(done: int, total: int, width: int = 20) -> str:
    filled = int(width * done / total)
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"


def load_scenario(sc: dict, seed: int = 42, dev_idx: int = 0, dev_total: int = 1):
    device_id = sc['device_id']
    gap_map   = build_gap_map(sc['gaps'])
    rng       = np.random.RandomState(seed)   # 线程独立 RNG，不影响全局状态

    imu_total = env_total = neck_total = 0
    t0 = time.time()
    gen_s = ins_s = 0.0

    valid_days = [i for i in range(DAYS) if i not in gap_map]

    with _print_lock:
        print(f"  [开始] device_id={device_id}", flush=True)

    for idx, i in enumerate(valid_days):
        temp        = float(_temperature[i])
        n_scratch   = scratch_count_for_day(i, sc['phases'], temp, sc['tc'], rng)
        sick_intens = 1.8 if is_sick_day(i, sc) else 1.0

        t_gen = time.time()
        ts_arr, imu_arr = gen_imu_day(i, n_scratch, sick_intens, rng)
        env_rows        = gen_env_day(i, rng)
        neck_rows       = gen_neck_day(i, sick_intens, rng)
        gen_s += time.time() - t_gen

        t_ins = time.time()
        insert_imu(device_id, ts_arr, imu_arr)
        insert_env(device_id, env_rows)
        insert_neck(device_id, neck_rows)
        ins_s += time.time() - t_ins

        imu_total  += len(ts_arr)
        env_total  += len(env_rows)
        neck_total += len(neck_rows)

        # 每 10 天打一次进度，减少锁争用
        if (idx + 1) % 10 == 0 or (idx + 1) == len(valid_days):
            elapsed = time.time() - t0
            done    = idx + 1
            eta     = (elapsed / done) * (len(valid_days) - done)
            with _print_lock:
                print(f"  dev={device_id:>2} {_bar(done, len(valid_days))}"
                      f"  gen={gen_s:.0f}s ins={ins_s:.0f}s ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    with _print_lock:
        print(f"  [完成] device_id={device_id:<3} imu={imu_total:>12,}"
              f"  env={env_total:>6,}  neck={neck_total:>4,}"
              f"  耗时 {elapsed:.1f}s (gen={gen_s:.1f}s ins={ins_s:.1f}s)", flush=True)
    return device_id, imu_total, env_total, neck_total, elapsed


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

    print(f"\n[2] 生成并写入数据（{PARALLEL_DEVS} 设备并行）...")
    t_all = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL_DEVS) as pool:
        futures = {
            pool.submit(load_scenario, sc, 42 + idx, idx, len(SCENARIOS)): sc['device_id']
            for idx, sc in enumerate(SCENARIOS)
        }
        for f in as_completed(futures):
            f.result()   # 传播异常
    print(f"\n  全部设备总耗时: {time.time() - t_all:.1f}s")

    print("\n[3] 查询验证...")
    query_summary()

    print("\n[完成]")


if __name__ == "__main__":
    main()
