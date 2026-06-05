"""
imu_scale_db.py — 万设备级批量加载
===============================================
数据库  : pet_collar_raw (同 imu_raw_db.py)
默认规模 : 10,000 设备 × 180 天

速度架构
  生成 : ThreadPoolExecutor (GEN_WORKERS 个线程)
         numpy 向量化；GIL 对 numpy 基本无效，多线程可真正并行
  写入 : asyncio + aiohttp (INSERT_CONC 路并发 HTTP)
  批量 : 多表 INSERT — 每条 SQL 打包 SQL_PARTS 个设备-时间块
         单 SQL 约 900KB，最大化每次 HTTP 的数据量

关键参数（可通过环境变量覆盖）
  NUM_DEVICES   设备总数          默认 10000
  SCALE_DAYS    天数              默认 180
  GEN_WORKERS   生成线程数        默认 cpu_count
  INSERT_CONC   并发 HTTP 请求数  默认 64
  GROUP_SIZE    每次生成的设备组  默认 8
  IMU_CHUNK     每设备每块的行数  默认 2200  (~114KB/块)
  SQL_PARTS     每条 SQL 的块数   默认 8     (~912KB/SQL ≤ 1MB)
"""

import os
import sys
import math
import time
import asyncio
import threading
import aiohttp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
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
TD_HOST  = os.environ.get("TD_HOST",  "127.0.0.1")
TD_PORT  = int(os.environ.get("TD_PORT",  "6041"))
TD_USER  = os.environ.get("TD_USER",  "root")
TD_PASS  = os.environ.get("TD_PASS",  "taosdata")
TD_DB    = os.environ.get("TD_DB",    "pet_collar_raw")

ENV_SAMPLE_INTERVAL  = int(os.environ.get("ENV_SAMPLE_INTERVAL",  "60"))
NECK_SAMPLE_INTERVAL = int(os.environ.get("NECK_SAMPLE_INTERVAL", "60"))
IMU_HZ               = int(os.environ.get("IMU_SAMPLE_HZ",        "50"))

NUM_DEVICES  = int(os.environ.get("NUM_DEVICES",  "10000"))
DAYS         = int(os.environ.get("SCALE_DAYS",   "180"))
GEN_WORKERS  = int(os.environ.get("GEN_WORKERS",  str(os.cpu_count() or 4)))
INSERT_CONC  = int(os.environ.get("INSERT_CONC",  "64"))
GROUP_SIZE   = int(os.environ.get("GROUP_SIZE",   "8"))   # devices per gen task
IMU_CHUNK    = int(os.environ.get("IMU_CHUNK",    "2200")) # rows per device per part
SQL_PARTS    = int(os.environ.get("SQL_PARTS",    "8"))    # parts per INSERT SQL

START_DATE = date(2024, 1, 1)
ACC_MAX    = 78.46
GYRO_MAX   = 17.87

TD_REST_URL = f"http://{TD_HOST}:{TD_PORT}/rest/sql"
TD_AUTH     = aiohttp.BasicAuth(TD_USER, TD_PASS)

# ══════════════════════════════════════════════════════
#  全局温度序列（与 imu_raw_db.py 一致）
# ══════════════════════════════════════════════════════
_rng_global = np.random.RandomState(42)
_temperature = (22 + 13 * np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, DAYS))
                + _rng_global.normal(0, 1.5, DAYS))

# ══════════════════════════════════════════════════════
#  向量化 IMU 生成常量
# ══════════════════════════════════════════════════════
_CLIPS = np.array([ACC_MAX] * 3 + [GYRO_MAX] * 3)
_MEANS = np.array([0., 0., 9.8, 0., 0., 0.])
_STDS  = {
    0: np.array([0.3,  0.3,  0.4,  0.05, 0.05, 0.04]),  # sleep
    1: np.array([4.0,  3.5,  5.0,  1.5,  1.2,  1.0 ]),  # move
    2: np.array([12.0, 8.0,  6.0,  5.0,  4.0,  3.5 ]),  # scratch
}

# ══════════════════════════════════════════════════════
#  场景随机分配（10k 设备）
# ══════════════════════════════════════════════════════
def make_scenarios(num_devices: int, seed: int = 0) -> dict:
    """返回 numpy 数组字典，每个数组长度 = num_devices"""
    rng = np.random.RandomState(seed)
    r   = rng.random(num_devices)

    scratch_mean = rng.uniform(4, 20, num_devices)
    tc           = rng.uniform(0.05, 0.20, num_devices)

    sick_start = np.full(num_devices, -1, dtype=np.int32)
    sick_end   = np.full(num_devices, -1, dtype=np.int32)

    # 20% 短暂发病
    ep = r >= 0.80
    if ep.any():
        ss = rng.randint(10, 150, ep.sum())
        sick_start[ep] = ss
        sick_end[ep]   = np.minimum(ss + rng.randint(5, 40, ep.sum()), DAYS)

    # 5% 慢性（从 day 0~30 开始持续到底）
    ch = r >= 0.95
    if ch.any():
        sick_start[ch] = rng.randint(0, 30, ch.sum())
        sick_end[ch]   = DAYS

    return {
        "scratch_mean": scratch_mean.astype(np.float32),
        "tc":           tc.astype(np.float32),
        "sick_start":   sick_start,
        "sick_end":     sick_end,
    }


# ══════════════════════════════════════════════════════
#  全向量化 IMU 单日生成（无 per-sample Python 循环）
# ══════════════════════════════════════════════════════
def gen_imu_day_vec(day_ts: int, n_scratch: int, sick_intensity: float,
                    rng: np.random.RandomState) -> tuple:
    """返回 (ts int64(N,), imu float64(N,6))，N = 86400 × IMU_HZ"""
    total  = 86400 * IMU_HZ
    step   = 1000 // IMU_HZ
    ts_arr = np.arange(total, dtype=np.int64) * step + day_ts

    # 睡眠区间：0:00-7:00 和 20:00-24:00
    sleep_end   = 7  * 3600 * IMU_HZ
    active_end  = 20 * 3600 * IMU_HZ

    sleep_mask   = np.zeros(total, dtype=bool)
    sleep_mask[:sleep_end]  = True
    sleep_mask[active_end:] = True

    # 活跃时段：50% 运动 / 50% 静止
    day_rand   = rng.random(total - sleep_end - (total - active_end))
    move_mask  = np.zeros(total, dtype=bool)
    move_mask[sleep_end:active_end] = day_rand < 0.50
    sleep_mask[sleep_end:active_end] = ~move_mask[sleep_end:active_end]

    # 抓挠窗口（随机落在活跃时段）
    scratch_mask = np.zeros(total, dtype=bool)
    if n_scratch > 0:
        active_range = active_end - sleep_end
        centers  = rng.randint(0, active_range, n_scratch) + sleep_end
        durations = rng.randint(50, 400, n_scratch)  # 1-8 秒 @ 50Hz
        for c, dur in zip(centers.tolist(), durations.tolist()):
            s = max(0,     c - dur // 2)
            e = min(total, c + dur // 2)
            scratch_mask[s:e] = True

    # scratch 优先级最高
    sleep_mask[scratch_mask] = False
    move_mask[scratch_mask]  = False

    imu = np.empty((total, 6), dtype=np.float64)

    for mask, btype, scale in (
        (sleep_mask,   0, 1.0),
        (move_mask,    1, 1.0),
        (scratch_mask, 2, sick_intensity),
    ):
        n = int(mask.sum())
        if n == 0:
            continue
        stds = _STDS[btype] * scale
        imu[mask] = np.clip(rng.normal(_MEANS, stds, (n, 6)), -_CLIPS, _CLIPS)

    return ts_arr, np.round(imu, 2)


# ══════════════════════════════════════════════════════
#  SQL 片段构建（返回字符串，供多表 INSERT 拼接）
# ══════════════════════════════════════════════════════
def _imu_part(dev_id: int, ts_chunk: np.ndarray, imu_chunk: np.ndarray) -> str:
    ts_l = ts_chunk.tolist()
    c    = [imu_chunk[:, i].tolist() for i in range(6)]
    n    = len(ts_l)
    vals = " ".join(
        f"({ts_l[j]},{c[0][j]},{c[1][j]},{c[2][j]},{c[3][j]},{c[4][j]},{c[5][j]})"
        for j in range(n)
    )
    return (f"d{dev_id}_imu USING {TD_DB}.imu_raw "
            f"TAGS ({dev_id}) VALUES {vals}")


def _env_part(dev_id: int, day_ts: int, temp_base: float,
              rng: np.random.RandomState) -> str:
    n    = 86400 // ENV_SAMPLE_INTERVAL
    secs = np.arange(n) * ENV_SAMPLE_INTERVAL
    ts_l = (day_ts + secs * 1000).astype(np.int64).tolist()
    t_l  = np.round(temp_base + rng.normal(0, 1.5, n), 1).tolist()
    h_l  = np.round(65 + rng.normal(0, 3.0, n), 1).tolist()
    vals = " ".join(f"({ts_l[j]},{t_l[j]},{h_l[j]})" for j in range(n))
    return (f"d{dev_id}_env USING {TD_DB}.env_raw "
            f"TAGS ({dev_id}) VALUES {vals}")


def _neck_part(dev_id: int, day_ts: int, si: float,
               rng: np.random.RandomState) -> str:
    n    = 86400 // NECK_SAMPLE_INTERVAL
    secs = np.arange(n) * NECK_SAMPLE_INTERVAL
    ts_l = (day_ts + secs * 1000).astype(np.int64).tolist()
    if si > 1.3:
        neck = np.round(38.5 + rng.uniform(0.0, 0.8, n) * (si - 0.3), 2).tolist()
    else:
        neck = np.round(37.5 + rng.uniform(-0.3, 0.3, n), 2).tolist()
    vals = " ".join(f"({ts_l[j]},{neck[j]})" for j in range(n))
    return (f"d{dev_id}_neck USING {TD_DB}.neck_temp_raw "
            f"TAGS ({dev_id}) VALUES {vals}")


def _pack_sqls(parts: list, parts_per_sql: int) -> list:
    """将 parts 按 parts_per_sql 打包成多表 INSERT SQL 列表"""
    sqls = []
    for i in range(0, len(parts), parts_per_sql):
        sqls.append("INSERT INTO " + " ".join(parts[i: i + parts_per_sql]))
    return sqls


# ══════════════════════════════════════════════════════
#  工作线程：生成一组设备 × 一天的全部 SQL
# ══════════════════════════════════════════════════════
def gen_group_day(device_ids: list, day_idx: int,
                  scenarios: dict, seed: int) -> list:
    """在线程池中运行。返回此组设备此天所有 INSERT SQL 字符串列表。"""
    rng  = np.random.RandomState(seed)
    temp = float(_temperature[day_idx])
    d    = START_DATE + timedelta(days=day_idx)
    day_ts = int(datetime(d.year, d.month, d.day,
                          tzinfo=timezone.utc).timestamp() * 1000)

    imu_parts  = []
    env_parts  = []
    neck_parts = []

    for dev_id in device_ids:
        i  = dev_id - 1
        ss = int(scenarios["sick_start"][i])
        se = int(scenarios["sick_end"][i])
        si = 1.8 if (ss >= 0 and ss <= day_idx < se) else 1.0

        n_sc = max(0, int(rng.normal(
            float(scenarios["scratch_mean"][i]) + float(scenarios["tc"][i]) * (temp - 20), 2
        )))

        ts_arr, imu_arr = gen_imu_day_vec(day_ts, n_sc, si, rng)

        # IMU → 分块
        n = len(ts_arr)
        for ci in range(0, n, IMU_CHUNK):
            imu_parts.append(
                _imu_part(dev_id, ts_arr[ci: ci + IMU_CHUNK], imu_arr[ci: ci + IMU_CHUNK])
            )

        env_parts.append(_env_part(dev_id, day_ts, temp, rng))
        neck_parts.append(_neck_part(dev_id, day_ts, si, rng))

    sqls = _pack_sqls(imu_parts, SQL_PARTS)
    sqls += _pack_sqls(env_parts, SQL_PARTS)
    sqls += _pack_sqls(neck_parts, SQL_PARTS)
    return sqls


# ══════════════════════════════════════════════════════
#  TDengine 同步（初始化用）
# ══════════════════════════════════════════════════════
def td_exec_sync(sql: str):
    import requests as _req
    resp = _req.post(TD_REST_URL, data=sql.encode(),
                     auth=(TD_USER, TD_PASS), timeout=30)
    resp.raise_for_status()
    r = resp.json()
    if r.get("code", 0) != 0:
        raise RuntimeError(f"TDengine: {r.get('desc', r)}")
    return r


def init_db():
    td_exec_sync(
        f"CREATE DATABASE IF NOT EXISTS {TD_DB} KEEP 3650 DURATION 10 COMP 2"
    )
    td_exec_sync(f"""
        CREATE STABLE IF NOT EXISTS {TD_DB}.imu_raw (
            ts TIMESTAMP, ax FLOAT, ay FLOAT, az FLOAT,
            gx FLOAT, gy FLOAT, gz FLOAT
        ) TAGS (device_id BIGINT)
    """)
    td_exec_sync(f"""
        CREATE STABLE IF NOT EXISTS {TD_DB}.env_raw (
            ts TIMESTAMP, env_temp FLOAT, env_humi FLOAT
        ) TAGS (device_id BIGINT)
    """)
    td_exec_sync(f"""
        CREATE STABLE IF NOT EXISTS {TD_DB}.neck_temp_raw (
            ts TIMESTAMP, neck_temp FLOAT
        ) TAGS (device_id BIGINT)
    """)
    print(f"[OK] 超级表已就绪（子表将在首次写入时自动创建）")


# ══════════════════════════════════════════════════════
#  异步写入 + 进度统计
# ══════════════════════════════════════════════════════
class _Stats:
    def __init__(self):
        self._lock   = threading.Lock()
        self.rows    = 0
        self.sqls    = 0
        self.errors  = 0

    def add(self, rows: int, sqls: int, err: int = 0):
        with self._lock:
            self.rows   += rows
            self.sqls   += sqls
            self.errors += err


async def _send_sql(session: aiohttp.ClientSession,
                    sem: asyncio.Semaphore,
                    sql: str,
                    stats: _Stats,
                    est_rows: int):
    async with sem:
        try:
            async with session.post(TD_REST_URL, data=sql.encode(),
                                    auth=TD_AUTH) as resp:
                r = await resp.json()
                if r.get("code", 0) != 0:
                    stats.add(0, 1, 1)
                    print(f"\n[ERR] {r.get('desc','?')[:120]}", flush=True)
                else:
                    stats.add(est_rows, 1)
        except Exception as exc:
            stats.add(0, 1, 1)
            print(f"\n[ERR] {exc}", flush=True)


# ══════════════════════════════════════════════════════
#  主异步流程
# ══════════════════════════════════════════════════════
async def amain():
    print("=" * 62)
    print(f"  设备数        : {NUM_DEVICES:,}")
    print(f"  天数          : {DAYS}")
    print(f"  生成线程      : {GEN_WORKERS}")
    print(f"  并发 HTTP     : {INSERT_CONC}")
    print(f"  设备组大小    : {GROUP_SIZE}")
    print(f"  IMU 块/设备   : {IMU_CHUNK} 行  ≈ {IMU_CHUNK*52//1024} KB")
    print(f"  块/SQL        : {SQL_PARTS}  ≈ {SQL_PARTS*IMU_CHUNK*52//1024} KB/SQL")
    imu_total_est = NUM_DEVICES * 86400 * IMU_HZ * DAYS
    print(f"  预估 IMU 总量 : {imu_total_est:,} 行")
    print("=" * 62)

    print("\n[1] 初始化超级表...")
    init_db()

    print(f"\n[2] 生成并写入数据（{NUM_DEVICES:,} 设备 × {DAYS} 天）...")

    scenarios = make_scenarios(NUM_DEVICES, seed=0)
    all_ids   = list(range(1, NUM_DEVICES + 1))
    groups    = [all_ids[i: i + GROUP_SIZE] for i in range(0, NUM_DEVICES, GROUP_SIZE)]
    n_groups  = len(groups)

    # IMU 行数估算（用于进度显示）
    imu_rows_per_dev_day = 86400 * IMU_HZ
    imu_parts_per_dev    = math.ceil(imu_rows_per_dev_day / IMU_CHUNK)
    sqls_per_group_day   = (
        math.ceil(GROUP_SIZE * imu_parts_per_dev / SQL_PARTS)  # imu
        + 2 * math.ceil(GROUP_SIZE / SQL_PARTS)                # env + neck
    )
    total_sqls_est = n_groups * DAYS * sqls_per_group_day

    stats = _Stats()
    t0    = time.time()

    executor  = ThreadPoolExecutor(max_workers=GEN_WORKERS,
                                   thread_name_prefix="gen")
    gen_sem   = asyncio.Semaphore(GEN_WORKERS * 2)   # 限制并发生成任务，控内存
    ins_sem   = asyncio.Semaphore(INSERT_CONC)
    loop      = asyncio.get_event_loop()

    conn = aiohttp.TCPConnector(
        limit          = INSERT_CONC + 10,
        keepalive_timeout = 30,
        enable_cleanup_closed = True,
    )
    async with aiohttp.ClientSession(connector=conn) as session:

        for day in range(DAYS):
            day_sqls  = 0
            t_day     = time.time()

            # --- 生成 + 发送（全天所有组并发） ---
            async def process_group(grp_ids, grp_idx):
                nonlocal day_sqls
                seed = day * n_groups + grp_idx
                async with gen_sem:
                    sqls = await loop.run_in_executor(
                        executor, gen_group_day, grp_ids, day, scenarios, seed
                    )
                # 并发发送此组的所有 SQL
                send_tasks = [
                    _send_sql(session, ins_sem, sql, stats,
                              GROUP_SIZE * IMU_CHUNK)
                    for sql in sqls
                ]
                await asyncio.gather(*send_tasks)
                day_sqls += len(sqls)

            await asyncio.gather(*[
                process_group(grp, gi) for gi, grp in enumerate(groups)
            ])

            # --- 进度输出 ---
            elapsed   = time.time() - t0
            day_secs  = time.time() - t_day
            days_done = day + 1
            eta_s     = elapsed / days_done * (DAYS - days_done)
            pct       = stats.sqls / max(total_sqls_est, 1) * 100
            spd_rows  = stats.rows / elapsed if elapsed > 0 else 0
            print(
                f"  day {days_done:>3}/{DAYS}"
                f"  sqls={day_sqls:>6}"
                f"  {day_secs:>5.1f}s/day"
                f"  {spd_rows/1e6:>5.1f}M行/s"
                f"  ETA {eta_s/60:>5.1f}min"
                f"  err={stats.errors}",
                flush=True,
            )

    executor.shutdown(wait=False)
    elapsed = time.time() - t0
    print(f"\n  总耗时    : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  总 SQL 数 : {stats.sqls:,}")
    print(f"  总行估算  : {stats.rows:,}")
    print(f"  错误数    : {stats.errors}")
    if stats.errors:
        print("  [警告] 有写入失败，建议检查后重跑")


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
