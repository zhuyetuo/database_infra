"""
双设备实时数据模拟器（15分钟批次管道）
========================================
模拟真实设备的数据上报流程：

  [设备端] 每 15 分钟采集一批 IMU 原始数据
       ↓  批量上传
  [TDengine] 存储原始 IMU 采样点（每条 = 一个采样时刻）
       ↓  行为识别算法
  [PostgreSQL behavior] 输出行为事件（运动/睡眠/抓挠片段）
       ↓  每天聚合（24 × 15min = 96 个窗口）
  [PostgreSQL skin_assessment / scratch_baseline] 每日皮肤健康评估

设备场景：
  sim_device_normal  — 全程正常，抓挠稳定 ~10次/天
  sim_device_sick    — 前 SICK_START_DAY 个模拟天正常，之后逐步发病

时间模式（CONFIG 区可调）：
  SAMPLE_HZ          IMU 采样率（Hz），决定每窗口多少条原始数据
  WINDOW_SEC         一个 15 分钟窗口对应多少真实秒（测试时可设短）
                     例：WINDOW_SEC=15  → 每 15 秒模拟一个 15 分钟窗口
  WINDOWS_PER_DAY    多少个窗口 = 1 个模拟天（实际是 96，测试可设少）
"""

import time
import signal
import sys
import math
import requests
import psycopg2
import numpy as np
from datetime import datetime, timezone, date, timedelta

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════
TD_HOST     = "127.0.0.1"
TD_PORT     = 6041
TD_USER     = "root"
TD_PASS     = "taosdata"
IMU_DB      = "pet_dog_imu"

PG_HOST     = "127.0.0.1"
PG_PORT     = 5432
PG_USER     = "postgres"
PG_PASSWORD = "123456"
PG_DB       = "pet_collar"

# ── 时间参数 ────────────────────────────────────────
SAMPLE_HZ       = 25           # IMU 采样率 Hz（真实设备常见值）
WINDOW_MINUTES  = 15           # 每批窗口时长（分钟）
WINDOW_SEC      = 15           # 每个窗口对应多少真实秒（测试加速用）
                               # 生产环境设为 WINDOW_MINUTES * 60 = 900
WINDOWS_PER_DAY = 12           # 多少窗口 = 1个模拟天（实际96，测试用12）

# ── 场景参数 ────────────────────────────────────────
SICK_START_DAY  = 5            # sim_device_sick 在第几模拟天开始发病

DEVICES = [
    {"sn": "sim_device_normal", "sick": False},
    {"sn": "sim_device_sick",   "sick": True},
]

# ── 算法常量（与 skin_assessment_db.py 保持一致）──
WARMUP   = 3
MIN_STD  = 2.0
NORMAL_W = 0.05
ABNORM_W = 0.01

BEHAVIOR_MOVE    = 1
BEHAVIOR_SLEEP   = 2
BEHAVIOR_SCRATCH = 3

# ══════════════════════════════════════════════════════
#  优雅停止
# ══════════════════════════════════════════════════════
_running = True

def _handle_sigint(sig, frame):
    global _running
    print("\n\n[停止] Ctrl+C，正在退出...")
    _running = False

signal.signal(signal.SIGINT, _handle_sigint)


# ══════════════════════════════════════════════════════
#  TDengine REST
# ══════════════════════════════════════════════════════

def td_exec(sql: str) -> dict:
    url  = f"http://{TD_HOST}:{TD_PORT}/rest/sql"
    resp = requests.post(url, data=sql.encode("utf-8"),
                         auth=(TD_USER, TD_PASS), timeout=60)
    resp.raise_for_status()
    result = resp.json()
    if result.get("code", 0) != 0:
        raise RuntimeError(f"TDengine error [{result.get('code')}]: {result.get('desc')}")
    return result


def td_init():
    """创建数据库、超级表、子表（idempotent）"""
    td_exec(f"CREATE DATABASE IF NOT EXISTS {IMU_DB} KEEP 3650 DURATION 10 COMP 2")
    # 超级表：每行 = 一个 IMU 采样点（单次采样的 6 轴数值）
    td_exec(f"""
        CREATE STABLE IF NOT EXISTS {IMU_DB}.imu_raw (
            ts  TIMESTAMP,
            ax  FLOAT,
            ay  FLOAT,
            az  FLOAT,
            gx  FLOAT,
            gy  FLOAT,
            gz  FLOAT
        ) TAGS (device_sn BINARY(64))
    """)
    for dev in DEVICES:
        sn = dev["sn"]
        td_exec(
            f"CREATE TABLE IF NOT EXISTS {IMU_DB}.{sn} "
            f"USING {IMU_DB}.imu_raw TAGS ('{sn}')"
        )
    print("[TDengine] 数据库 & 子表就绪")


def td_insert_batch(sn: str, samples: list):
    """
    批量插入原始 IMU 采样点。
    samples: list of (ts_ms, ax, ay, az, gx, gy, gz)
    每批最多 1000 条发一次请求避免 SQL 过长。
    """
    CHUNK = 1000
    total = 0
    for i in range(0, len(samples), CHUNK):
        chunk = samples[i: i + CHUNK]
        vals  = " ".join(
            f"({s[0]},{s[1]},{s[2]},{s[3]},{s[4]},{s[5]},{s[6]})"
            for s in chunk
        )
        td_exec(f"INSERT INTO {IMU_DB}.{sn} VALUES {vals}")
        total += len(chunk)
    return total


# ══════════════════════════════════════════════════════
#  PostgreSQL
# ══════════════════════════════════════════════════════

def pg_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER,
        password=PG_PASSWORD, dbname=PG_DB
    )


def pg_init():
    conn = pg_conn()
    cur  = conn.cursor()

    for s in ["pet_dog_environment", "pet_dog_behavior",
              "pet_dog_skin_assessment", "pet_dog_scratch_baseline"]:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")

    for dev in DEVICES:
        sn = dev["sn"]

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS pet_dog_environment.{sn} (
                id           BIGSERIAL    PRIMARY KEY,
                ts           BIGINT       NOT NULL UNIQUE,
                neck_temp    NUMERIC(5,2),
                env_temp     NUMERIC(5,1) NOT NULL,
                env_humidity NUMERIC(5,1) NOT NULL
            )
        """)

        # behavior 表：每行 = 算法输出的一个行为片段
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS pet_dog_behavior.{sn} (
                id           BIGSERIAL     PRIMARY KEY,
                ts_start     BIGINT        NOT NULL,
                ts_end       BIGINT        NOT NULL,
                behavior     SMALLINT      NOT NULL,
                duration_sec NUMERIC(10,2) NOT NULL,
                confidence   NUMERIC(5,3)  NOT NULL,
                -- 片段内 IMU 均值（行为识别时已计算好）
                ax_mean      NUMERIC(8,2),
                ay_mean      NUMERIC(8,2),
                az_mean      NUMERIC(8,2),
                gx_mean      NUMERIC(8,2),
                gy_mean      NUMERIC(8,2),
                gz_mean      NUMERIC(8,2)
            )
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {sn}_beh_ts
            ON pet_dog_behavior.{sn} (ts_start)
        """)

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS pet_dog_skin_assessment.{sn} (
                stat_date        DATE         PRIMARY KEY,
                scratch_count    INT          NOT NULL DEFAULT 0,
                baseline_mean    NUMERIC(6,2),
                baseline_std     NUMERIC(6,2),
                zscore           NUMERIC(6,2),
                avg_zscore       NUMERIC(6,2),
                consec_abnormal  INT          NOT NULL DEFAULT 0,
                eval_phase       SMALLINT     NOT NULL DEFAULT 0,
                threshold_z      NUMERIC(4,2),
                threshold_consec SMALLINT,
                is_abnormal      SMALLINT     NOT NULL DEFAULT 0,
                alert_triggered  SMALLINT     NOT NULL DEFAULT 0,
                alert_reason     VARCHAR(256),
                data_quality     SMALLINT     NOT NULL DEFAULT 0,
                wear_minutes     INT          NOT NULL DEFAULT 0
            )
        """)

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS pet_dog_scratch_baseline.{sn} (
                stat_date     DATE         PRIMARY KEY,
                baseline_mean NUMERIC(6,2) NOT NULL,
                baseline_std  NUMERIC(6,2) NOT NULL,
                temp_coef     NUMERIC(5,3) NOT NULL DEFAULT 0,
                confidence    NUMERIC(4,2) NOT NULL DEFAULT 0,
                valid_days    INT          NOT NULL DEFAULT 0
            )
        """)

    conn.commit()
    cur.close()
    conn.close()
    print("[PostgreSQL] 所有表就绪")


# ══════════════════════════════════════════════════════
#  IMU 原始数据生成（模拟设备采样）
# ══════════════════════════════════════════════════════

def _imu_sample(behavior: int, sick_intensity: float = 1.0) -> tuple:
    """生成一个采样点的 6 轴数值"""
    si = sick_intensity
    if behavior == BEHAVIOR_SLEEP:
        return (
            round(float(np.random.normal(0,   20)),  2),
            round(float(np.random.normal(0,   15)),  2),
            round(float(np.random.normal(980, 25)),  2),
            round(float(np.random.normal(0,   2)),   2),
            round(float(np.random.normal(0,   2)),   2),
            round(float(np.random.normal(0,   1.5)), 2),
        )
    elif behavior == BEHAVIOR_MOVE:
        return (
            round(float(np.random.normal(40,  150)), 2),
            round(float(np.random.normal(20,  120)), 2),
            round(float(np.random.normal(650, 280)), 2),
            round(float(np.random.normal(0,   90)),  2),
            round(float(np.random.normal(0,   70)),  2),
            round(float(np.random.normal(0,   55)),  2),
        )
    else:  # SCRATCH
        return (
            round(float(np.random.normal(180 * si, 80)),  2),
            round(float(np.random.normal(40,        60)),  2),
            round(float(np.random.normal(800,       180)), 2),
            round(float(np.random.normal(0, 130 * si)),    2),
            round(float(np.random.normal(0,  90)),          2),
            round(float(np.random.normal(0,  70)),          2),
        )


def generate_window(window_start_ms: int, sick_intensity: float) -> tuple:
    """
    模拟一个 15 分钟窗口内的 IMU 采集过程。

    返回：
      raw_samples  — list of (ts_ms, ax, ay, az, gx, gy, gz)
                     直接写入 TDengine
      segments     — list of dict {ts_start, ts_end, behavior, samples}
                     由"行为识别算法"切分出来，写入 PostgreSQL behavior 表
    """
    window_ms   = WINDOW_MINUTES * 60 * 1000
    step_ms     = int(1000 / SAMPLE_HZ)   # 采样间隔（ms）
    total_steps = WINDOW_MINUTES * 60 * SAMPLE_HZ  # 共多少个采样点

    raw_samples = []
    segments    = []

    cursor_ms   = window_start_ms
    step_idx    = 0

    while step_idx < total_steps:
        # 随机选一个行为片段（设备运动学上的连续行为）
        # 根据时间段决定行为分布
        elapsed_frac = (cursor_ms - window_start_ms) / window_ms
        hour_of_day  = (datetime.fromtimestamp(cursor_ms / 1000, tz=timezone.utc).hour)

        # 行为概率权重 [sleep, move, scratch]
        if   0 <= hour_of_day < 6:   w = [0.80, 0.15, 0.05]
        elif 6 <= hour_of_day < 8:   w = [0.30, 0.55, 0.15]
        elif 8 <= hour_of_day < 12:  w = [0.10, 0.70, 0.20]
        elif 12 <= hour_of_day < 14: w = [0.65, 0.25, 0.10]
        elif 14 <= hour_of_day < 20: w = [0.10, 0.70, 0.20]
        elif 20 <= hour_of_day < 22: w = [0.45, 0.45, 0.10]
        else:                         w = [0.75, 0.20, 0.05]

        # 发病后提高抓挠权重
        scratch_boost = (sick_intensity - 1.0) * 0.35
        w[2] = min(w[2] + scratch_boost, 0.65)
        total_w = sum(w); w = [x / total_w for x in w]

        btype = np.random.choice([BEHAVIOR_SLEEP, BEHAVIOR_MOVE, BEHAVIOR_SCRATCH],
                                 p=w)

        # 片段持续采样点数
        if btype == BEHAVIOR_SLEEP:
            seg_steps = np.random.randint(SAMPLE_HZ * 30,  SAMPLE_HZ * 600)  # 30s~10min
        elif btype == BEHAVIOR_MOVE:
            seg_steps = np.random.randint(SAMPLE_HZ * 5,   SAMPLE_HZ * 120)  # 5s~2min
        else:  # SCRATCH
            seg_steps = np.random.randint(SAMPLE_HZ * 1,   SAMPLE_HZ * 8)    # 1s~8s

        seg_steps = min(seg_steps, total_steps - step_idx)
        if seg_steps <= 0:
            break

        seg_start_ms = cursor_ms
        seg_samples  = []

        for _ in range(seg_steps):
            feat = _imu_sample(btype, sick_intensity)
            raw_samples.append((cursor_ms,) + feat)
            seg_samples.append(feat)
            cursor_ms += step_ms

        step_idx += seg_steps

        seg_end_ms   = cursor_ms
        dur_sec      = round((seg_end_ms - seg_start_ms) / 1000.0, 2)

        # 置信度：抓挠识别稍低，睡眠最高
        if btype == BEHAVIOR_SLEEP:
            conf = round(np.random.uniform(0.88, 0.99), 3)
        elif btype == BEHAVIOR_MOVE:
            conf = round(np.random.uniform(0.82, 0.97), 3)
        else:
            conf = round(np.random.uniform(0.72, 0.93), 3)

        # 片段内 IMU 均值（行为识别模型的输入特征）
        arr = np.array(seg_samples)
        segments.append({
            "ts_start":    seg_start_ms,
            "ts_end":      seg_end_ms,
            "behavior":    btype,
            "duration_sec": dur_sec,
            "confidence":  conf,
            "ax_mean": round(float(arr[:, 0].mean()), 2),
            "ay_mean": round(float(arr[:, 1].mean()), 2),
            "az_mean": round(float(arr[:, 2].mean()), 2),
            "gx_mean": round(float(arr[:, 3].mean()), 2),
            "gy_mean": round(float(arr[:, 4].mean()), 2),
            "gz_mean": round(float(arr[:, 5].mean()), 2),
        })

    return raw_samples, segments


# ══════════════════════════════════════════════════════
#  写入行为事件
# ══════════════════════════════════════════════════════

def pg_insert_behavior(conn, sn: str, segments: list) -> int:
    """将行为识别结果写入 PostgreSQL behavior 表，返回抓挠次数"""
    if not segments:
        return 0
    cur = conn.cursor()
    sql = f"""
        INSERT INTO pet_dog_behavior.{sn}
            (ts_start, ts_end, behavior, duration_sec, confidence,
             ax_mean, ay_mean, az_mean, gx_mean, gy_mean, gz_mean)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT DO NOTHING
    """
    rows = [
        (s["ts_start"], s["ts_end"], s["behavior"], s["duration_sec"], s["confidence"],
         s["ax_mean"], s["ay_mean"], s["az_mean"],
         s["gx_mean"], s["gy_mean"], s["gz_mean"])
        for s in segments
    ]
    cur.executemany(sql, rows)
    conn.commit()
    cur.close()
    scratch_cnt = sum(1 for s in segments if s["behavior"] == BEHAVIOR_SCRATCH)
    return scratch_cnt


# ══════════════════════════════════════════════════════
#  环境 & 皮肤评估（每日结算）
# ══════════════════════════════════════════════════════

def _get_thresholds(vd: int):
    if vd < 1:      return None, None, None
    elif vd <= 11:  return 4.0, 5, 5.0
    elif vd <= 27:  return 3.5, 4, 4.5
    else:           return 2.5, 3, 3.5

def _get_phase(vd: int) -> int:
    if vd == 0:     return 0
    elif vd <= 11:  return 1
    elif vd <= 27:  return 2
    else:           return 3

def _temp_coef(bc: list, bt: list) -> float:
    if len(bc) < 20:
        return 0.0
    x = np.array(bt, float); y = np.array(bc, float)
    c = np.sum((x - x.mean()) * (y - y.mean())) / (np.sum((x - x.mean()) ** 2) + 1e-8)
    return round(float(np.clip(c, 0.0, 0.4)), 3)


def settle_day(conn, state: "DeviceState"):
    """一天所有窗口跑完后，写环境数据 + 皮肤评估 + 基线"""
    stat_date = (date.today() + timedelta(days=state.day_idx)).isoformat()
    temp      = state.env_temp()
    humi      = state.env_humidity()
    count     = state.day_scratch_count
    si        = state.sick_intensity()
    wear_min  = int(np.random.uniform(1380, 1440))
    neck_temp = (round(38.5 + np.random.uniform(0.0, 0.8) * (si - 0.3), 2)
                 if si > 1.3 else
                 round(37.5 + np.random.uniform(-0.3, 0.3), 2))

    cur = conn.cursor()

    # 环境
    ts_env = int(datetime.now(timezone.utc).timestamp() * 1000)
    cur.execute(f"""
        INSERT INTO pet_dog_environment.{state.sn}
            (ts, neck_temp, env_temp, env_humidity)
        VALUES (%s,%s,%s,%s)
        ON CONFLICT (ts) DO UPDATE
          SET neck_temp=EXCLUDED.neck_temp,
              env_temp=EXCLUDED.env_temp,
              env_humidity=EXCLUDED.env_humidity
    """, (ts_env, neck_temp, temp, humi))

    state.buf_t.append(temp)

    # 热身期
    if state.day_idx < WARMUP:
        state.buf_c.append(count)
        cur.execute(f"""
            INSERT INTO pet_dog_skin_assessment.{state.sn}
                (stat_date, scratch_count, eval_phase, data_quality, wear_minutes)
            VALUES (%s,%s,0,0,%s)
            ON CONFLICT (stat_date) DO UPDATE
              SET scratch_count=EXCLUDED.scratch_count,
                  wear_minutes=EXCLUDED.wear_minutes
        """, (stat_date, count, wear_min))
        conn.commit()
        cur.close()
        print(f"  [{state.sn}] 模拟天{state.day_idx:3d} [热身期] 抓挠={count:3d}次")
        state.day_scratch_count = 0
        state.day_idx += 1
        return

    # 初始化基线
    if state.mean is None:
        state.mean = float(np.mean(state.buf_c)) if state.buf_c else float(count)
        state.std  = max(float(np.std(state.buf_c)) if len(state.buf_c) > 1 else MIN_STD, MIN_STD)

    state.valid_days += 1
    tz, tc, ta = _get_thresholds(state.valid_days)
    coef       = _temp_coef(state.buf_c, state.buf_t)
    zscore     = round(((count - state.mean) - coef * (temp - 20)) / state.std, 2)
    is_abn     = bool(tz is not None and zscore > tz)

    if is_abn:
        state.consec += 1
        state.mean = state.mean * (1 - ABNORM_W) + count * ABNORM_W
    else:
        state.consec = 0
        state.mean   = state.mean * (1 - NORMAL_W) + count * NORMAL_W
        state.buf_c.append(count)
    if len(state.buf_c) > 1:
        state.std = max(float(np.std(state.buf_c[-30:])), MIN_STD)

    nb    = max((tc - 1) if tc else 2, 1)
    avg_z = round(float(np.mean(state.recent_z[-nb:] + [zscore])), 2)
    state.recent_z.append(zscore)
    if len(state.recent_z) > 10:
        state.recent_z.pop(0)

    alert  = bool(tc and state.consec >= tc and avg_z >= ta)
    reason = (f"连续{state.consec}天z>{tz:.1f}，均值z={avg_z:.2f}，抓挠{count}次"
              if alert else None)

    cur.execute(f"""
        INSERT INTO pet_dog_skin_assessment.{state.sn}
            (stat_date, scratch_count,
             baseline_mean, baseline_std, zscore, avg_zscore,
             consec_abnormal, eval_phase,
             threshold_z, threshold_consec,
             is_abnormal, alert_triggered, alert_reason,
             data_quality, wear_minutes)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,%s)
        ON CONFLICT (stat_date) DO UPDATE
          SET scratch_count    = EXCLUDED.scratch_count,
              baseline_mean    = EXCLUDED.baseline_mean,
              baseline_std     = EXCLUDED.baseline_std,
              zscore           = EXCLUDED.zscore,
              avg_zscore       = EXCLUDED.avg_zscore,
              consec_abnormal  = EXCLUDED.consec_abnormal,
              is_abnormal      = EXCLUDED.is_abnormal,
              alert_triggered  = EXCLUDED.alert_triggered,
              alert_reason     = EXCLUDED.alert_reason,
              wear_minutes     = EXCLUDED.wear_minutes
    """, (
        stat_date, count,
        round(state.mean, 2), round(state.std, 2), zscore, avg_z,
        state.consec, _get_phase(state.valid_days),
        tz, tc,
        int(is_abn), int(alert), reason, wear_min,
    ))

    confidence = round(min(1.0, state.valid_days / 30), 2)
    cur.execute(f"""
        INSERT INTO pet_dog_scratch_baseline.{state.sn}
            (stat_date, baseline_mean, baseline_std, temp_coef, confidence, valid_days)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (stat_date) DO UPDATE
          SET baseline_mean=EXCLUDED.baseline_mean,
              baseline_std=EXCLUDED.baseline_std,
              temp_coef=EXCLUDED.temp_coef,
              confidence=EXCLUDED.confidence,
              valid_days=EXCLUDED.valid_days
    """, (stat_date, round(state.mean, 2), round(state.std, 2),
          coef, confidence, state.valid_days))

    conn.commit()
    cur.close()

    alert_tag = "  🚨 报警!" if alert else ""
    abn_tag   = " ⚠️ 异常" if is_abn else ""
    print(f"  [{state.sn}] 模拟天{state.day_idx:3d}  抓挠={count:3d}次  "
          f"z={zscore:+.2f}  基线={state.mean:.1f}±{state.std:.1f}  "
          f"phase={_get_phase(state.valid_days)}{abn_tag}{alert_tag}")

    state.day_scratch_count = 0
    state.day_idx += 1


# ══════════════════════════════════════════════════════
#  设备状态
# ══════════════════════════════════════════════════════

class DeviceState:
    def __init__(self, sn: str, is_sick: bool):
        self.sn            = sn
        self.is_sick       = is_sick
        self.day_idx       = 0
        self.window_in_day = 0      # 当天已完成的窗口数

        # 基线状态
        self.mean       = None
        self.std        = MIN_STD
        self.buf_c      = []
        self.buf_t      = []
        self.consec     = 0
        self.valid_days = 0
        self.recent_z   = []

        # 当天累计抓挠次数（跨窗口累加）
        self.day_scratch_count = 0

    def sick_intensity(self) -> float:
        if not self.is_sick or self.day_idx < SICK_START_DAY:
            return 1.0
        progress = min((self.day_idx - SICK_START_DAY) / 20.0, 1.0)
        return 1.0 + 1.2 * progress

    def env_temp(self) -> float:
        doy  = (date.today() + timedelta(days=self.day_idx)).timetuple().tm_yday
        base = 22 + 13 * math.sin((doy - 80) / 365 * 2 * math.pi)
        return round(base + np.random.normal(0, 1.5), 1)

    def env_humidity(self) -> float:
        doy  = (date.today() + timedelta(days=self.day_idx)).timetuple().tm_yday
        base = 65 + 15 * math.sin((doy - 80) / 365 * 2 * math.pi)
        return round(base + np.random.normal(0, 3.0), 1)


# ══════════════════════════════════════════════════════
#  主循环
# ══════════════════════════════════════════════════════

def process_window(state: DeviceState, conn, window_ts_ms: int):
    """处理一个 15 分钟窗口：采集 → TDengine，识别 → PostgreSQL behavior"""
    si = state.sick_intensity()

    # 1. 生成原始 IMU 数据（模拟设备端采集）
    raw_samples, segments = generate_window(window_ts_ms, si)

    # 2. 写入 TDengine（原始采样点）
    n_raw = 0
    try:
        n_raw = td_insert_batch(state.sn, raw_samples)
    except Exception as e:
        print(f"    ⚠ TDengine 写入失败: {e}")

    # 3. 写入 PostgreSQL（行为识别结果）
    scratch_in_window = 0
    try:
        scratch_in_window = pg_insert_behavior(conn, state.sn, segments)
        state.day_scratch_count += scratch_in_window
    except Exception as e:
        conn.rollback()
        print(f"    ⚠ 行为写入失败: {e}")

    n_move    = sum(1 for s in segments if s["behavior"] == BEHAVIOR_MOVE)
    n_sleep   = sum(1 for s in segments if s["behavior"] == BEHAVIOR_SLEEP)
    n_scratch = scratch_in_window

    print(f"  [{state.sn}] 窗口 day{state.day_idx}-w{state.window_in_day+1:02d}  "
          f"原始点={n_raw:5d}  片段={len(segments):3d}"
          f"（运动{n_move} 睡眠{n_sleep} 抓挠{n_scratch}）  si={si:.2f}")

    state.window_in_day += 1

    # 4. 攒够一天的窗口 → 结算每日评估
    if state.window_in_day >= WINDOWS_PER_DAY:
        state.window_in_day = 0
        settle_day(conn, state)


def main():
    real_window = WINDOW_MINUTES * 60
    scale       = real_window / WINDOW_SEC
    print("=" * 65)
    print("  宠物项圈双设备模拟器  （15分钟批次管道）")
    print(f"  sim_device_normal — 全程正常")
    print(f"  sim_device_sick   — 第{SICK_START_DAY}模拟天起逐步发病")
    print(f"  IMU 采样率:    {SAMPLE_HZ} Hz")
    print(f"  窗口时长:      {WINDOW_MINUTES} 分钟 / {SAMPLE_HZ*WINDOW_MINUTES*60:,} 采样点/窗口/设备")
    print(f"  每天窗口数:    {WINDOWS_PER_DAY} 个")
    print(f"  加速比:        1 真实秒 ≈ {scale:.0f} 模拟秒  "
          f"（{WINDOW_SEC}s 真实 = {WINDOW_MINUTES}min 模拟）")
    print(f"  Ctrl+C 停止")
    print("=" * 65)

    print("\n[初始化] TDengine...")
    td_init()
    print("[初始化] PostgreSQL...")
    pg_init()

    states = [DeviceState(d["sn"], d["sick"]) for d in DEVICES]
    conn   = pg_conn()

    # window_ts 用来模拟时间：从当前时刻向后推
    # 每个窗口按 WINDOW_MINUTES 递增
    base_ts = _now_ms()
    window_counter = 0

    print(f"\n[运行中] 每 {WINDOW_SEC}s 处理一个 {WINDOW_MINUTES}min 数据窗口...\n")

    while _running:
        window_ts_ms = base_ts + window_counter * WINDOW_MINUTES * 60 * 1000
        window_counter += 1

        print(f"── 窗口 #{window_counter}  "
              f"模拟时间: {datetime.fromtimestamp(window_ts_ms/1000, tz=timezone.utc).strftime('%m-%d %H:%M')} ──")

        for state in states:
            process_window(state, conn, window_ts_ms)

        # 等待下一个窗口
        if _running:
            time.sleep(WINDOW_SEC)

    conn.close()
    print("\n[完成] 模拟器已停止")


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


if __name__ == "__main__":
    main()
