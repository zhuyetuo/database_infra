"""
双设备实时数据模拟器
====================
模拟两个宠物项圈设备持续上报数据，直到手动 Ctrl+C 停止。

设备：
  sim_device_normal  — 完全正常，抓挠稳定在 ~10次/模拟天
  sim_device_sick    — 前10模拟天正常，之后皮肤异常，抓挠逐步升至 ~30次/模拟天

时间节奏（可在 CONFIG 区修改）：
  IMU_INTERVAL     每隔 N 秒生成一批 IMU 原始事件 → TDengine
  BEHAV_INTERVAL   每隔 N 秒聚合一次行为事件      → PostgreSQL behavior
  DAY_INTERVAL     每隔 N 秒推进一个"模拟天"       → PostgreSQL environment / assessment / baseline

写入目标：
  TDengine  pet_dog_imu.sim_device_normal / sim_device_sick
  PG schema pet_dog_environment    → sim_device_normal / sim_device_sick
  PG schema pet_dog_behavior       → sim_device_normal / sim_device_sick
  PG schema pet_dog_skin_assessment → sim_device_normal / sim_device_sick
  PG schema pet_dog_scratch_baseline → sim_device_normal / sim_device_sick
"""

import time
import signal
import sys
import math
import requests
import psycopg2
import psycopg2.extras
import numpy as np
from datetime import datetime, timezone, date, timedelta

# ══════════════════════════════════════════════════════
#  CONFIG — 按需调整
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

IMU_INTERVAL   = 5    # 秒：每隔多久写一批 IMU 事件
BEHAV_INTERVAL = 30   # 秒：每隔多久聚合一次行为事件
DAY_INTERVAL   = 60   # 秒：每隔多久推进一个"模拟天"（含环境/评估/基线）

SICK_START_DAY = 10   # sim_device_sick 在第几模拟天开始发病

DEVICES = [
    {"sn": "sim_device_normal", "sick": False},
    {"sn": "sim_device_sick",   "sick": True},
]

# ══════════════════════════════════════════════════════
#  算法常量（与 skin_assessment_db.py 保持一致）
# ══════════════════════════════════════════════════════
WARMUP   = 3
MIN_STD  = 2.0
NORMAL_W = 0.05
ABNORM_W = 0.01
GAP_RESET = 30

BEHAVIOR_MOVE    = 1
BEHAVIOR_SLEEP   = 2
BEHAVIOR_SCRATCH = 3

# ══════════════════════════════════════════════════════
#  全局运行状态
# ══════════════════════════════════════════════════════
_running = True

def _handle_sigint(sig, frame):
    global _running
    print("\n\n[停止] 收到 Ctrl+C，正在退出...")
    _running = False

signal.signal(signal.SIGINT, _handle_sigint)


# ══════════════════════════════════════════════════════
#  TDengine REST
# ══════════════════════════════════════════════════════

def td_exec(sql: str) -> dict:
    url  = f"http://{TD_HOST}:{TD_PORT}/rest/sql"
    resp = requests.post(url, data=sql.encode("utf-8"),
                         auth=(TD_USER, TD_PASS), timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if result.get("code", 0) != 0:
        raise RuntimeError(f"TDengine error: {result.get('desc', result)}")
    return result


def td_init():
    """确保 TDengine 数据库、超级表、子表都存在"""
    td_exec(f"CREATE DATABASE IF NOT EXISTS {IMU_DB} KEEP 3650 DURATION 10 COMP 2")
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
        ) TAGS (device_sn BINARY(64))
    """)
    for dev in DEVICES:
        sn = dev["sn"]
        td_exec(
            f"CREATE TABLE IF NOT EXISTS {IMU_DB}.{sn} "
            f"USING {IMU_DB}.imu_events TAGS ('{sn}')"
        )
    print("[TDengine] 数据库 & 子表 就绪")


def td_insert_imu(sn: str, rows: list):
    """rows: list of (ts_start_ms, ts_end_ms, ax, ay, az, gx, gy, gz)"""
    if not rows:
        return
    vals = " ".join(
        f"({r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},{r[7]})"
        for r in rows
    )
    td_exec(f"INSERT INTO {IMU_DB}.{sn} VALUES {vals}")


# ══════════════════════════════════════════════════════
#  PostgreSQL
# ══════════════════════════════════════════════════════

def pg_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER,
        password=PG_PASSWORD, dbname=PG_DB
    )


def pg_init():
    """确保所有 schema 和表都存在"""
    conn = pg_conn()
    cur  = conn.cursor()

    schemas = [
        "pet_dog_environment",
        "pet_dog_behavior",
        "pet_dog_skin_assessment",
        "pet_dog_scratch_baseline",
    ]
    for s in schemas:
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

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS pet_dog_behavior.{sn} (
                id           BIGSERIAL     PRIMARY KEY,
                ts_start     BIGINT        NOT NULL,
                ts_end       BIGINT        NOT NULL,
                behavior     SMALLINT      NOT NULL,
                duration_sec NUMERIC(10,2) NOT NULL,
                confidence   NUMERIC(5,3)  NOT NULL
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
#  IMU 特征生成（与 imu_raw_db.py 一致）
# ══════════════════════════════════════════════════════

def _gen_imu_feat(behavior: int, sick_intensity: float = 1.0) -> tuple:
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
        si = sick_intensity
        return (
            round(float(np.random.normal(180 * si, 80)),  2),
            round(float(np.random.normal(40,        60)),  2),
            round(float(np.random.normal(800,       180)), 2),
            round(float(np.random.normal(0, 130 * si)),    2),
            round(float(np.random.normal(0,  90)),          2),
            round(float(np.random.normal(0,  70)),          2),
        )


# ══════════════════════════════════════════════════════
#  评估算法工具函数（与 skin_assessment_db.py 一致）
# ══════════════════════════════════════════════════════

def _get_thresholds(valid_days: int):
    if valid_days < 1:      return None, None, None
    elif valid_days <= 11:  return 4.0, 5, 5.0
    elif valid_days <= 27:  return 3.5, 4, 4.5
    else:                   return 2.5, 3, 3.5


def _get_phase(valid_days: int) -> int:
    if valid_days == 0:     return 0
    elif valid_days <= 11:  return 1
    elif valid_days <= 27:  return 2
    else:                   return 3


def _estimate_temp_coef(buf_c: list, buf_t: list) -> float:
    if len(buf_c) < 20:
        return 0.0
    x = np.array(buf_t, dtype=float)
    y = np.array(buf_c, dtype=float)
    coef = (np.sum((x - x.mean()) * (y - y.mean()))
            / (np.sum((x - x.mean()) ** 2) + 1e-8))
    return round(float(np.clip(coef, 0.0, 0.4)), 3)


# ══════════════════════════════════════════════════════
#  每个设备的运行状态
# ══════════════════════════════════════════════════════

class DeviceState:
    def __init__(self, sn: str, is_sick: bool):
        self.sn       = sn
        self.is_sick  = is_sick
        self.day_idx  = 0          # 当前模拟天索引

        # 基线状态
        self.mean       = None
        self.std        = MIN_STD
        self.buf_c      = []
        self.buf_t      = []
        self.consec     = 0
        self.valid_days = 0
        self.recent_z   = []

        # 行为计数（当前模拟天内积累的抓挠次数）
        self.day_scratch_count = 0
        self.day_start_ts      = _now_ms()

    def sick_intensity(self) -> float:
        if not self.is_sick:
            return 1.0
        if self.day_idx < SICK_START_DAY:
            return 1.0
        # 发病后逐步增强，最高 2.2
        progress = min((self.day_idx - SICK_START_DAY) / 20.0, 1.0)
        return 1.0 + 1.2 * progress

    def target_scratch_per_day(self) -> float:
        """目标每天抓挠次数（用于泊松分布采样）"""
        si = self.sick_intensity()
        base = 10.0
        # sick_intensity 1.0 → ~10次/天，2.2 → ~28次/天
        return base + (si - 1.0) * 15.0

    def env_temp(self) -> float:
        """简单正弦季节温度"""
        doy  = (date.today() + timedelta(days=self.day_idx)).timetuple().tm_yday
        base = 22 + 13 * math.sin((doy - 80) / 365 * 2 * math.pi)
        return round(base + np.random.normal(0, 1.5), 1)

    def env_humidity(self) -> float:
        doy  = (date.today() + timedelta(days=self.day_idx)).timetuple().tm_yday
        base = 65 + 15 * math.sin((doy - 80) / 365 * 2 * math.pi)
        return round(base + np.random.normal(0, 3.0), 1)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


# ══════════════════════════════════════════════════════
#  IMU 事件生成 & 写入
# ══════════════════════════════════════════════════════

def step_imu(state: DeviceState):
    """生成 3~8 个 IMU 事件，写入 TDengine"""
    now_ms  = _now_ms()
    si      = state.sick_intensity()

    # 根据时间段决定当前行为权重
    hour = datetime.now().hour
    if   6 <= hour < 8:   sleep_w, scratch_prob = 0.3, 0.15
    elif 8 <= hour < 12:  sleep_w, scratch_prob = 0.1, 0.20
    elif 12 <= hour < 14: sleep_w, scratch_prob = 0.6, 0.05
    elif 14 <= hour < 20: sleep_w, scratch_prob = 0.1, 0.20
    elif 20 <= hour < 22: sleep_w, scratch_prob = 0.4, 0.10
    else:                 sleep_w, scratch_prob = 0.8, 0.05

    # 发病后增加抓挠概率
    scratch_prob = min(scratch_prob * si, 0.6)

    n_events = np.random.randint(3, 9)
    rows     = []
    cursor   = now_ms - n_events * 8000  # 从约 N*8秒 前倒推

    for _ in range(n_events):
        r = np.random.random()
        if r < scratch_prob:
            btype = BEHAVIOR_SCRATCH
            state.day_scratch_count += 1
        elif r < scratch_prob + sleep_w:
            btype = BEHAVIOR_SLEEP
        else:
            btype = BEHAVIOR_MOVE

        dur_ms = (int(np.random.uniform(1000, 8000))   if btype == BEHAVIOR_SCRATCH else
                  int(np.random.uniform(30000, 180000)) if btype == BEHAVIOR_SLEEP   else
                  int(np.random.uniform(5000, 60000)))

        ts_start = cursor
        ts_end   = cursor + dur_ms
        feat     = _gen_imu_feat(btype, si)
        rows.append((ts_start, ts_end) + feat)
        cursor   = ts_end + np.random.randint(500, 3000)

    try:
        td_insert_imu(state.sn, rows)
        print(f"  [{state.sn}] IMU +{len(rows)}条  "
              f"(si={si:.2f}, 今日抓挠已累计{state.day_scratch_count}次)")
    except Exception as e:
        print(f"  [{state.sn}] IMU 写入失败: {e}")


# ══════════════════════════════════════════════════════
#  行为事件聚合 & 写入
# ══════════════════════════════════════════════════════

def step_behavior(state: DeviceState, conn):
    """将近期积累的行为写为一条行为事件"""
    now_ms   = _now_ms()
    dur_sec  = BEHAV_INTERVAL
    si       = state.sick_intensity()

    # 这段时间内按抓挠概率决定行为类型
    if state.day_scratch_count > 0 and np.random.random() < 0.4:
        btype    = BEHAVIOR_SCRATCH
        dur_sec  = round(np.random.uniform(2, 15), 2)
        conf     = round(np.random.uniform(0.75, 0.95), 3)
    elif np.random.random() < 0.35:
        btype    = BEHAVIOR_SLEEP
        dur_sec  = round(np.random.uniform(300, 1800), 2)
        conf     = round(np.random.uniform(0.88, 0.98), 3)
    else:
        btype    = BEHAVIOR_MOVE
        dur_sec  = round(np.random.uniform(30, 600), 2)
        conf     = round(np.random.uniform(0.82, 0.97), 3)

    ts_start = now_ms - int(dur_sec * 1000)
    ts_end   = now_ms

    try:
        cur = conn.cursor()
        cur.execute(f"""
            INSERT INTO pet_dog_behavior.{state.sn}
                (ts_start, ts_end, behavior, duration_sec, confidence)
            VALUES (%s, %s, %s, %s, %s)
        """, (ts_start, ts_end, btype, dur_sec, conf))
        conn.commit()
        cur.close()
        bname = {1:"运动",2:"睡眠",3:"抓挠"}[btype]
        print(f"  [{state.sn}] 行为事件: {bname} {dur_sec:.0f}s  conf={conf}")
    except Exception as e:
        conn.rollback()
        print(f"  [{state.sn}] 行为写入失败: {e}")


# ══════════════════════════════════════════════════════
#  模拟天推进：环境 + 皮肤评估 + 基线
# ══════════════════════════════════════════════════════

def step_day(state: DeviceState, conn):
    """推进一个模拟天，写入环境数据、皮肤评估、基线快照"""
    stat_date = (date.today() + timedelta(days=state.day_idx)).isoformat()
    temp      = state.env_temp()
    humi      = state.env_humidity()
    count     = state.day_scratch_count
    wear_min  = int(np.random.uniform(1350, 1440))
    si        = state.sick_intensity()

    # 脖颈温度：发病时偏高
    neck_temp = (round(38.5 + np.random.uniform(0.0, 0.8) * (si - 0.5), 2)
                 if si > 1.2 else
                 round(37.5 + np.random.uniform(-0.3, 0.3), 2))

    cur = conn.cursor()

    # ── 环境数据 ─────────────────────────────────────
    try:
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        cur.execute(f"""
            INSERT INTO pet_dog_environment.{state.sn}
                (ts, neck_temp, env_temp, env_humidity)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (ts) DO UPDATE
              SET neck_temp=EXCLUDED.neck_temp,
                  env_temp=EXCLUDED.env_temp,
                  env_humidity=EXCLUDED.env_humidity
        """, (ts, neck_temp, temp, humi))
    except Exception as e:
        print(f"  [{state.sn}] 环境写入失败: {e}")
        conn.rollback()
        return

    # ── 皮肤评估算法 ─────────────────────────────────
    state.buf_t.append(temp)

    if state.day_idx < WARMUP:
        # 热身期
        state.buf_c.append(count)
        cur.execute(f"""
            INSERT INTO pet_dog_skin_assessment.{state.sn}
                (stat_date, scratch_count, eval_phase, data_quality, wear_minutes)
            VALUES (%s, %s, 0, 0, %s)
            ON CONFLICT (stat_date) DO UPDATE
              SET scratch_count=EXCLUDED.scratch_count,
                  wear_minutes=EXCLUDED.wear_minutes
        """, (stat_date, count, wear_min))
    else:
        if state.mean is None:
            state.mean = float(np.mean(state.buf_c)) if state.buf_c else float(count)
            state.std  = max(float(np.std(state.buf_c)) if len(state.buf_c) > 1 else MIN_STD, MIN_STD)

        state.valid_days += 1
        tz, tc, ta = _get_thresholds(state.valid_days)
        coef       = _estimate_temp_coef(state.buf_c, state.buf_t)
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

        nb     = max((tc - 1) if tc else 2, 1)
        avg_z  = round(float(np.mean(state.recent_z[-nb:] + [zscore])), 2)
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
            int(is_abn), int(alert), reason,
            wear_min,
        ))

        # ── 基线快照 ───────────────────────────────
        confidence = round(min(1.0, state.valid_days / 30), 2)
        cur.execute(f"""
            INSERT INTO pet_dog_scratch_baseline.{state.sn}
                (stat_date, baseline_mean, baseline_std, temp_coef, confidence, valid_days)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (stat_date) DO UPDATE
              SET baseline_mean = EXCLUDED.baseline_mean,
                  baseline_std  = EXCLUDED.baseline_std,
                  temp_coef     = EXCLUDED.temp_coef,
                  confidence    = EXCLUDED.confidence,
                  valid_days    = EXCLUDED.valid_days
        """, (
            stat_date,
            round(state.mean, 2), round(state.std, 2),
            coef, confidence, state.valid_days,
        ))

        alert_tag = " 🚨 报警触发！" if alert else ""
        abn_tag   = " ⚠️  异常" if is_abn else ""
        print(f"  [{state.sn}] 模拟天{state.day_idx:3d}  抓挠={count:3d}次  "
              f"z={zscore:+.2f}  基线={state.mean:.1f}±{state.std:.1f}"
              f"  phase={_get_phase(state.valid_days)}"
              f"{abn_tag}{alert_tag}")

    conn.commit()
    cur.close()

    # 重置当天抓挠计数
    state.day_scratch_count = 0
    state.day_idx += 1


# ══════════════════════════════════════════════════════
#  主循环
# ══════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  宠物项圈双设备模拟器")
    print(f"  sim_device_normal  — 全程正常")
    print(f"  sim_device_sick    — 第{SICK_START_DAY}模拟天起逐步异常")
    print(f"  IMU 写入间隔:  {IMU_INTERVAL}s")
    print(f"  行为写入间隔:  {BEHAV_INTERVAL}s")
    print(f"  模拟天推进间隔:{DAY_INTERVAL}s")
    print("  Ctrl+C 停止")
    print("=" * 60)

    # 初始化数据库
    print("\n[初始化] TDengine...")
    td_init()
    print("[初始化] PostgreSQL...")
    pg_init()

    states = [DeviceState(d["sn"], d["sick"]) for d in DEVICES]
    conn   = pg_conn()

    last_imu   = 0.0
    last_behav = 0.0
    last_day   = 0.0

    print("\n[运行中] 开始生成数据...\n")

    while _running:
        now = time.time()

        # ── IMU 事件 ───────────────────────────────
        if now - last_imu >= IMU_INTERVAL:
            for s in states:
                step_imu(s)
            last_imu = now

        # ── 行为事件 ───────────────────────────────
        if now - last_behav >= BEHAV_INTERVAL:
            for s in states:
                step_behavior(s, conn)
            last_behav = now

        # ── 模拟天推进 ─────────────────────────────
        if now - last_day >= DAY_INTERVAL:
            print(f"\n── 推进模拟天 ──")
            for s in states:
                step_day(s, conn)
            last_day = now
            print()

        time.sleep(0.5)

    conn.close()
    print("[完成] 模拟器已停止")


if __name__ == "__main__":
    main()
