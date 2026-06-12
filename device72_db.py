"""
设备 72 专用数据生成器
====================================
设备信息:
  device_id  : 72
  device_sn  : EA:CB:3E:CF:00:11

TDengine (hiccpet_device):
  imu_ea_cb_3e_cf_00_11       — IMU 6轴 50Hz
  env_ea_cb_3e_cf_00_11       — 环境温湿度 每60s
  bodytemp_ea_cb_3e_cf_00_11  — 脖颈温度   每60s

MySQL (生产服务器):
  pet_dog_behavior.d_72
  pet_dog_environment.d_72
  pet_dog_skin_assessment.d_72
  pet_dog_scratch_baseline.pet_skin_baseline (device_id=72)

场景: 健康犬，中段出现皮肤炎症发作，后恢复正常
  day   0~59  : 正常（每天约 10 次抓挠）
  day  60~89  : 炎症期（每天约 35 次抓挠，颈温升高）
  day  90~180 : 恢复正常

运行: python device72_db.py
"""

import os
import math
import time
import threading
import requests
import pymysql
import pymysql.cursors
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
#  设备信息（固定）
# ══════════════════════════════════════════════════════
DEVICE_ID  = 72
DEVICE_SN  = "EA:CB:3E:CF:00:11"
DEVICE_TZ  = "America/New_York"

# ── 场景参数 ──────────────────────────────────────────
PHASES = [
    (0,   60,  10.0, 2.0),   # 正常期
    (60,  90,  35.0, 5.0),   # 炎症期
    (90,  180, 10.0, 2.0),   # 恢复期
]
SICK_RANGE = (60, 90)        # 炎症天区间
TC         = 0.10            # 温度系数

# ── 时间设置 ─────────────────────────────────────────
DAYS       = 180
START_DATE = date(2024, 1, 1)

# ── 采样率 ───────────────────────────────────────────
IMU_HZ               = int(os.environ.get("IMU_SAMPLE_HZ",        "50"))
ENV_SAMPLE_INTERVAL  = int(os.environ.get("ENV_SAMPLE_INTERVAL",  "60"))
NECK_SAMPLE_INTERVAL = int(os.environ.get("NECK_SAMPLE_INTERVAL", "60"))

# ── IMU 物理极值 ──────────────────────────────────────
ACC_MAX  = 78.46
GYRO_MAX = 17.87

BEHAVIOR_MOVE    = 1
BEHAVIOR_SLEEP   = 2
BEHAVIOR_SCRATCH = 3

# ── 性能参数 ─────────────────────────────────────────
IMU_CHUNK = int(os.environ.get("IMU_CHUNK",  "100000"))
ENV_CHUNK = int(os.environ.get("ENV_CHUNK",  "5000"))
HTTP_CONC = int(os.environ.get("HTTP_CONC",  "8"))

# ══════════════════════════════════════════════════════
#  TDengine 连接
# ══════════════════════════════════════════════════════
TD_HOST = os.environ.get("TD_HOST", "192.168.33.253")
TD_PORT = int(os.environ.get("TD_PORT", "6041"))
TD_USER = os.environ.get("TD_USER", "root")
TD_PASS = os.environ.get("TD_PASS", "taosdata")
TD_DB   = os.environ.get("TD_DB",   "hiccpet_device")

_TD_URL  = f"http://{TD_HOST}:{TD_PORT}/rest/sql"
_session = requests.Session()
_session.auth = (TD_USER, TD_PASS)


def td_exec(sql: str) -> dict:
    resp = _session.post(_TD_URL, data=sql.encode('utf-8'), timeout=120)
    resp.raise_for_status()
    r = resp.json()
    if r.get('code', 0) != 0:
        raise RuntimeError(f"TDengine error: {r.get('desc', r)}")
    return r


# ══════════════════════════════════════════════════════
#  MySQL 连接
# ══════════════════════════════════════════════════════
MYSQL_HOST     = os.environ.get("MYSQL_HOST",     "192.168.33.253")
MYSQL_PORT     = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER     = os.environ.get("MYSQL_USER",     "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "Hicc-mysql-2026")


def mysql_conn(database: str = None):
    return pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT,
        user=MYSQL_USER, password=MYSQL_PASSWORD,
        database=database,
        cursorclass=pymysql.cursors.DictCursor,
        charset="utf8mb4",
    )


# ══════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════
def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def now_ms() -> int:
    return int(time.time() * 1000)


def sn_to_tbl(sn: str) -> str:
    return sn.lower().replace(':', '_').replace('-', '_')


def is_sick_day(day_idx: int) -> bool:
    return SICK_RANGE[0] <= day_idx < SICK_RANGE[1]


def scratch_count_for_day(day_idx: int, temp: float) -> int:
    for s, e, mean, std in PHASES:
        if s <= day_idx < e:
            rng = np.random.RandomState(day_idx + 1000)
            return max(0, int(rng.normal(mean + TC * (temp - 20), std)))
    return 0


# ══════════════════════════════════════════════════════
#  全局温湿度序列（seed=42）
# ══════════════════════════════════════════════════════
np.random.seed(42)
_temperature = (22 + 13 * np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, DAYS))
                + np.random.normal(0, 1.5, DAYS))
_humidity    = (65 + 15 * np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, DAYS))
                + np.random.normal(0, 3.0, DAYS))
np.random.seed(42)

# ══════════════════════════════════════════════════════
#  IMU 生成
# ══════════════════════════════════════════════════════
_IMU_CLIPS = np.array([ACC_MAX, ACC_MAX, ACC_MAX, GYRO_MAX, GYRO_MAX, GYRO_MAX])
_IMU_MEANS = {
    BEHAVIOR_SLEEP:   np.array([0.0, 0.0, 9.8, 0.0,  0.0,  0.0]),
    BEHAVIOR_MOVE:    np.array([0.0, 0.0, 9.8, 0.0,  0.0,  0.0]),
    BEHAVIOR_SCRATCH: np.array([0.0, 0.0, 9.8, 0.0,  0.0,  0.0]),
}
_IMU_STDS = {
    BEHAVIOR_SLEEP:   np.array([0.3,  0.3,  0.4,  0.05, 0.05, 0.04]),
    BEHAVIOR_MOVE:    np.array([4.0,  3.5,  5.0,  1.5,  1.2,  1.0]),
    BEHAVIOR_SCRATCH: np.array([12.0, 8.0,  6.0,  5.0,  4.0,  3.5]),
}


def gen_imu_batch(n: int, behavior: int, si: float, rng: np.random.RandomState) -> np.ndarray:
    stds = _IMU_STDS[behavior] * (si if behavior == BEHAVIOR_SCRATCH else 1.0)
    data = rng.normal(_IMU_MEANS[behavior], stds, size=(n, 6))
    return np.round(np.clip(data, -_IMU_CLIPS, _IMU_CLIPS), 2)


def gen_imu_day(day_idx: int, n_scratch: int, sick_intensity: float,
                rng: np.random.RandomState) -> tuple:
    d      = START_DATE + timedelta(days=day_idx)
    day_ts = to_ts(d)
    step_ms = int(1000 / IMU_HZ)

    s_morn = n_scratch // 3
    s_aftn = n_scratch - s_morn
    segments = [
        (0,         7 * 3600,  0.85, 0),
        (7 * 3600,  12 * 3600, 0.10, s_morn),
        (12 * 3600, 14 * 3600, 0.80, 0),
        (14 * 3600, 20 * 3600, 0.10, s_aftn),
        (20 * 3600, 24 * 3600, 0.75, 0),
    ]

    ts_parts, imu_parts = [], []
    for seg_s, seg_e, sleep_w, n_sc in segments:
        scratch_times = sorted(rng.randint(seg_s, seg_e, int(n_sc)).tolist()) if n_sc > 0 else []
        sc_idx = 0
        cursor = seg_s
        while cursor < seg_e:
            if sc_idx < len(scratch_times) and cursor >= scratch_times[sc_idx]:
                dur_sec = int(rng.uniform(1, 8))
                btype, si = BEHAVIOR_SCRATCH, sick_intensity
                sc_idx += 1
            else:
                btype = BEHAVIOR_SLEEP if rng.random() < sleep_w else BEHAVIOR_MOVE
                dur_sec = (int(rng.uniform(600, 3600)) if btype == BEHAVIOR_SLEEP
                           else int(rng.uniform(60, 900)))
                si = 1.0
            dur_sec = min(dur_sec, seg_e - cursor)
            if dur_sec <= 0:
                break
            n_samp = dur_sec * IMU_HZ
            imu = gen_imu_batch(n_samp, btype, si, rng)
            ts  = np.arange(n_samp, dtype=np.int64) * step_ms + (day_ts + cursor * 1000)
            ts_parts.append(ts)
            imu_parts.append(imu)
            cursor += dur_sec

    return np.concatenate(ts_parts), np.concatenate(imu_parts, axis=0)


def gen_env_day(day_idx: int, rng: np.random.RandomState) -> list:
    d      = START_DATE + timedelta(days=day_idx)
    day_ts = to_ts(d)
    secs   = np.arange(0, 86400, ENV_SAMPLE_INTERVAL)
    n      = len(secs)
    temps  = np.round(float(_temperature[day_idx]) + rng.normal(0, 0.3, n), 1)
    humis  = np.round(float(_humidity[day_idx])    + rng.normal(0, 1.0, n), 1)
    ts_arr = (day_ts + secs * 1000).astype(np.int64)
    return list(zip(ts_arr.tolist(), temps.tolist(), humis.tolist()))


def gen_neck_day(day_idx: int, sick_intensity: float, rng: np.random.RandomState) -> list:
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
#  TDengine 写入（并发）
# ══════════════════════════════════════════════════════
SN_SUFFIX = sn_to_tbl(DEVICE_SN)
IMU_TBL   = f"{TD_DB}.imu_{SN_SUFFIX}"
ENV_TBL   = f"{TD_DB}.env_{SN_SUFFIX}"
NECK_TBL  = f"{TD_DB}.bodytemp_{SN_SUFFIX}"


def _send_concurrent(sqls: list):
    with ThreadPoolExecutor(max_workers=HTTP_CONC) as pool:
        for f in [pool.submit(td_exec, sql) for sql in sqls]:
            f.result()


def td_init():
    td_exec(f"CREATE TABLE IF NOT EXISTS {IMU_TBL}  USING {TD_DB}.imu_data       TAGS ('{DEVICE_SN}')")
    td_exec(f"CREATE TABLE IF NOT EXISTS {ENV_TBL}  USING {TD_DB}.env_data        TAGS ('{DEVICE_SN}')")
    td_exec(f"CREATE TABLE IF NOT EXISTS {NECK_TBL} USING {TD_DB}.body_temp_data  TAGS ('{DEVICE_SN}')")
    print(f"  [TD] 子表已就绪: {SN_SUFFIX}")


def td_insert_imu(ts_arr: np.ndarray, imu_arr: np.ndarray):
    ts_l  = ts_arr.tolist()
    ax = imu_arr[:, 0].tolist(); ay = imu_arr[:, 1].tolist()
    az = imu_arr[:, 2].tolist(); gx = imu_arr[:, 3].tolist()
    gy = imu_arr[:, 4].tolist(); gz = imu_arr[:, 5].tolist()
    n = len(ts_l)
    sqls = []
    for i in range(0, n, IMU_CHUNK):
        chunk = range(i, min(i + IMU_CHUNK, n))
        vals = " ".join(f"({ts_l[j]},{ax[j]},{ay[j]},{az[j]},{gx[j]},{gy[j]},{gz[j]})" for j in chunk)
        sqls.append(f"INSERT INTO {IMU_TBL} VALUES {vals}")
    _send_concurrent(sqls)


def td_insert_env(rows: list):
    sqls = []
    for i in range(0, len(rows), ENV_CHUNK):
        b = rows[i:i + ENV_CHUNK]
        vals = " ".join(f"({r[0]},{r[1]},{r[2]})" for r in b)
        sqls.append(f"INSERT INTO {ENV_TBL} VALUES {vals}")
    _send_concurrent(sqls)


def td_insert_neck(rows: list):
    sqls = []
    for i in range(0, len(rows), ENV_CHUNK):
        b = rows[i:i + ENV_CHUNK]
        vals = " ".join(f"({r[0]},{r[1]})" for r in b)
        sqls.append(f"INSERT INTO {NECK_TBL} VALUES {vals}")
    _send_concurrent(sqls)


# ══════════════════════════════════════════════════════
#  MySQL 写入
# ══════════════════════════════════════════════════════

def mysql_init():
    conn = mysql_conn()
    cur  = conn.cursor()

    # pet_dog_environment.d_72
    cur.execute("""
        CREATE TABLE IF NOT EXISTS `pet_dog_environment`.`d_72` (
          `ts`            BIGINT       NOT NULL,
          `env_temp`      DECIMAL(5,2) DEFAULT NULL,
          `env_humidity`  DECIMAL(5,1) DEFAULT NULL,
          `neck_temp`     DECIMAL(5,2) DEFAULT NULL,
          `local_date`    VARCHAR(12)  DEFAULT NULL,
          `user_timezone` VARCHAR(32)  DEFAULT NULL,
          `created_at`    BIGINT       NOT NULL,
          `updated_at`    BIGINT       NOT NULL,
          PRIMARY KEY (`ts`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # pet_dog_behavior.d_72
    cur.execute("""
        CREATE TABLE IF NOT EXISTS `pet_dog_behavior`.`d_72` (
          `id`            BIGINT       NOT NULL AUTO_INCREMENT,
          `ts_start`      BIGINT       NOT NULL,
          `ts_end`        BIGINT       NOT NULL,
          `behavior`      SMALLINT     NOT NULL,
          `duration_sec`  DECIMAL(10,2) NOT NULL,
          `confidence`    DECIMAL(5,3)  NOT NULL,
          `local_start`   VARCHAR(24)  DEFAULT NULL,
          `local_end`     VARCHAR(24)  DEFAULT NULL,
          `user_timezone` VARCHAR(32)  DEFAULT NULL,
          PRIMARY KEY (`id`),
          UNIQUE KEY `uq_ts_start` (`ts_start`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # pet_dog_skin_assessment.d_72
    cur.execute("""
        CREATE TABLE IF NOT EXISTS `pet_dog_skin_assessment`.`d_72` (
          `stat_date_ts`       BIGINT        NOT NULL,
          `local_date`         VARCHAR(12)   DEFAULT NULL,
          `user_timezone`      VARCHAR(32)   DEFAULT NULL,
          `scratch_count`      INT           NOT NULL DEFAULT 0,
          `scratch_duration`   BIGINT        NOT NULL DEFAULT 0,
          `scratch_avg_dur`    INT           NOT NULL DEFAULT 0,
          `scratch_max_dur`    INT           NOT NULL DEFAULT 0,
          `night_scratch_count` INT          NOT NULL DEFAULT 0,
          `wpeb_score`         DECIMAL(10,4) NOT NULL DEFAULT 0,
          `avg_humidity`       DECIMAL(5,1)  DEFAULT NULL,
          `baseline_mean`      DECIMAL(6,2)  DEFAULT NULL,
          `baseline_std`       DECIMAL(6,2)  DEFAULT NULL,
          `temp_coef`          DECIMAL(5,3)  DEFAULT NULL,
          `temp_effect`        DECIMAL(6,2)  DEFAULT NULL,
          `zscore`             DECIMAL(6,2)  DEFAULT NULL,
          `avg_zscore`         DECIMAL(6,2)  DEFAULT NULL,
          `consec_abnormal`    INT           NOT NULL DEFAULT 0,
          `eval_phase`         SMALLINT      NOT NULL DEFAULT 0,
          `threshold_z`        DECIMAL(4,2)  DEFAULT NULL,
          `threshold_consec`   SMALLINT      DEFAULT NULL,
          `threshold_avgz`     DECIMAL(4,2)  DEFAULT NULL,
          `valid_days`         INT           NOT NULL DEFAULT 0,
          `is_abnormal`        SMALLINT      NOT NULL DEFAULT 0,
          `alert_level`        SMALLINT      NOT NULL DEFAULT 0,
          `alert_reason`       VARCHAR(256)  DEFAULT NULL,
          `s1_score`           DECIMAL(5,1)  NOT NULL DEFAULT 0,
          `s2_score`           DECIMAL(5,1)  NOT NULL DEFAULT 0,
          `s3_score`           DECIMAL(5,1)  NOT NULL DEFAULT 0,
          `s4_score`           DECIMAL(5,1)  NOT NULL DEFAULT 0,
          `s5_score`           DECIMAL(5,1)  NOT NULL DEFAULT 0,
          `s6_score`           DECIMAL(5,1)  NOT NULL DEFAULT 0,
          `total_score`        DECIMAL(6,1)  NOT NULL DEFAULT 0,
          `health_level`       SMALLINT      NOT NULL DEFAULT 0,
          `data_quality`       SMALLINT      NOT NULL DEFAULT 0,
          `wear_minutes`       INT           NOT NULL DEFAULT 0,
          `worn_loose_minutes` DECIMAL(8,1)  NOT NULL DEFAULT 0,
          `created_at`         BIGINT        NOT NULL,
          `updated_at`         BIGINT        NOT NULL,
          PRIMARY KEY (`stat_date_ts`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("  [MySQL] 三张表已就绪 (d_72)")


def mysql_insert_env(rows_by_day: list):
    """rows_by_day: list of (ts_ms, temperature, humidity) per sample"""
    conn = mysql_conn()
    cur  = conn.cursor()
    now  = now_ms()
    data = []
    for ts_ms, temp, humi in rows_by_day:
        d = datetime.utcfromtimestamp(ts_ms / 1000)
        local_date = d.strftime('%Y-%m-%d')
        data.append((ts_ms, round(temp, 2), round(humi, 1), None, local_date, DEVICE_TZ, now, now))
    cur.executemany("""
        INSERT INTO `pet_dog_environment`.`d_72`
          (ts, env_temp, env_humidity, neck_temp, local_date, user_timezone, created_at, updated_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE updated_at=VALUES(updated_at)
    """, data)
    conn.commit()
    print(f"  [MySQL env] 插入 {cur.rowcount} 条")
    cur.close()
    conn.close()


def mysql_insert_behavior(seg_rows: list):
    """seg_rows: list of (ts_start, ts_end, behavior, duration_sec, confidence)"""
    conn = mysql_conn()
    cur  = conn.cursor()
    data = []
    for ts_start, ts_end, behavior, dur, conf in seg_rows:
        fmt = lambda ms: datetime.utcfromtimestamp(ms/1000).strftime('%Y-%m-%d %H:%M:%S')
        data.append((ts_start, ts_end, behavior, round(dur, 2), round(conf, 3),
                     fmt(ts_start), fmt(ts_end), DEVICE_TZ))
    cur.executemany("""
        INSERT IGNORE INTO `pet_dog_behavior`.`d_72`
          (ts_start, ts_end, behavior, duration_sec, confidence, local_start, local_end, user_timezone)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """, data)
    conn.commit()
    print(f"  [MySQL behavior] 插入 {cur.rowcount} 条行为片段")
    cur.close()
    conn.close()


def mysql_insert_daily_assessment(day_rows: list):
    """day_rows: list of dicts with all assessment fields"""
    conn = mysql_conn()
    cur  = conn.cursor()
    now  = now_ms()
    data = [(
        r['stat_date_ts'], r['local_date'], DEVICE_TZ,
        r['scratch_count'], r['scratch_duration'], r['scratch_avg_dur'],
        r['scratch_max_dur'], r['night_scratch_count'],
        r['wpeb_score'], r.get('avg_humidity'),
        r.get('baseline_mean'), r.get('baseline_std'),
        TC, r.get('temp_effect', 0.0),
        r.get('zscore'), r.get('avg_zscore'),
        r['consec_abnormal'], r['eval_phase'],
        r.get('threshold_z'), r.get('threshold_consec'), r.get('threshold_avgz'),
        r['valid_days'], r['is_abnormal'], r['alert_level'], r.get('alert_reason'),
        r['s1_score'], r['s2_score'], r['s3_score'],
        r['s4_score'], r['s5_score'], r['s6_score'],
        r['total_score'], r['health_level'],
        r['data_quality'], r['wear_minutes'], r['worn_loose_minutes'],
        now, now,
    ) for r in day_rows]
    cur.executemany("""
        INSERT INTO `pet_dog_skin_assessment`.`d_72`
          (stat_date_ts, local_date, user_timezone,
           scratch_count, scratch_duration, scratch_avg_dur, scratch_max_dur, night_scratch_count,
           wpeb_score, avg_humidity, baseline_mean, baseline_std, temp_coef, temp_effect,
           zscore, avg_zscore, consec_abnormal, eval_phase,
           threshold_z, threshold_consec, threshold_avgz,
           valid_days, is_abnormal, alert_level, alert_reason,
           s1_score, s2_score, s3_score, s4_score, s5_score, s6_score,
           total_score, health_level, data_quality, wear_minutes, worn_loose_minutes,
           created_at, updated_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE updated_at=VALUES(updated_at), scratch_count=VALUES(scratch_count)
    """, data)
    conn.commit()
    print(f"  [MySQL assessment] 插入 {cur.rowcount} 天评估")
    cur.close()
    conn.close()


def mysql_upsert_baseline(baseline_mean: float, baseline_std: float, valid_days: int):
    conn = mysql_conn()
    cur  = conn.cursor()
    now  = now_ms()
    cur.execute("""
        INSERT INTO `pet_dog_scratch_baseline`.`pet_skin_baseline`
          (device_id, baseline_mean, baseline_std, temp_coef, valid_days,
           eval_phase, confidence, wpeb_mean, wpeb_std, last_updated_ts, created_at)
        VALUES (%s,%s,%s,%s,%s,1,0.85,NULL,NULL,%s,%s)
        ON DUPLICATE KEY UPDATE
          baseline_mean=VALUES(baseline_mean), baseline_std=VALUES(baseline_std),
          valid_days=VALUES(valid_days), last_updated_ts=VALUES(last_updated_ts)
    """, (DEVICE_ID, round(baseline_mean, 2), round(baseline_std, 2), TC, valid_days, now, now))
    conn.commit()
    print(f"  [MySQL baseline] baseline_mean={baseline_mean:.2f}  baseline_std={baseline_std:.2f}  valid_days={valid_days}")
    cur.close()
    conn.close()


# ══════════════════════════════════════════════════════
#  每日评估计算
# ══════════════════════════════════════════════════════

def compute_daily_assessment(day_idx: int, scratch_count: int, temp: float,
                             scratch_history: list, valid_days: int) -> dict:
    d           = START_DATE + timedelta(days=day_idx)
    stat_date_ts = to_ts(d)
    local_date  = d.strftime('%Y-%m-%d')
    humi        = round(float(_humidity[day_idx]), 1)

    # 基线（用前30天正常期均值估算）
    warmup     = day_idx < 14
    eval_phase = 0 if warmup else 1
    bsl_mean   = 10.0
    bsl_std    = 2.0

    # zscore
    if not warmup and bsl_std > 0:
        zscore = round((scratch_count - bsl_mean) / bsl_std, 2)
    else:
        zscore = None

    history_zscores = [z for z in scratch_history[-7:] if z is not None]
    avg_zscore = round(sum(history_zscores) / len(history_zscores), 2) if history_zscores else None

    # 连续异常天计数
    consec = 0
    for z in reversed(scratch_history):
        if z is not None and z > 2.0:
            consec += 1
        else:
            break

    is_abnormal = 1 if (zscore is not None and zscore > 2.0 and consec >= 3) else 0
    alert_level = 2 if (is_abnormal and consec >= 7) else (1 if is_abnormal else 0)
    alert_reason = f"连续{consec}天异常，zscore={zscore}" if is_abnormal else None

    avg_dur = 30
    s_dur   = scratch_count * avg_dur
    s_max   = avg_dur * 2 if scratch_count > 0 else 0
    night_sc = int(scratch_count * 0.3)

    wpeb = round(abs(zscore) * 10, 4) if zscore else 0.0
    s1 = round(min(scratch_count * 0.5, 30.0), 1)
    s2 = round(min(s_dur / 60.0, 20.0), 1)
    s3 = round(min(night_sc * 2.0, 15.0), 1)
    s4 = round(min(abs(zscore or 0) * 5, 15.0), 1)
    s5 = round(min(consec * 2.0, 10.0), 1)
    s6 = round(min((humi - 40) * 0.1, 10.0) if humi > 40 else 0.0, 1)
    total = round(s1 + s2 + s3 + s4 + s5 + s6, 1)
    health_level = 2 if total >= 60 else (1 if total >= 30 else 0)

    return {
        'stat_date_ts':       stat_date_ts,
        'local_date':         local_date,
        'scratch_count':      scratch_count,
        'scratch_duration':   s_dur,
        'scratch_avg_dur':    avg_dur,
        'scratch_max_dur':    s_max,
        'night_scratch_count': night_sc,
        'wpeb_score':         wpeb,
        'avg_humidity':       humi,
        'baseline_mean':      bsl_mean,
        'baseline_std':       bsl_std,
        'temp_effect':        round(TC * (temp - 20), 2),
        'zscore':             zscore,
        'avg_zscore':         avg_zscore,
        'consec_abnormal':    consec,
        'eval_phase':         eval_phase,
        'threshold_z':        2.0,
        'threshold_consec':   3,
        'threshold_avgz':     1.5,
        'valid_days':         valid_days,
        'is_abnormal':        is_abnormal,
        'alert_level':        alert_level,
        'alert_reason':       alert_reason,
        's1_score':           s1, 's2_score': s2, 's3_score': s3,
        's4_score':           s4, 's5_score': s5, 's6_score': s6,
        'total_score':        total,
        'health_level':       health_level,
        'data_quality':       1,
        'wear_minutes':       1440,
        'worn_loose_minutes': 0.0,
    }


# ══════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(f"  设备 {DEVICE_ID}  SN={DEVICE_SN}")
    print(f"  天数={DAYS}  IMU={IMU_HZ}Hz  ENV每{ENV_SAMPLE_INTERVAL}s")
    print(f"  TDengine={TD_HOST}  MySQL={MYSQL_HOST}")
    print("=" * 60)

    print("\n[1] 初始化表结构...")
    td_init()
    mysql_init()

    rng = np.random.RandomState(72)

    all_env_rows      = []
    all_behavior_rows = []
    all_assessment    = []
    zscore_history    = []
    valid_days        = 0
    scratch_counts    = []

    print(f"\n[2] 生成并写入 {DAYS} 天数据...")
    t0 = time.time()

    for day_idx in range(DAYS):
        d    = START_DATE + timedelta(days=day_idx)
        temp = float(_temperature[day_idx])
        n_sc = scratch_count_for_day(day_idx, temp)
        sick = 1.8 if is_sick_day(day_idx) else 1.0

        # ── TDengine: IMU ─────────────────────────────
        ts_arr, imu_arr = gen_imu_day(day_idx, n_sc, sick, rng)
        td_insert_imu(ts_arr, imu_arr)

        # ── TDengine: ENV + NECK ──────────────────────
        env_rows  = gen_env_day(day_idx, rng)
        neck_rows = gen_neck_day(day_idx, sick, rng)
        td_insert_env(env_rows)
        td_insert_neck(neck_rows)

        # ── MySQL: env (按天汇总第一条作为代表) ──────
        # 取当天第一条采样作为日汇总
        day_ts = to_ts(d)
        neck_val = neck_rows[0][1] if neck_rows else None
        env_temp_val  = round(temp, 2)
        env_humi_val  = round(float(_humidity[day_idx]), 1)
        now = now_ms()
        conn = mysql_conn()
        cur  = conn.cursor()
        ld   = d.strftime('%Y-%m-%d')
        cur.execute("""
            INSERT INTO `pet_dog_environment`.`d_72`
              (ts, env_temp, env_humidity, neck_temp, local_date, user_timezone, created_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE updated_at=VALUES(updated_at)
        """, (day_ts, env_temp_val, env_humi_val, neck_val, ld, DEVICE_TZ, now, now))
        conn.commit()
        cur.close()
        conn.close()

        # ── MySQL: behavior 片段 ──────────────────────
        seg_ts = day_ts
        seg_rows = []
        for _, n_samp, btype, si in _iter_segments(day_idx, n_sc, sick, rng):
            dur_sec  = n_samp / IMU_HZ
            ts_end   = seg_ts + int(dur_sec * 1000)
            conf     = round(0.5 + rng.random() * 0.5, 3)
            seg_rows.append((seg_ts, ts_end, btype, dur_sec, conf))
            seg_ts   = ts_end
        mysql_insert_behavior(seg_rows)

        # ── MySQL: 每日评估 ───────────────────────────
        assess = compute_daily_assessment(day_idx, n_sc, temp, zscore_history, valid_days)
        zscore_history.append(assess['zscore'])
        scratch_counts.append(n_sc)
        if not (day_idx < 14):
            valid_days += 1
        all_assessment.append(assess)

        elapsed = time.time() - t0
        eta     = elapsed / (day_idx + 1) * (DAYS - day_idx - 1)
        print(f"  day {day_idx+1:3d}/{DAYS}  scratch={n_sc:3d}  "
              f"zscore={assess['zscore']}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    # 批量写评估
    print("\n[3] 写入每日评估...")
    mysql_insert_daily_assessment(all_assessment)

    # 写入基线
    print("\n[4] 写入抓挠基线...")
    normal_counts = [scratch_counts[i] for i in range(min(60, DAYS))]
    bsl_mean = float(np.mean(normal_counts))
    bsl_std  = float(np.std(normal_counts))
    mysql_upsert_baseline(bsl_mean, bsl_std, valid_days)

    print(f"\n[完成] 总耗时 {time.time()-t0:.1f}s")


def _iter_segments(day_idx, n_scratch, sick_intensity, rng):
    """重放当天的行为段（仅用于提取行为片段时间轴，不重新生成 IMU 数值）"""
    rng2 = np.random.RandomState(rng.randint(0, 2**31))
    s_morn = n_scratch // 3
    s_aftn = n_scratch - s_morn
    segments = [
        (0,         7*3600,  0.85, 0),
        (7*3600,    12*3600, 0.10, s_morn),
        (12*3600,   14*3600, 0.80, 0),
        (14*3600,   20*3600, 0.10, s_aftn),
        (20*3600,   24*3600, 0.75, 0),
    ]
    d      = START_DATE + timedelta(days=day_idx)
    day_ts = to_ts(d)
    for seg_s, seg_e, sleep_w, n_sc in segments:
        scratch_times = sorted(rng2.randint(seg_s, seg_e, int(n_sc)).tolist()) if n_sc > 0 else []
        sc_idx = 0
        cursor = seg_s
        while cursor < seg_e:
            if sc_idx < len(scratch_times) and cursor >= scratch_times[sc_idx]:
                dur_sec = int(rng2.uniform(1, 8))
                btype   = BEHAVIOR_SCRATCH
                sc_idx += 1
            else:
                btype   = BEHAVIOR_SLEEP if rng2.random() < sleep_w else BEHAVIOR_MOVE
                dur_sec = (int(rng2.uniform(600, 3600)) if btype == BEHAVIOR_SLEEP
                           else int(rng2.uniform(60, 900)))
            dur_sec = min(dur_sec, seg_e - cursor)
            if dur_sec <= 0:
                break
            yield day_ts + cursor * 1000, dur_sec * IMU_HZ, btype, sick_intensity
            cursor += dur_sec


if __name__ == "__main__":
    main()
