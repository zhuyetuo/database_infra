"""
Dual-device real-time data simulator (batch pipeline)
======================================================
Pipeline:

  [Device] every WINDOW_MINUTES -> upload a batch
       |
  [TDengine] imu_raw       - raw IMU samples at IMU_SAMPLE_HZ
  [TDengine] env_raw       - env temp/humidity samples at ENV_SAMPLE_INTERVAL
  [TDengine] neck_temp_raw - neck temp samples at NECK_SAMPLE_INTERVAL
       |
  [behavior recognition]
       |
  [PostgreSQL behavior] - behavior events (move/sleep/scratch segments)
       |  (after WINDOWS_PER_DAY windows)
  [PostgreSQL skin_assessment / scratch_baseline] - daily health assessment

All parameters are injected via environment variables set in run_sim.sh.
"""

import os
import time
import signal
import math
import requests
import psycopg2
import numpy as np
from datetime import datetime, timezone, date, timedelta


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

# ============================================================
#  Config from environment variables
# ============================================================
TD_HOST     = os.environ.get("TD_HOST",     "127.0.0.1")
TD_PORT     = int(os.environ.get("TD_PORT", "6041"))
TD_USER     = os.environ.get("TD_USER",     "root")
TD_PASS     = os.environ.get("TD_PASS",     "taosdata")
TD_DB       = os.environ.get("TD_DB",       "pet_collar_raw")

PG_HOST     = os.environ.get("PG_HOST",     "127.0.0.1")
PG_PORT     = int(os.environ.get("PG_PORT", "5432"))
PG_USER     = os.environ.get("PG_USER",     "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "123456")
PG_DB       = os.environ.get("PG_DB",       "pet_collar")

# Window / timing
WINDOW_MINUTES  = int(os.environ.get("WINDOW_MINUTES",  "15"))  # data window length (min)
WINDOW_SEC      = int(os.environ.get("WINDOW_SEC",      "15"))  # real seconds to wait per window
WINDOWS_PER_DAY = int(os.environ.get("WINDOWS_PER_DAY", "12"))  # windows per simulated day

# Sampling rates
IMU_SAMPLE_HZ        = int(os.environ.get("IMU_SAMPLE_HZ",        "25"))   # IMU Hz
ENV_SAMPLE_INTERVAL  = int(os.environ.get("ENV_SAMPLE_INTERVAL",  "60"))   # env temp/humidity (sec)
NECK_SAMPLE_INTERVAL = int(os.environ.get("NECK_SAMPLE_INTERVAL", "60"))   # neck temp (sec)

# Scenario
SICK_START_DAY = int(os.environ.get("SICK_START_DAY", "5"))

# Derived constants
WINDOW_SECONDS = WINDOW_MINUTES * 60  # window length in seconds

# ============================================================
#  Algorithm constants (must match skin_assessment_db.py)
# ============================================================
WARMUP   = 3
MIN_STD  = 2.0
NORMAL_W = 0.05
ABNORM_W = 0.01

BEHAVIOR_MOVE    = 1
BEHAVIOR_SLEEP   = 2
BEHAVIOR_SCRATCH = 3

DEVICES = [
    {"sn": "device_sn_1", "sick": False},
    {"sn": "device_sn_2", "sick": True},
]

# ============================================================
#  Graceful stop
# ============================================================
_running = True

def _handle_sigint(sig, frame):
    global _running
    print("\n\n[stop] Ctrl+C received, exiting...")
    _running = False

signal.signal(signal.SIGINT, _handle_sigint)


# ============================================================
#  TDengine REST
# ============================================================

def td_exec(sql: str) -> dict:
    url  = f"http://{TD_HOST}:{TD_PORT}/rest/sql"
    resp = requests.post(url, data=sql.encode("utf-8"),
                         auth=(TD_USER, TD_PASS), timeout=60)
    resp.raise_for_status()
    result = resp.json()
    if result.get("code", 0) != 0:
        raise RuntimeError(f"TDengine [{result.get('code')}]: {result.get('desc')}")
    return result


def td_init():
    td_exec(f"CREATE DATABASE IF NOT EXISTS {TD_DB} KEEP 3650 DURATION 10 COMP 2")

    # Super table: raw IMU samples
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

    # Super table: environment temperature + humidity
    td_exec(f"""
        CREATE STABLE IF NOT EXISTS {TD_DB}.env_raw (
            ts        TIMESTAMP,
            env_temp  FLOAT,
            env_humi  FLOAT
        ) TAGS (device_sn BINARY(64))
    """)

    # Super table: neck temperature
    td_exec(f"""
        CREATE STABLE IF NOT EXISTS {TD_DB}.neck_temp_raw (
            ts        TIMESTAMP,
            neck_temp FLOAT
        ) TAGS (device_sn BINARY(64))
    """)

    for dev in DEVICES:
        sn = dev["sn"]
        td_exec(f"CREATE TABLE IF NOT EXISTS {TD_DB}.{sn}_imu "
                f"USING {TD_DB}.imu_raw TAGS ('{sn}')")
        td_exec(f"CREATE TABLE IF NOT EXISTS {TD_DB}.{sn}_env "
                f"USING {TD_DB}.env_raw TAGS ('{sn}')")
        td_exec(f"CREATE TABLE IF NOT EXISTS {TD_DB}.{sn}_neck "
                f"USING {TD_DB}.neck_temp_raw TAGS ('{sn}')")

    print("[TDengine] DB & tables ready")


def td_insert_imu(sn: str, samples: list):
    """samples: list of (ts_ms, ax, ay, az, gx, gy, gz)"""
    CHUNK = 1000
    for i in range(0, len(samples), CHUNK):
        c    = samples[i: i + CHUNK]
        vals = " ".join(f"({r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]})" for r in c)
        td_exec(f"INSERT INTO {TD_DB}.{sn}_imu VALUES {vals}")


def td_insert_env(sn: str, env_samples: list, neck_samples: list):
    """
    env_samples  : list of (ts_ms, env_temp, env_humi)
    neck_samples : list of (ts_ms, neck_temp)
    """
    CHUNK = 500
    for i in range(0, len(env_samples), CHUNK):
        c    = env_samples[i: i + CHUNK]
        vals = " ".join(f"({r[0]},{r[1]},{r[2]})" for r in c)
        td_exec(f"INSERT INTO {TD_DB}.{sn}_env VALUES {vals}")

    for i in range(0, len(neck_samples), CHUNK):
        c    = neck_samples[i: i + CHUNK]
        vals = " ".join(f"({r[0]},{r[1]})" for r in c)
        td_exec(f"INSERT INTO {TD_DB}.{sn}_neck VALUES {vals}")


# ============================================================
#  PostgreSQL
# ============================================================

def pg_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER,
        password=PG_PASSWORD, dbname=PG_DB
    )


def pg_init():
    conn = pg_conn()
    cur  = conn.cursor()

    for schema in ["pet_dog_behavior", "pet_dog_skin_assessment", "pet_dog_scratch_baseline"]:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    for dev in DEVICES:
        sn = dev["sn"]

        # behavior: one row per recognized behavior segment
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS pet_dog_behavior.{sn} (
                id           BIGSERIAL     PRIMARY KEY,
                ts_start     BIGINT        NOT NULL,
                ts_end       BIGINT        NOT NULL,
                behavior     SMALLINT      NOT NULL,
                duration_sec NUMERIC(10,2) NOT NULL,
                confidence   NUMERIC(5,3)  NOT NULL,
                ax_mean      NUMERIC(8,2),
                ay_mean      NUMERIC(8,2),
                az_mean      NUMERIC(8,2),
                gx_mean      NUMERIC(8,2),
                gy_mean      NUMERIC(8,2),
                gz_mean      NUMERIC(8,2)
            )
        """)
        cur.execute(f"CREATE INDEX IF NOT EXISTS {sn}_beh_ts "
                    f"ON pet_dog_behavior.{sn} (ts_start)")

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
    print("[PostgreSQL] all tables ready")


# ============================================================
#  IMU sample generator
# ============================================================

ACC_MAX  =  78.46   # m/s²  (~±8g)
GYRO_MAX =  17.87   # rad/s (~±1024 °/s)

def _clip_imu(ax, ay, az, gx, gy, gz) -> tuple:
    c = lambda v, m: round(float(np.clip(v, -m, m)), 2)
    return (c(ax, ACC_MAX), c(ay, ACC_MAX), c(az, ACC_MAX),
            c(gx, GYRO_MAX), c(gy, GYRO_MAX), c(gz, GYRO_MAX))

def _imu_sample(behavior: int, si: float = 1.0) -> tuple:
    """Return (ax, ay, az, gx, gy, gz) for one sample. Units: m/s², rad/s."""
    if behavior == BEHAVIOR_SLEEP:
        # Nearly still; az ≈ +9.8 m/s² (gravity), minimal rotation
        return _clip_imu(
            np.random.normal(0,    0.3),
            np.random.normal(0,    0.3),
            np.random.normal(9.8,  0.4),
            np.random.normal(0,    0.05),
            np.random.normal(0,    0.05),
            np.random.normal(0,    0.04),
        )
    elif behavior == BEHAVIOR_MOVE:
        # Walking/trotting: moderate acc variance, noticeable rotation
        return _clip_imu(
            np.random.normal(0,   4.0),
            np.random.normal(0,   3.5),
            np.random.normal(9.8, 5.0),
            np.random.normal(0,   1.5),
            np.random.normal(0,   1.2),
            np.random.normal(0,   1.0),
        )
    else:  # SCRATCH — rapid limb movement, intensity scaled by si
        return _clip_imu(
            np.random.normal(0,  12.0 * si),
            np.random.normal(0,   8.0 * si),
            np.random.normal(9.8, 6.0 * si),
            np.random.normal(0,   5.0 * si),
            np.random.normal(0,   4.0 * si),
            np.random.normal(0,   3.5 * si),
        )


# ============================================================
#  Window data generation
# ============================================================

def generate_imu_window(window_start_ms: int, si: float) -> tuple:
    """
    Generate one window of IMU data.

    Returns:
      raw_samples  - list of (ts_ms, ax, ay, az, gx, gy, gz)  -> TDengine
      segments     - list of dicts (recognized behavior events) -> PostgreSQL
    """
    step_ms     = int(1000 / IMU_SAMPLE_HZ)
    total_steps = WINDOW_SECONDS * IMU_SAMPLE_HZ

    raw_samples = []
    segments    = []
    cursor_ms   = window_start_ms
    steps_done  = 0

    while steps_done < total_steps:
        hour = datetime.fromtimestamp(cursor_ms / 1000, tz=timezone.utc).hour

        # Behavior probability [sleep, move, scratch]
        if   0  <= hour < 6:  w = [0.80, 0.15, 0.05]
        elif 6  <= hour < 8:  w = [0.30, 0.55, 0.15]
        elif 8  <= hour < 12: w = [0.10, 0.70, 0.20]
        elif 12 <= hour < 14: w = [0.65, 0.25, 0.10]
        elif 14 <= hour < 20: w = [0.10, 0.70, 0.20]
        elif 20 <= hour < 22: w = [0.45, 0.45, 0.10]
        else:                  w = [0.75, 0.20, 0.05]

        boost = (si - 1.0) * 0.35
        w[2]  = min(w[2] + boost, 0.65)
        s     = sum(w); w = [x / s for x in w]

        btype = int(np.random.choice([BEHAVIOR_SLEEP, BEHAVIOR_MOVE, BEHAVIOR_SCRATCH], p=w))

        if btype == BEHAVIOR_SLEEP:
            seg_steps = np.random.randint(IMU_SAMPLE_HZ * 30,  IMU_SAMPLE_HZ * 600)
        elif btype == BEHAVIOR_MOVE:
            seg_steps = np.random.randint(IMU_SAMPLE_HZ * 5,   IMU_SAMPLE_HZ * 120)
        else:
            seg_steps = np.random.randint(IMU_SAMPLE_HZ * 1,   IMU_SAMPLE_HZ * 8)

        seg_steps    = min(seg_steps, total_steps - steps_done)
        if seg_steps <= 0:
            break

        seg_start_ms = cursor_ms
        seg_data     = []

        for _ in range(seg_steps):
            feat = _imu_sample(btype, si)
            raw_samples.append((cursor_ms,) + feat)
            seg_data.append(feat)
            cursor_ms  += step_ms
            steps_done += 1

        seg_end_ms = cursor_ms
        dur_sec    = round((seg_end_ms - seg_start_ms) / 1000.0, 2)

        if btype == BEHAVIOR_SLEEP:
            conf = round(np.random.uniform(0.88, 0.99), 3)
        elif btype == BEHAVIOR_MOVE:
            conf = round(np.random.uniform(0.82, 0.97), 3)
        else:
            conf = round(np.random.uniform(0.72, 0.93), 3)

        arr = np.array(seg_data)
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


def generate_env_window(window_start_ms: int, day_idx: int, si: float) -> tuple:
    """
    Generate env and neck-temp samples for one window.

    Returns:
      env_samples  - list of (ts_ms, env_temp, env_humi)  -> TDengine env_raw
      neck_samples - list of (ts_ms, neck_temp)           -> TDengine neck_temp_raw
    """
    doy      = (date.today() + timedelta(days=day_idx)).timetuple().tm_yday
    env_base = 22 + 13 * math.sin((doy - 80) / 365 * 2 * math.pi)
    hum_base = 65 + 15 * math.sin((doy - 80) / 365 * 2 * math.pi)

    env_samples  = []
    neck_samples = []
    cursor_s     = 0

    while cursor_s < WINDOW_SECONDS:
        ts_ms    = window_start_ms + cursor_s * 1000
        env_temp = round(env_base + np.random.normal(0, 1.5), 1)
        env_humi = round(hum_base + np.random.normal(0, 3.0), 1)
        env_samples.append((ts_ms, env_temp, env_humi))

        if cursor_s % NECK_SAMPLE_INTERVAL == 0:
            if si > 1.3:
                neck_temp = round(38.5 + np.random.uniform(0.0, 0.8) * (si - 0.3), 2)
            else:
                neck_temp = round(37.5 + np.random.uniform(-0.3, 0.3), 2)
            neck_samples.append((ts_ms, neck_temp))

        cursor_s += ENV_SAMPLE_INTERVAL

    return env_samples, neck_samples


# ============================================================
#  PostgreSQL: write behavior events
# ============================================================

def pg_insert_behavior(conn, sn: str, segments: list) -> int:
    """Write behavior segments, return scratch count."""
    if not segments:
        return 0
    cur = conn.cursor()
    cur.executemany(
        f"""
        INSERT INTO pet_dog_behavior.{sn}
            (ts_start, ts_end, behavior, duration_sec, confidence,
             ax_mean, ay_mean, az_mean, gx_mean, gy_mean, gz_mean)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT DO NOTHING
        """,
        [(s["ts_start"], s["ts_end"], s["behavior"], s["duration_sec"], s["confidence"],
          s["ax_mean"], s["ay_mean"], s["az_mean"],
          s["gx_mean"], s["gy_mean"], s["gz_mean"])
         for s in segments],
    )
    conn.commit()
    cur.close()
    return sum(1 for s in segments if s["behavior"] == BEHAVIOR_SCRATCH)


# ============================================================
#  Daily settlement: skin assessment + baseline
# ============================================================

def _thresholds(vd):
    if vd < 1:      return None, None, None
    elif vd <= 11:  return 4.0, 5, 5.0
    elif vd <= 27:  return 3.5, 4, 4.5
    else:           return 2.5, 3, 3.5

def _phase(vd):
    return 0 if vd == 0 else 1 if vd <= 11 else 2 if vd <= 27 else 3

def _temp_coef(bc, bt):
    if len(bc) < 20:
        return 0.0
    x = np.array(bt, float); y = np.array(bc, float)
    c = np.sum((x - x.mean()) * (y - y.mean())) / (np.sum((x - x.mean()) ** 2) + 1e-8)
    return round(float(np.clip(c, 0.0, 0.4)), 3)


def settle_day(conn, state):
    stat_date = (date.today() + timedelta(days=state.day_idx)).isoformat()
    count     = state.day_scratch_count
    si        = state.sick_intensity()
    wear_min  = int(np.random.uniform(1380, 1440))

    # estimate daily temperature from env_raw via buf_t
    temp = state.buf_t[-1] if state.buf_t else 20.0

    cur = conn.cursor()

    # warmup
    if state.day_idx < WARMUP:
        state.buf_c.append(count)
        cur.execute(
            f"""INSERT INTO pet_dog_skin_assessment.{state.sn}
                (stat_date, scratch_count, eval_phase, data_quality, wear_minutes)
                VALUES (%s,%s,0,0,%s)
                ON CONFLICT (stat_date) DO UPDATE
                  SET scratch_count=EXCLUDED.scratch_count, wear_minutes=EXCLUDED.wear_minutes""",
            (stat_date, count, wear_min),
        )
        conn.commit(); cur.close()
        print(f"  [{state.sn}] day {state.day_idx:3d} [warmup] scratch={count:3d}")
        state.day_scratch_count = 0
        state.day_idx += 1
        return

    if state.mean is None:
        state.mean = float(np.mean(state.buf_c)) if state.buf_c else float(count)
        state.std  = max(float(np.std(state.buf_c)) if len(state.buf_c) > 1 else MIN_STD, MIN_STD)

    state.valid_days += 1
    tz, tc, ta = _thresholds(state.valid_days)
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
    reason = (f"consec {state.consec}d z>{tz:.1f}, avg_z={avg_z:.2f}, scratch={count}"
              if alert else None)

    cur.execute(
        f"""INSERT INTO pet_dog_skin_assessment.{state.sn}
            (stat_date, scratch_count, baseline_mean, baseline_std,
             zscore, avg_zscore, consec_abnormal, eval_phase,
             threshold_z, threshold_consec, is_abnormal, alert_triggered,
             alert_reason, data_quality, wear_minutes)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,0,%s)
            ON CONFLICT (stat_date) DO UPDATE
              SET scratch_count=EXCLUDED.scratch_count,
                  baseline_mean=EXCLUDED.baseline_mean,
                  baseline_std=EXCLUDED.baseline_std,
                  zscore=EXCLUDED.zscore, avg_zscore=EXCLUDED.avg_zscore,
                  consec_abnormal=EXCLUDED.consec_abnormal,
                  is_abnormal=EXCLUDED.is_abnormal,
                  alert_triggered=EXCLUDED.alert_triggered,
                  alert_reason=EXCLUDED.alert_reason,
                  wear_minutes=EXCLUDED.wear_minutes""",
        (stat_date, count,
         round(state.mean, 2), round(state.std, 2), zscore, avg_z,
         state.consec, _phase(state.valid_days),
         tz, tc, int(is_abn), int(alert), reason, wear_min),
    )

    confidence = round(min(1.0, state.valid_days / 30), 2)
    cur.execute(
        f"""INSERT INTO pet_dog_scratch_baseline.{state.sn}
            (stat_date, baseline_mean, baseline_std, temp_coef, confidence, valid_days)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (stat_date) DO UPDATE
              SET baseline_mean=EXCLUDED.baseline_mean,
                  baseline_std=EXCLUDED.baseline_std,
                  temp_coef=EXCLUDED.temp_coef,
                  confidence=EXCLUDED.confidence,
                  valid_days=EXCLUDED.valid_days""",
        (stat_date, round(state.mean, 2), round(state.std, 2), coef, confidence, state.valid_days),
    )

    conn.commit(); cur.close()

    alert_tag = "  !! ALERT" if alert else ""
    abn_tag   = " [ABN]"    if is_abn else ""
    print(f"  [{state.sn}] day {state.day_idx:3d}  scratch={count:3d}  "
          f"z={zscore:+.2f}  baseline={state.mean:.1f}+/-{state.std:.1f}  "
          f"phase={_phase(state.valid_days)}{abn_tag}{alert_tag}")

    state.day_scratch_count = 0
    state.day_idx += 1


# ============================================================
#  Device state
# ============================================================

class DeviceState:
    def __init__(self, sn, is_sick):
        self.sn             = sn
        self.is_sick        = is_sick
        self.day_idx        = 0
        self.window_in_day  = 0
        self.mean           = None
        self.std            = MIN_STD
        self.buf_c          = []
        self.buf_t          = []
        self.consec         = 0
        self.valid_days     = 0
        self.recent_z       = []
        self.day_scratch_count = 0

    def sick_intensity(self):
        if not self.is_sick or self.day_idx < SICK_START_DAY:
            return 1.0
        progress = min((self.day_idx - SICK_START_DAY) / 20.0, 1.0)
        return 1.0 + 1.2 * progress


# ============================================================
#  Per-window processing
# ============================================================

def process_window(state, conn, window_ts_ms):
    si = state.sick_intensity()

    # --- IMU: generate raw samples + run behavior recognition ---
    raw_imu, segments = generate_imu_window(window_ts_ms, si)

    n_imu = 0
    try:
        td_insert_imu(f"{state.sn}", raw_imu)
        n_imu = len(raw_imu)
    except Exception as e:
        print(f"    [warn] TDengine IMU write failed: {e}")

    scratch_in_window = 0
    try:
        scratch_in_window      = pg_insert_behavior(conn, state.sn, segments)
        state.day_scratch_count += scratch_in_window
    except Exception as e:
        conn.rollback()
        print(f"    [warn] PG behavior write failed: {e}")

    # --- Env + neck temp: generate samples at configured intervals ---
    env_samples, neck_samples = generate_env_window(window_ts_ms, state.day_idx, si)

    # Keep a running temp average for daily assessment
    if env_samples:
        state.buf_t.append(round(float(np.mean([s[1] for s in env_samples])), 1))

    n_env = 0
    n_neck = 0
    try:
        td_insert_env(state.sn, env_samples, neck_samples)
        n_env  = len(env_samples)
        n_neck = len(neck_samples)
    except Exception as e:
        print(f"    [warn] TDengine env write failed: {e}")
    n_move  = sum(1 for s in segments if s["behavior"] == BEHAVIOR_MOVE)
    n_sleep = sum(1 for s in segments if s["behavior"] == BEHAVIOR_SLEEP)

    print(f"  [{state.sn}] "
          f"day{state.day_idx}-w{state.window_in_day+1:02d}  "
          f"imu={n_imu:,}pts  "
          f"segs={len(segments)}(mv{n_move}/sl{n_sleep}/sc{scratch_in_window})  "
          f"env={n_env}pts neck={n_neck}pts  "
          f"si={si:.2f}")

    state.window_in_day += 1
    if state.window_in_day >= WINDOWS_PER_DAY:
        state.window_in_day = 0
        settle_day(conn, state)


# ============================================================
#  Main loop
# ============================================================

def _now_ms():
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def main():
    imu_pts_per_window = WINDOW_SECONDS * IMU_SAMPLE_HZ
    env_pts_per_window = WINDOW_SECONDS // ENV_SAMPLE_INTERVAL
    neck_pts_per_window = WINDOW_SECONDS // NECK_SAMPLE_INTERVAL

    print("=" * 65)
    print("  Pet collar dual-device simulator")
    print(f"  sim_device_normal : always healthy")
    print(f"  sim_device_sick   : sick from day {SICK_START_DAY}")
    print()
    print(f"  Window length      : {WINDOW_MINUTES} min")
    print(f"  Real wait per win  : {WINDOW_SEC} s")
    print(f"  Windows per day    : {WINDOWS_PER_DAY}")
    print()
    print(f"  IMU sample rate    : {IMU_SAMPLE_HZ} Hz  "
          f"-> {imu_pts_per_window:,} pts/window")
    print(f"  Env sample interval: every {ENV_SAMPLE_INTERVAL} s  "
          f"-> {env_pts_per_window} pts/window")
    print(f"  Neck temp interval : every {NECK_SAMPLE_INTERVAL} s  "
          f"-> {neck_pts_per_window} pts/window")
    print()
    print("  Ctrl+C to stop")
    print("=" * 65)

    print("\n[init] TDengine...")
    td_init()
    print("[init] PostgreSQL...")
    pg_init()

    states  = [DeviceState(d["sn"], d["sick"]) for d in DEVICES]
    conn    = pg_conn()
    base_ms = _now_ms()
    win_idx = 0

    print(f"\n[running] processing one {WINDOW_MINUTES}-min window every {WINDOW_SEC}s ...\n")

    while _running:
        win_ts_ms = base_ms + win_idx * WINDOW_MINUTES * 60 * 1000
        win_idx  += 1

        ts_str = datetime.fromtimestamp(win_ts_ms / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")
        print(f"-- window #{win_idx}  sim-time {ts_str} --")

        for state in states:
            process_window(state, conn, win_ts_ms)

        if _running:
            time.sleep(WINDOW_SEC)

    conn.close()
    print("\n[done] simulator stopped")


if __name__ == "__main__":
    main()
