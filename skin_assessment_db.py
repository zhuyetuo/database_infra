"""
皮肤健康评估数据库（MySQL）
=====================
数据库: pet_dog_skin_assessment
每个设备独立一张每日评估表: d_{device_id}

24 个场景设备（与 imu_raw_db.py 一一对应）

评估算法:
  - 热身期（默认3天，SIM-D021 为7天）：只收集，不评估
  - 个体动态基线：异常天权重 0.01，正常天权重 0.05
  - 标准差保底 2.0
  - 温度修正：20 天数据后启用，系数上限 0.4
  - 动态阈值：早期 z>4.0 连续5天，过渡期 z>3.5 连续4天，稳定期 z>2.5 连续3天
  - 缺口处理：基线冻结 → 恢复后缓冲天（data_quality=5）→ 缺口≥30天重置门槛
"""

import os
import pymysql
import pymysql.cursors
import numpy as np
import math
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
MYSQL_HOST     = os.environ.get("MYSQL_HOST",     "127.0.0.1")
MYSQL_PORT     = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER     = os.environ.get("MYSQL_USER",     "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "Hicc-mysql-2026")
SKIN_SCHEMA    = "pet_dog_skin_assessment"

DAYS       = 180
WARMUP     = 3
MIN_STD    = 2.0
NORMAL_W   = 0.05
ABNORM_W   = 0.01
GAP_RESET  = 30
START_DATE = date(2024, 1, 1)

DATA_QUALITY_NORMAL   = 0
DATA_QUALITY_UNWORN   = 1
DATA_QUALITY_BATTERY  = 2
DATA_QUALITY_SIGNAL   = 3
DATA_QUALITY_LOOSE    = 4
DATA_QUALITY_BUFFER   = 5

GAP_REASON_TO_DQ = {
    'unworn':  DATA_QUALITY_UNWORN,
    'battery': DATA_QUALITY_BATTERY,
    'signal':  DATA_QUALITY_SIGNAL,
    'loose':   DATA_QUALITY_LOOSE,
}

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
    # 健康场景 1-9
    {
        'device_id': 1001,
        'sn':     'SIM-D001',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
    },
    {
        'device_id': 1002,
        'sn':     'SIM-D002',
        'phases': [(0, 60, 10.0, 2.0), (60, 80, 30.0, 4.0), (80, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   (60, 80),
    },
    {
        'device_id': 1003,
        'sn':     'SIM-D003',
        'phases': [(0, 60, 10.0, 2.0), (60, 180, 28.0, 4.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   (60, 180),
    },
    {
        'device_id': 1004,
        'sn':          'SIM-D004',
        'phases':      [(0, 40, 10.0, 2.0), (40, 55, 28.0, 4.0),
                        (55, 120, 10.0, 2.0), (120, 135, 30.0, 4.0),
                        (135, 180, 10.0, 2.0)],
        'tc':          0.10,
        'gaps':        [],
        'sick':        None,
        'sick_episodes': [(40, 55), (120, 135)],
    },
    {
        'device_id': 1005,
        'sn':     'SIM-D005',
        'phases': [(0, 60, 10.0, 2.0), (60, 120, 15.0, 2.0), (120, 180, 22.0, 3.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
    },
    {
        'device_id': 1006,
        'sn':     'SIM-D006',
        'phases': [(0, 90, 10.0, 2.0), (90, 180, 25.0, 3.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   (90, 180),
    },
    {
        'device_id': 1007,
        'sn':     'SIM-D007',
        'phases': [(0, 50, 10.0, 2.0), (50, 80, 45.0, 6.0), (80, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   (50, 80),
    },
    {
        'device_id': 1008,
        'sn':     'SIM-D008',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.35,
        'gaps':   [],
        'sick':   None,
    },
    {
        'device_id': 1009,
        'sn':     'SIM-D009',
        'phases': [(0, 30, 10.0, 2.0), (30, 90, 3.0, 1.0), (90, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
    },
    # 设备/数据质量场景 10-16
    {
        'device_id': 1010,
        'sn':     'SIM-D010',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(35, 38, 'unworn')],
        'sick':   None,
    },
    {
        'device_id': 1011,
        'sn':     'SIM-D011',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(40, 45, 'battery')],
        'sick':   None,
    },
    {
        'device_id': 1012,
        'sn':     'SIM-D012',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(30, 65, 'battery')],
        'sick':   None,
    },
    {
        'device_id': 1013,
        'sn':     'SIM-D013',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(d, d + 1, 'signal') for d in sorted(_signal_gap_days)],
        'sick':   None,
    },
    {
        'device_id': 1014,
        'sn':     'SIM-D014',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(50, 58, 'loose')],
        'sick':   None,
    },
    {
        'device_id': 1015,
        'sn':     'SIM-D015',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(88, 92, 'battery')],
        'sick':   None,
    },
    {
        'device_id': 1016,
        'sn':     'SIM-D016',
        'phases': [(0, 70, 10.0, 2.0), (70, 90, 35.0, 5.0), (90, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
        'drift_range': (70, 90),
    },
    # 环境场景 17-20
    {
        'device_id': 1017,
        'sn':     'SIM-D017',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.30,
        'gaps':   [],
        'sick':   None,
    },
    {
        'device_id': 1018,
        'sn':     'SIM-D018',
        'phases': [(0, 60, 10.0, 2.0), (60, 180, 13.0, 2.0)],
        'tc':     0.15,
        'gaps':   [],
        'sick':   None,
        'temp_shift': (60, 5.0),
    },
    {
        'device_id': 1019,
        'sn':     'SIM-D019',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(80, 90, 'unworn')],
        'sick':   None,
    },
    {
        'device_id': 1020,
        'sn':     'SIM-D020',
        'phases': [(0, 180, 14.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
    },
    # 个体类型场景 21-24
    {
        'device_id': 1021,
        'sn':     'SIM-D021',
        'phases': [(0, 180, 15.0, 4.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
        'warmup': 7,
    },
    {
        'device_id': 1022,
        'sn':     'SIM-D022',
        'phases': [(0, 180, 5.0, 1.0)],
        'tc':     0.05,
        'gaps':   [],
        'sick':   None,
    },
    {
        'device_id': 1023,
        'sn':     'SIM-D023',
        'phases': [(0, 180, 20.0, 3.0)],
        'tc':     0.12,
        'gaps':   [],
        'sick':   None,
    },
    {
        'device_id': 1024,
        'sn':     'SIM-D024',
        'phases': [(0, 180, 4.0, 1.0)],
        'tc':     0.08,
        'gaps':   [],
        'sick':   None,
    },
]


# ══════════════════════════════════════════════════════
#  算法函数
# ══════════════════════════════════════════════════════

def get_thresholds(valid_days: int):
    if valid_days < 1:      return None, None, None
    elif valid_days <= 11:  return 4.0, 5, 5.0
    elif valid_days <= 27:  return 3.5, 4, 4.5
    else:                   return 2.5, 3, 3.5


def get_phase(valid_days: int) -> int:
    if valid_days == 0:     return 0
    elif valid_days <= 11:  return 1
    elif valid_days <= 27:  return 2
    else:                   return 3


def estimate_temp_coef(buf_c: list, buf_t: list) -> float:
    if len(buf_c) < 20:
        return 0.0
    x = np.array(buf_t, dtype=float)
    y = np.array(buf_c, dtype=float)
    coef = (np.sum((x - x.mean()) * (y - y.mean()))
            / (np.sum((x - x.mean()) ** 2) + 1e-8))
    return round(float(np.clip(coef, 0.0, 0.4)), 3)


def scratch_count_for_day(day_idx: int, phases: list,
                          temp: float, temp_coef: float) -> int:
    for s, e, mean, std in phases:
        if s <= day_idx < e:
            return max(0, int(np.random.normal(mean + temp_coef * (temp - 20), std)))
    return 0


def is_sick_day(day_idx: int, sc: dict) -> bool:
    episodes = sc.get('sick_episodes')
    if episodes:
        return any(s <= day_idx < e for s, e in episodes)
    sick = sc.get('sick')
    if sick:
        return sick[0] <= day_idx < sick[1]
    return False


def safe(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def build_gap_map(gaps: list) -> dict:
    gm = {}
    for start, end, reason in gaps:
        for d in range(start, min(end, DAYS)):
            gm[d] = reason
    return gm


def to_date_str(d: date) -> str:
    return d.strftime('%Y-%m-%d')


def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def tbl(device_id: int) -> str:
    return f"`{SKIN_SCHEMA}`.`d_{device_id}`"


# ══════════════════════════════════════════════════════
#  核心：运行评估算法，构建每日行记录
# ══════════════════════════════════════════════════════

def build_daily_rows(sc: dict, seed: int = 42) -> list:
    np.random.seed(seed)

    phases    = sc['phases']
    temp_coef = sc['tc']
    gap_map   = build_gap_map(sc['gaps'])
    warmup    = sc.get('warmup', WARMUP)

    mean  = None
    std   = None
    buf_c = []
    buf_t = []
    consec      = 0
    valid_days  = 0
    gap_counter = 0
    in_gap      = False
    just_resumed = False
    recent_z_buf = []

    rows = []
    now  = now_ms()

    scratch_avg_dur_default = 30  # seconds

    for i in range(DAYS):
        d    = START_DATE + timedelta(days=i)
        stat_date_ts = to_ts(d)
        local_date   = to_date_str(d)
        temp = round(float(_temperature[i]), 1)
        sick = is_sick_day(i, sc)
        wear = int(np.random.uniform(1350, 1440))

        # ── 缺口天 ───────────────────────────────────────────
        if i in gap_map:
            gap_reason = gap_map[i]
            dq         = GAP_REASON_TO_DQ.get(gap_reason, DATA_QUALITY_UNWORN)
            gap_counter += 1
            in_gap       = True
            consec       = 0
            worn_loose   = 0.0 if gap_reason != 'loose' else float(wear)

            rows.append((
                stat_date_ts,
                local_date, 'UTC',
                0,           # scratch_count
                0,           # scratch_duration
                scratch_avg_dur_default,  # scratch_avg_dur
                scratch_avg_dur_default * 2,  # scratch_max_dur
                0,           # night_scratch_count
                0.0,         # wpeb_score
                None,        # avg_humidity
                safe(mean), safe(std),
                temp_coef,   # temp_coef
                0.0,         # temp_effect
                None, None,  # zscore, avg_zscore
                consec,
                get_phase(valid_days),
                None, None, 1.5,  # threshold_z, threshold_consec, threshold_avgz
                valid_days,
                0, 0, None,  # is_abnormal, alert_level, alert_reason
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # s1-s6, total_score
                0,           # health_level
                dq,
                wear, worn_loose,
                now, now,
            ))
            continue

        # ── 恢复佩戴 ─────────────────────────────────────────
        if in_gap:
            in_gap        = False
            just_resumed  = True
            if gap_counter >= GAP_RESET:
                valid_days = 0
            gap_counter = 0
            consec      = 0
        else:
            just_resumed = False

        count = scratch_count_for_day(i, phases, temp, temp_coef)

        # ── 热身期 ────────────────────────────────────────────
        if i < warmup:
            buf_c.append(count)
            buf_t.append(temp)
            rows.append((
                stat_date_ts,
                local_date, 'UTC',
                count,
                count * scratch_avg_dur_default,
                scratch_avg_dur_default,
                scratch_avg_dur_default * 2,
                int(count * 0.3),
                0.0,
                None,
                None, None,
                temp_coef,
                0.0,
                None, None,
                0,
                0,
                None, None, 1.5,
                0,   # valid_days=0 during warmup
                0, 0, None,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0,
                DATA_QUALITY_NORMAL,
                wear, 0.0,
                now, now,
            ))
            continue

        # ── 初始化基线 ────────────────────────────────────────
        if mean is None:
            mean = float(np.mean(buf_c))
            std  = max(float(np.std(buf_c)) if len(buf_c) > 1 else MIN_STD, MIN_STD)

        valid_days += 1
        tz, tc, ta  = get_thresholds(valid_days)

        # ── 缓冲天（恢复佩戴后第一天） ───────────────────────
        if just_resumed:
            mean = mean * (1 - NORMAL_W) + count * NORMAL_W
            buf_c.append(count)
            buf_t.append(temp)
            scratch_duration = count * scratch_avg_dur_default
            night_sc = int(count * 0.3)
            rows.append((
                stat_date_ts,
                local_date, 'UTC',
                count,
                scratch_duration,
                scratch_avg_dur_default,
                scratch_avg_dur_default * 2,
                night_sc,
                0.0,
                None,
                round(mean, 2), round(std, 2),
                temp_coef,
                0.0,
                None, None,
                0,
                get_phase(valid_days),
                safe(tz), safe(tc), 1.5,
                valid_days,
                0, 0, None,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0,
                DATA_QUALITY_BUFFER,
                wear, 0.0,
                now, now,
            ))
            continue

        # ── 正常评估 ──────────────────────────────────────────
        coef        = estimate_temp_coef(buf_c, buf_t)
        zscore      = round(((count - mean) - coef * (temp - 20)) / std, 2)
        is_abn      = bool(zscore > tz) if tz is not None else False

        if is_abn:
            consec += 1
            mean = mean * (1 - ABNORM_W) + count * ABNORM_W
        else:
            consec = 0
            mean = mean * (1 - NORMAL_W) + count * NORMAL_W
            buf_c.append(count)
            buf_t.append(temp)

        if len(buf_c) > 1:
            std = max(float(np.std(buf_c[-30:])), MIN_STD)

        n_back   = max((tc - 1) if tc else 2, 1)
        z_window = [z for z in recent_z_buf[-n_back:]] + [zscore]
        avg_z    = round(float(np.mean(z_window)), 2)

        recent_z_buf.append(zscore)
        if len(recent_z_buf) > 10:
            recent_z_buf.pop(0)

        alert  = bool((tc is not None) and (consec >= tc) and (avg_z >= ta))
        reason = (f'连续{consec}天z>{tz:.1f}，均值z={avg_z:.2f}，抓挠{count}次'
                  if alert else None)

        alert_level  = 1 if is_abn else 0
        if alert:
            alert_level = 2
        health_level = 0 if not is_abn else (2 if alert else 1)

        scratch_duration = count * scratch_avg_dur_default
        night_sc = int(count * 0.3)
        wpeb_score = float(zscore) if zscore is not None else 0.0

        # s1-s6 simple distribution (equal split)
        s_unit = round(count / 6.0, 1)
        s_scores = [s_unit] * 6
        total_score = round(sum(s_scores), 1)

        gap_type = gap_map.get(i, None)
        worn_loose = float(wear) if gap_type == 'loose' else 0.0

        rows.append((
            stat_date_ts,
            local_date, 'UTC',
            count,
            scratch_duration,
            scratch_avg_dur_default,
            scratch_avg_dur_default * 2,
            night_sc,
            wpeb_score,
            None,           # avg_humidity
            round(mean, 2), round(std, 2),
            coef,
            0.0,            # temp_effect
            zscore, avg_z,
            consec,
            get_phase(valid_days),
            safe(tz), safe(tc), 1.5,
            valid_days,
            int(is_abn), alert_level, reason,
            s_scores[0], s_scores[1], s_scores[2],
            s_scores[3], s_scores[4], s_scores[5],
            total_score,
            health_level,
            DATA_QUALITY_NORMAL,
            wear, worn_loose,
            now, now,
        ))

    return rows


# ══════════════════════════════════════════════════════
#  数据库操作
# ══════════════════════════════════════════════════════

def get_conn(database=None):
    kwargs = dict(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        cursorclass=pymysql.cursors.DictCursor,
        charset="utf8mb4",
    )
    if database:
        kwargs['database'] = database
    return pymysql.connect(**kwargs)


def create_schema():
    conn = get_conn(database=None)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{SKIN_SCHEMA}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[OK] 数据库 {SKIN_SCHEMA} 已就绪")


def create_table(conn, device_id: int):
    t = tbl(device_id)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {t} (
          stat_date_ts      BIGINT        NOT NULL,
          local_date        VARCHAR(12)   DEFAULT NULL,
          user_timezone     VARCHAR(32)   DEFAULT NULL,
          scratch_count     INT           NOT NULL DEFAULT 0,
          scratch_duration  BIGINT        NOT NULL DEFAULT 0,
          scratch_avg_dur   INT           NOT NULL DEFAULT 0,
          scratch_max_dur   INT           NOT NULL DEFAULT 0,
          night_scratch_count INT         NOT NULL DEFAULT 0,
          wpeb_score        DECIMAL(10,4) NOT NULL DEFAULT 0,
          avg_humidity      DECIMAL(5,1)  DEFAULT NULL,
          baseline_mean     DECIMAL(6,2)  DEFAULT NULL,
          baseline_std      DECIMAL(6,2)  DEFAULT NULL,
          temp_coef         DECIMAL(5,3)  DEFAULT NULL,
          temp_effect       DECIMAL(6,2)  DEFAULT NULL,
          zscore            DECIMAL(6,2)  DEFAULT NULL,
          avg_zscore        DECIMAL(6,2)  DEFAULT NULL,
          consec_abnormal   INT           NOT NULL DEFAULT 0,
          eval_phase        SMALLINT      NOT NULL DEFAULT 0,
          threshold_z       DECIMAL(4,2)  DEFAULT NULL,
          threshold_consec  SMALLINT      DEFAULT NULL,
          threshold_avgz    DECIMAL(4,2)  DEFAULT NULL,
          valid_days        INT           NOT NULL DEFAULT 0,
          is_abnormal       SMALLINT      NOT NULL DEFAULT 0,
          alert_level       SMALLINT      NOT NULL DEFAULT 0,
          alert_reason      VARCHAR(256)  DEFAULT NULL,
          s1_score          DECIMAL(5,1)  NOT NULL DEFAULT 0,
          s2_score          DECIMAL(5,1)  NOT NULL DEFAULT 0,
          s3_score          DECIMAL(5,1)  NOT NULL DEFAULT 0,
          s4_score          DECIMAL(5,1)  NOT NULL DEFAULT 0,
          s5_score          DECIMAL(5,1)  NOT NULL DEFAULT 0,
          s6_score          DECIMAL(5,1)  NOT NULL DEFAULT 0,
          total_score       DECIMAL(6,1)  NOT NULL DEFAULT 0,
          health_level      SMALLINT      NOT NULL DEFAULT 0,
          data_quality      SMALLINT      NOT NULL DEFAULT 0,
          wear_minutes      INT           NOT NULL DEFAULT 0,
          worn_loose_minutes DECIMAL(8,1) NOT NULL DEFAULT 0,
          created_at        BIGINT        NOT NULL,
          updated_at        BIGINT        NOT NULL,
          PRIMARY KEY (stat_date_ts)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    conn.commit()
    cursor.close()
    print(f"  [OK] 表 {t} 已就绪")


def insert_rows(conn, device_id: int, rows: list):
    if not rows:
        print(f"  [device_id={device_id}] 无数据，跳过")
        return

    t   = tbl(device_id)
    sql = f"""
        INSERT IGNORE INTO {t}
          (stat_date_ts, local_date, user_timezone,
           scratch_count, scratch_duration, scratch_avg_dur, scratch_max_dur,
           night_scratch_count, wpeb_score, avg_humidity,
           baseline_mean, baseline_std, temp_coef, temp_effect,
           zscore, avg_zscore, consec_abnormal, eval_phase,
           threshold_z, threshold_consec, threshold_avgz,
           valid_days, is_abnormal, alert_level, alert_reason,
           s1_score, s2_score, s3_score, s4_score, s5_score, s6_score,
           total_score, health_level, data_quality,
           wear_minutes, worn_loose_minutes,
           created_at, updated_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    cursor = conn.cursor()
    cursor.executemany(sql, rows)
    conn.commit()
    print(f"  [device_id={device_id}] 插入 {cursor.rowcount} 条每日评估记录")
    cursor.close()


# ══════════════════════════════════════════════════════
#  查询验证
# ══════════════════════════════════════════════════════

def query_summary():
    conn   = get_conn(database=SKIN_SCHEMA)
    cursor = conn.cursor()

    print("\n======= 各设备每日评估概况 =======")
    for sc in SCENARIOS:
        t = tbl(sc['device_id'])
        try:
            cursor.execute(f"""
                SELECT
                    COUNT(*)                                                          AS total_days,
                    SUM(CASE WHEN data_quality = 0 AND eval_phase = 0 THEN 1 ELSE 0 END) AS warmup_days,
                    SUM(CASE WHEN data_quality IN (1,2,3,4) THEN 1 ELSE 0 END)       AS gap_days,
                    SUM(CASE WHEN data_quality = 5 THEN 1 ELSE 0 END)                AS buffer_days,
                    SUM(is_abnormal)                                                  AS abnormal_days,
                    SUM(CASE WHEN alert_level >= 2 THEN 1 ELSE 0 END)                AS alert_count,
                    ROUND(AVG(CASE WHEN data_quality=0 THEN scratch_count END), 1)   AS avg_scratch,
                    ROUND(MAX(zscore), 2)                                             AS max_zscore
                FROM {t}
            """)
            row = cursor.fetchone()
            print(f"  {sc['sn']:12s} (id={sc['device_id']})  "
                  f"总={row['total_days']:3d}天  热身={row['warmup_days']}  缺口={row['gap_days']}  缓冲={row['buffer_days']}  "
                  f"异常={row['abnormal_days']}  推送={row['alert_count']}  "
                  f"日均={row['avg_scratch']}次  最高z={row['max_zscore']}")
        except Exception as e:
            print(f"  device_id={sc['device_id']}: 查询失败 {e}")

    cursor.close()
    conn.close()


# ══════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════

def main():
    print("=== 第一步：创建模式 ===")
    create_schema()

    conn = get_conn(database=SKIN_SCHEMA)

    print("\n=== 第二步：建表 ===")
    for sc in SCENARIOS:
        create_table(conn, sc['device_id'])

    print("\n=== 第三步：生成并插入每日评估数据 ===")
    for idx, sc in enumerate(SCENARIOS):
        rows = build_daily_rows(sc, seed=42 + idx)
        print(f"  [{sc['sn']}] 生成 {len(rows)} 天数据，开始插入...")
        insert_rows(conn, sc['device_id'], rows)

    conn.close()

    print("\n=== 第四步：查询验证 ===")
    query_summary()

    print("\n[完成] 皮肤健康评估数据库写入完毕！")


if __name__ == "__main__":
    main()
