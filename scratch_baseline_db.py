"""
抓挠基线数据库（MySQL）
=====================
数据库: pet_dog_scratch_baseline
表: pet_skin_baseline（单张，每行一个设备的最新基线）

每行 = 设备基线快照
  device_id      设备ID (BIGINT PK)
  baseline_mean  基线均值 次/天
  baseline_std   基线标准差
  temp_coef      温度修正系数 次/°C
  valid_days     参与计算的有效正常天数
  eval_phase     评估阶段
  confidence     基线置信度 0.00-1.00
  wpeb_mean      WPEB均值（可选）
  wpeb_std       WPEB标准差（可选）
  last_updated_ts 最后更新时间 ms
  created_at     创建时间 ms
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
BSL_SCHEMA     = "pet_dog_scratch_baseline"
BSL_TABLE      = "pet_skin_baseline"

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
#  算法函数（与 skin_assessment_db.py 一致）
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


def build_gap_map(gaps: list) -> dict:
    gm = {}
    for start, end, reason in gaps:
        for d in range(start, min(end, DAYS)):
            gm[d] = reason
    return gm


def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def tbl() -> str:
    return f"`{BSL_SCHEMA}`.`{BSL_TABLE}`"


# ══════════════════════════════════════════════════════
#  核心：运行基线算法，得出最终基线状态
# ══════════════════════════════════════════════════════

def build_final_baseline(sc: dict, seed: int = 42) -> dict:
    """Run the baseline algorithm and return the final state as a dict."""
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
    last_ts     = None

    for i in range(DAYS):
        d    = START_DATE + timedelta(days=i)
        temp = round(float(_temperature[i]), 1)

        if i in gap_map:
            gap_counter += 1
            in_gap       = True
            consec       = 0
            continue

        last_ts = to_ts(d)

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

        if i < warmup:
            buf_c.append(count)
            buf_t.append(temp)
            continue

        if mean is None:
            mean = float(np.mean(buf_c))
            std  = max(float(np.std(buf_c)) if len(buf_c) > 1 else MIN_STD, MIN_STD)

        valid_days += 1
        tz, tc, ta = get_thresholds(valid_days)

        if just_resumed:
            mean = mean * (1 - NORMAL_W) + count * NORMAL_W
            buf_c.append(count)
            buf_t.append(temp)
            continue

        coef   = estimate_temp_coef(buf_c, buf_t)
        zscore = ((count - mean) - coef * (temp - 20)) / std
        is_abn = bool(zscore > tz) if tz is not None else False

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

    final_coef = estimate_temp_coef(buf_c, buf_t)
    confidence = round(min(1.0, valid_days / 30), 2)
    phase = get_phase(valid_days)

    return {
        'device_id':    sc['device_id'],
        'mean':         round(mean, 2) if mean is not None else 0.0,
        'std':          round(std, 2) if std is not None else MIN_STD,
        'temp_coef':    final_coef,
        'valid_days':   valid_days,
        'eval_phase':   phase,
        'confidence':   confidence,
        'last_ts':      last_ts,
    }


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
    conn   = get_conn(database=None)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{BSL_SCHEMA}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[OK] 数据库 {BSL_SCHEMA} 已就绪")


def create_table(conn):
    t = tbl()
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {t} (
          device_id       BIGINT        NOT NULL,
          baseline_mean   DECIMAL(6,2)  NOT NULL DEFAULT 0,
          baseline_std    DECIMAL(6,2)  NOT NULL DEFAULT 0,
          temp_coef       DECIMAL(5,3)  NOT NULL DEFAULT 0,
          valid_days      INT           NOT NULL DEFAULT 0,
          eval_phase      SMALLINT      NOT NULL DEFAULT 0,
          confidence      DECIMAL(4,2)  NOT NULL DEFAULT 0,
          wpeb_mean       DECIMAL(10,4) DEFAULT NULL,
          wpeb_std        DECIMAL(10,4) DEFAULT NULL,
          last_updated_ts BIGINT        DEFAULT NULL,
          created_at      BIGINT        NOT NULL,
          PRIMARY KEY (device_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    conn.commit()
    cursor.close()
    print(f"  [OK] 表 {t} 已就绪")


def insert_baselines(conn, baselines: list):
    if not baselines:
        print("  无数据，跳过")
        return

    t   = tbl()
    now = now_ms()
    sql = f"""
        INSERT INTO {t}
          (device_id, baseline_mean, baseline_std, temp_coef,
           valid_days, eval_phase, confidence,
           wpeb_mean, wpeb_std, last_updated_ts, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
          baseline_mean=VALUES(baseline_mean),
          baseline_std=VALUES(baseline_std),
          temp_coef=VALUES(temp_coef),
          valid_days=VALUES(valid_days),
          eval_phase=VALUES(eval_phase),
          confidence=VALUES(confidence),
          last_updated_ts=VALUES(last_updated_ts)
    """
    rows = [
        (b['device_id'], b['mean'], b['std'], b['temp_coef'],
         b['valid_days'], b['eval_phase'], b['confidence'],
         None, None, b['last_ts'], now)
        for b in baselines
    ]
    cursor = conn.cursor()
    cursor.executemany(sql, rows)
    conn.commit()
    print(f"  插入/更新 {cursor.rowcount} 条基线记录")
    cursor.close()


# ══════════════════════════════════════════════════════
#  查询验证
# ══════════════════════════════════════════════════════

def query_summary():
    conn   = get_conn(database=BSL_SCHEMA)
    cursor = conn.cursor()

    print("\n======= 基线快照概况 =======")
    t = tbl()
    try:
        cursor.execute(f"""
            SELECT
                COUNT(*)                        AS total,
                MAX(valid_days)                 AS max_valid_days,
                ROUND(MAX(confidence), 2)       AS max_conf,
                ROUND(AVG(baseline_mean), 2)    AS avg_mean,
                ROUND(AVG(baseline_std), 2)     AS avg_std,
                ROUND(AVG(temp_coef), 3)        AS avg_coef
            FROM {t}
        """)
        row = cursor.fetchone()
        print(f"  总设备={int(row['total'])}  最终有效天={row['max_valid_days']}  置信度={row['max_conf']}  "
              f"均值={row['avg_mean']}  标准差={row['avg_std']}  温度系数={row['avg_coef']}")

        cursor.execute(f"SELECT device_id, baseline_mean, baseline_std, temp_coef, valid_days, confidence FROM {t} ORDER BY device_id")
        rows = cursor.fetchall()
        for r in rows:
            print(f"  device_id={r['device_id']}  mean={r['baseline_mean']}  std={r['baseline_std']}  "
                  f"tc={r['temp_coef']}  valid_days={r['valid_days']}  conf={r['confidence']}")
    except Exception as e:
        print(f"  查询失败: {e}")

    cursor.close()
    conn.close()


# ══════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════

def main():
    print("=== 第一步：创建模式 ===")
    create_schema()

    conn = get_conn(database=BSL_SCHEMA)

    print("\n=== 第二步：建表 ===")
    create_table(conn)

    print("\n=== 第三步：计算基线并插入 ===")
    baselines = []
    for idx, sc in enumerate(SCENARIOS):
        b = build_final_baseline(sc, seed=42 + idx)
        print(f"  [{sc['sn']}] mean={b['mean']}  std={b['std']}  tc={b['temp_coef']}  valid_days={b['valid_days']}")
        baselines.append(b)

    insert_baselines(conn, baselines)
    conn.close()

    print("\n=== 第四步：查询验证 ===")
    query_summary()

    print("\n[完成] 抓挠基线数据库写入完毕！")


if __name__ == "__main__":
    main()
