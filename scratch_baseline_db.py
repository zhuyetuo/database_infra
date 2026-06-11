"""
抓挠基线数据库（MySQL）
=====================
数据库: pet_dog_scratch_baseline
每个设备独立一张表: {device_id}

每行 = 当天算法运行后的基线快照
  stat_date      统计日期
  baseline_mean  基线均值 次/天
  baseline_std   基线标准差
  temp_coef      温度修正系数 次/°C
  confidence     基线置信度 0.00-1.00
  valid_days     参与计算的有效正常天数
"""

import pymysql
import pymysql.cursors
import numpy as np
import math
from datetime import date, timedelta, datetime, timezone

# ══════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════
MYSQL_HOST     = "127.0.0.1"
MYSQL_PORT     = 3306
MYSQL_USER     = "appuser"
MYSQL_PASSWORD = "123456"
BSL_SCHEMA     = "pet_dog_scratch_baseline"

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
        'sn':     'device_id_1',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
    },
    {
        'sn':     'device_id_2',
        'phases': [(0, 60, 10.0, 2.0), (60, 80, 30.0, 4.0), (80, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   (60, 80),
    },
    {
        'sn':     'device_id_3',
        'phases': [(0, 60, 10.0, 2.0), (60, 180, 28.0, 4.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   (60, 180),
    },
    {
        'sn':          'device_id_4',
        'phases':      [(0, 40, 10.0, 2.0), (40, 55, 28.0, 4.0),
                        (55, 120, 10.0, 2.0), (120, 135, 30.0, 4.0),
                        (135, 180, 10.0, 2.0)],
        'tc':          0.10,
        'gaps':        [],
        'sick':        None,
        'sick_episodes': [(40, 55), (120, 135)],
    },
    {
        'sn':     'device_id_5',
        'phases': [(0, 60, 10.0, 2.0), (60, 120, 15.0, 2.0), (120, 180, 22.0, 3.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
    },
    {
        'sn':     'device_id_6',
        'phases': [(0, 90, 10.0, 2.0), (90, 180, 25.0, 3.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   (90, 180),
    },
    {
        'sn':     'device_id_7',
        'phases': [(0, 50, 10.0, 2.0), (50, 80, 45.0, 6.0), (80, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   (50, 80),
    },
    {
        'sn':     'device_id_8',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.35,
        'gaps':   [],
        'sick':   None,
    },
    {
        'sn':     'device_id_9',
        'phases': [(0, 30, 10.0, 2.0), (30, 90, 3.0, 1.0), (90, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
    },
    # 设备/数据质量场景 10-16
    {
        'sn':     'device_id_10',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(35, 38, 'unworn')],
        'sick':   None,
    },
    {
        'sn':     'device_id_11',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(40, 45, 'battery')],
        'sick':   None,
    },
    {
        'sn':     'device_id_12',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(30, 65, 'battery')],
        'sick':   None,
    },
    {
        'sn':     'device_id_13',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(d, d + 1, 'signal') for d in sorted(_signal_gap_days)],
        'sick':   None,
    },
    {
        'sn':     'device_id_14',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(50, 58, 'loose')],
        'sick':   None,
    },
    {
        'sn':     'device_id_15',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(88, 92, 'battery')],
        'sick':   None,
    },
    {
        'sn':     'device_id_16',
        'phases': [(0, 70, 10.0, 2.0), (70, 90, 35.0, 5.0), (90, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
        'drift_range': (70, 90),
    },
    # 环境场景 17-20
    {
        'sn':     'device_id_17',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.30,
        'gaps':   [],
        'sick':   None,
    },
    {
        'sn':     'device_id_18',
        'phases': [(0, 60, 10.0, 2.0), (60, 180, 13.0, 2.0)],
        'tc':     0.15,
        'gaps':   [],
        'sick':   None,
        'temp_shift': (60, 5.0),
    },
    {
        'sn':     'device_id_19',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(80, 90, 'unworn')],
        'sick':   None,
    },
    {
        'sn':     'device_id_20',
        'phases': [(0, 180, 14.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
    },
    # 个体类型场景 21-24
    {
        'sn':     'device_id_21',
        'phases': [(0, 180, 15.0, 4.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
        'warmup': 7,
    },
    {
        'sn':     'device_id_22',
        'phases': [(0, 180, 5.0, 1.0)],
        'tc':     0.05,
        'gaps':   [],
        'sick':   None,
    },
    {
        'sn':     'device_id_23',
        'phases': [(0, 180, 20.0, 3.0)],
        'tc':     0.12,
        'gaps':   [],
        'sick':   None,
    },
    {
        'sn':     'device_id_24',
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


def to_date(d: date) -> str:
    return d.strftime('%Y-%m-%d')


def tbl(sn: str) -> str:
    return f"`{BSL_SCHEMA}`.`{sn.lower()}`"


# ══════════════════════════════════════════════════════
#  核心：运行基线算法，构建每日快照记录
# ══════════════════════════════════════════════════════

def build_baseline_rows(sc: dict, seed: int = 42) -> list:
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

    rows = []

    for i in range(DAYS):
        d         = START_DATE + timedelta(days=i)
        stat_date = to_date(d)
        temp      = round(float(_temperature[i]), 1)

        # ── 缺口天：基线冻结，不保存快照 ─────────────────────
        if i in gap_map:
            gap_counter += 1
            in_gap       = True
            consec       = 0
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
            continue

        # ── 初始化基线 ────────────────────────────────────────
        if mean is None:
            mean = float(np.mean(buf_c))
            std  = max(float(np.std(buf_c)) if len(buf_c) > 1 else MIN_STD, MIN_STD)

        valid_days += 1
        tz, tc, ta = get_thresholds(valid_days)

        # ── 缓冲天 ────────────────────────────────────────────
        if just_resumed:
            mean = mean * (1 - NORMAL_W) + count * NORMAL_W
            buf_c.append(count)
            buf_t.append(temp)
            coef       = estimate_temp_coef(buf_c, buf_t)
            confidence = round(min(1.0, valid_days / 30), 2)
            rows.append((
                stat_date,
                round(mean, 2),
                round(std, 2),
                coef,
                confidence,
                valid_days,
            ))
            continue

        # ── 正常评估 ──────────────────────────────────────────
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

        confidence = round(min(1.0, valid_days / 30), 2)
        rows.append((
            stat_date,
            round(mean, 2),
            round(std, 2),
            coef,
            confidence,
            valid_days,
        ))

    return rows


# ══════════════════════════════════════════════════════
#  数据库操作
# ══════════════════════════════════════════════════════

def get_conn():
    return pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, database=BSL_SCHEMA,
        cursorclass=pymysql.cursors.DictCursor,
        charset="utf8mb4",
    )


def create_schema():
    # 连接不指定 database，先建库
    conn   = pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{BSL_SCHEMA}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[OK] 数据库 {BSL_SCHEMA} 已就绪")


def create_table(conn, sn: str):
    t = tbl(sn)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {t} (
          stat_date     DATE          NOT NULL,
          baseline_mean DECIMAL(6,2)  NOT NULL,
          baseline_std  DECIMAL(6,2)  NOT NULL,
          temp_coef     DECIMAL(5,3)  NOT NULL DEFAULT 0.000,
          confidence    DECIMAL(4,2)  NOT NULL DEFAULT 0.00,
          valid_days    INT           NOT NULL DEFAULT 0,
          PRIMARY KEY (stat_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    conn.commit()
    cursor.close()
    print(f"  [OK] 表 {t} 已就绪")


def insert_rows(conn, sn: str, rows: list):
    if not rows:
        print(f"  [{sn}] 无数据，跳过")
        return

    t   = tbl(sn)
    sql = f"""
        INSERT IGNORE INTO {t}
          (stat_date, baseline_mean, baseline_std,
           temp_coef, confidence, valid_days)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor = conn.cursor()
    cursor.executemany(sql, rows)
    conn.commit()
    print(f"  [{sn}] 插入 {cursor.rowcount} 条基线快照记录")
    cursor.close()


# ══════════════════════════════════════════════════════
#  查询验证
# ══════════════════════════════════════════════════════

def query_summary():
    conn   = get_conn()
    cursor = conn.cursor()

    print("\n======= 基线快照概况 =======")
    for sc in SCENARIOS:
        t = tbl(sc['sn'])
        try:
            cursor.execute(f"""
                SELECT
                    COUNT(*)                        AS total,
                    MAX(valid_days)                 AS max_valid_days,
                    ROUND(MAX(confidence), 2)    AS max_conf,
                    ROUND(AVG(baseline_mean), 2) AS avg_mean,
                    ROUND(AVG(baseline_std), 2)  AS avg_std,
                    ROUND(AVG(temp_coef), 3)     AS avg_coef,
                    MIN(stat_date)                  AS earliest,
                    MAX(stat_date)                  AS latest
                FROM {t}
            """)
            row = cursor.fetchone()
            print(f"  {sc['sn']:20s}  快照={int(row['total']):3d}天  "
                  f"最终有效天={row['max_valid_days']}  置信度={row['max_conf']}  "
                  f"均值={row['avg_mean']}  标准差={row['avg_std']}  温度系数={row['avg_coef']}  "
                  f"{row['earliest']} ~ {row['latest']}")
        except Exception as e:
            print(f"  {sc['sn']}: 查询失败 {e}")

    cursor.close()
    conn.close()


# ══════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════

def main():
    print("=== 第一步：创建模式 ===")
    create_schema()

    conn = get_conn()

    print("\n=== 第二步：建表 ===")
    for sc in SCENARIOS:
        create_table(conn, sc['sn'])

    print("\n=== 第三步：生成并插入基线快照数据 ===")
    for idx, sc in enumerate(SCENARIOS):
        rows = build_baseline_rows(sc, seed=42 + idx)
        print(f"  [{sc['sn']}] 生成 {len(rows)} 条快照，开始插入...")
        insert_rows(conn, sc['sn'], rows)

    conn.close()

    print("\n=== 第四步：查询验证 ===")
    query_summary()

    print("\n[完成] 抓挠基线数据库写入完毕！")


if __name__ == "__main__":
    main()
