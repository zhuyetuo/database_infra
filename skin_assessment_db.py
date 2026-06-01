"""
皮肤健康评估数据库（PostgreSQL）
=====================
数据库: pet_collar，模式: pet_dog_skin_assessment
每个设备独立一张每日评估表: {device_sn}

24 个场景设备（与 imu_raw_db.py 一一对应）

评估算法:
  - 热身期（默认3天，device_sn_21 为7天）：只收集，不评估
  - 个体动态基线：异常天权重 0.01，正常天权重 0.05
  - 标准差保底 2.0
  - 温度修正：20 天数据后启用，系数上限 0.4
  - 动态阈值：早期 z>4.0 连续5天，过渡期 z>3.5 连续4天，稳定期 z>2.5 连续3天
  - 缺口处理：基线冻结 → 恢复后缓冲天（data_quality=5）→ 缺口≥30天重置门槛
"""

import psycopg2
import numpy as np
import math
from datetime import date, timedelta, datetime, timezone

# ══════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════
PG_HOST     = "127.0.0.1"
PG_PORT     = 5432
PG_USER     = "postgres"
PG_PASSWORD = "123456"
PG_DB       = "pet_collar"
SKIN_SCHEMA = "pet_dog_skin_assessment"

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
        'sn':     'device_sn_1',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
    },
    {
        'sn':     'device_sn_2',
        'phases': [(0, 60, 10.0, 2.0), (60, 80, 30.0, 4.0), (80, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   (60, 80),
    },
    {
        'sn':     'device_sn_3',
        'phases': [(0, 60, 10.0, 2.0), (60, 180, 28.0, 4.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   (60, 180),
    },
    {
        'sn':          'device_sn_4',
        'phases':      [(0, 40, 10.0, 2.0), (40, 55, 28.0, 4.0),
                        (55, 120, 10.0, 2.0), (120, 135, 30.0, 4.0),
                        (135, 180, 10.0, 2.0)],
        'tc':          0.10,
        'gaps':        [],
        'sick':        None,
        'sick_episodes': [(40, 55), (120, 135)],
    },
    {
        'sn':     'device_sn_5',
        'phases': [(0, 60, 10.0, 2.0), (60, 120, 15.0, 2.0), (120, 180, 22.0, 3.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
    },
    {
        'sn':     'device_sn_6',
        'phases': [(0, 90, 10.0, 2.0), (90, 180, 25.0, 3.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   (90, 180),
    },
    {
        'sn':     'device_sn_7',
        'phases': [(0, 50, 10.0, 2.0), (50, 80, 45.0, 6.0), (80, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   (50, 80),
    },
    {
        'sn':     'device_sn_8',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.35,
        'gaps':   [],
        'sick':   None,
    },
    {
        'sn':     'device_sn_9',
        'phases': [(0, 30, 10.0, 2.0), (30, 90, 3.0, 1.0), (90, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
    },
    # 设备/数据质量场景 10-16
    {
        'sn':     'device_sn_10',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(35, 38, 'unworn')],
        'sick':   None,
    },
    {
        'sn':     'device_sn_11',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(40, 45, 'battery')],
        'sick':   None,
    },
    {
        'sn':     'device_sn_12',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(30, 65, 'battery')],
        'sick':   None,
    },
    {
        'sn':     'device_sn_13',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(d, d + 1, 'signal') for d in sorted(_signal_gap_days)],
        'sick':   None,
    },
    {
        'sn':     'device_sn_14',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(50, 58, 'loose')],
        'sick':   None,
    },
    {
        'sn':     'device_sn_15',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(88, 92, 'battery')],
        'sick':   None,
    },
    {
        'sn':     'device_sn_16',
        'phases': [(0, 70, 10.0, 2.0), (70, 90, 35.0, 5.0), (90, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
        'drift_range': (70, 90),
    },
    # 环境场景 17-20
    {
        'sn':     'device_sn_17',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.30,
        'gaps':   [],
        'sick':   None,
    },
    {
        'sn':     'device_sn_18',
        'phases': [(0, 60, 10.0, 2.0), (60, 180, 13.0, 2.0)],
        'tc':     0.15,
        'gaps':   [],
        'sick':   None,
        'temp_shift': (60, 5.0),
    },
    {
        'sn':     'device_sn_19',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(80, 90, 'unworn')],
        'sick':   None,
    },
    {
        'sn':     'device_sn_20',
        'phases': [(0, 180, 14.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
    },
    # 个体类型场景 21-24
    {
        'sn':     'device_sn_21',
        'phases': [(0, 180, 15.0, 4.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
        'warmup': 7,
    },
    {
        'sn':     'device_sn_22',
        'phases': [(0, 180, 5.0, 1.0)],
        'tc':     0.05,
        'gaps':   [],
        'sick':   None,
    },
    {
        'sn':     'device_sn_23',
        'phases': [(0, 180, 20.0, 3.0)],
        'tc':     0.12,
        'gaps':   [],
        'sick':   None,
    },
    {
        'sn':     'device_sn_24',
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


def to_date(d: date) -> str:
    return d.strftime('%Y-%m-%d')


def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def tbl(sn: str) -> str:
    return f"{SKIN_SCHEMA}.{sn.lower()}"


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

    for i in range(DAYS):
        d    = START_DATE + timedelta(days=i)
        stat_date = to_date(d)
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

            rows.append((
                stat_date,
                0,
                safe(mean), safe(std),
                None, None,
                0,
                get_phase(valid_days),
                None, None,
                0, 0, None,
                dq, 0,
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
                stat_date,
                count,
                None, None,
                None, None,
                0,
                0,
                None, None,
                0, 0, None,
                DATA_QUALITY_NORMAL, wear,
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
            rows.append((
                stat_date,
                count,
                round(mean, 2), round(std, 2),
                None, None,
                0,
                get_phase(valid_days),
                safe(tz), safe(tc),
                0, 0, None,
                DATA_QUALITY_BUFFER, wear,
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

        rows.append((
            stat_date,
            count,
            round(mean, 2), round(std, 2),
            zscore, avg_z,
            consec,
            get_phase(valid_days),
            safe(tz), safe(tc),
            int(is_abn), int(alert), reason,
            DATA_QUALITY_NORMAL, wear,
        ))

    return rows


# ══════════════════════════════════════════════════════
#  数据库操作
# ══════════════════════════════════════════════════════

def get_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER,
        password=PG_PASSWORD, dbname=PG_DB
    )


def create_schema():
    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {SKIN_SCHEMA}")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[OK] 模式 {SKIN_SCHEMA} 已就绪")


def create_table(conn, sn: str):
    t = tbl(sn)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {t} (
          stat_date        date          NOT NULL,
          scratch_count    int           NOT NULL DEFAULT 0,
          baseline_mean    decimal(6,2)  DEFAULT NULL,
          baseline_std     decimal(6,2)  DEFAULT NULL,
          zscore           decimal(6,2)  DEFAULT NULL,
          avg_zscore       decimal(6,2)  DEFAULT NULL,
          consec_abnormal  int           NOT NULL DEFAULT 0,
          eval_phase       SMALLINT      NOT NULL DEFAULT 0,
          threshold_z      decimal(4,2)  DEFAULT NULL,
          threshold_consec SMALLINT      DEFAULT NULL,
          is_abnormal      SMALLINT      NOT NULL DEFAULT 0,
          alert_triggered  SMALLINT      NOT NULL DEFAULT 0,
          alert_reason     VARCHAR(256)  DEFAULT NULL,
          data_quality     SMALLINT      NOT NULL DEFAULT 0,
          wear_minutes     int           NOT NULL DEFAULT 0,
          PRIMARY KEY (stat_date)
        )
    """)
    cursor.execute(f"CREATE INDEX IF NOT EXISTS {sn}_idx_abn ON {t} (is_abnormal, stat_date)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS {sn}_idx_alt ON {t} (alert_triggered, stat_date)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS {sn}_idx_dq  ON {t} (data_quality, stat_date)")
    conn.commit()
    cursor.close()
    print(f"  [OK] 表 {t} 已就绪")


def insert_rows(conn, sn: str, rows: list):
    if not rows:
        print(f"  [{sn}] 无数据，跳过")
        return

    t   = tbl(sn)
    sql = f"""
        INSERT INTO {t}
          (stat_date, scratch_count,
           baseline_mean, baseline_std,
           zscore, avg_zscore,
           consec_abnormal, eval_phase,
           threshold_z, threshold_consec,
           is_abnormal, alert_triggered, alert_reason,
           data_quality, wear_minutes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (stat_date) DO NOTHING
    """
    cursor = conn.cursor()
    cursor.executemany(sql, rows)
    conn.commit()
    print(f"  [{sn}] 插入 {cursor.rowcount} 条每日评估记录")
    cursor.close()


# ══════════════════════════════════════════════════════
#  查询验证
# ══════════════════════════════════════════════════════

def query_summary():
    conn   = get_conn()
    cursor = conn.cursor()

    print("\n======= 各设备每日评估概况 =======")
    for sc in SCENARIOS:
        t = tbl(sc['sn'])
        try:
            cursor.execute(f"""
                SELECT
                    COUNT(*)                                                     AS 总天数,
                    SUM(CASE WHEN data_quality = 0 AND eval_phase = 0 THEN 1 ELSE 0 END) AS 热身期,
                    SUM(CASE WHEN data_quality IN (1,2,3,4) THEN 1 ELSE 0 END)  AS 缺口天,
                    SUM(CASE WHEN data_quality = 5 THEN 1 ELSE 0 END)           AS 缓冲天,
                    SUM(is_abnormal)                                             AS 异常天,
                    SUM(alert_triggered)                                         AS 推送次数,
                    ROUND(AVG(CASE WHEN data_quality=0 THEN scratch_count END)::numeric, 1) AS 日均抓挠,
                    ROUND(MAX(zscore)::numeric, 2)                               AS 最高z_score
                FROM {t}
            """)
            row = cursor.fetchone()
            print(f"  {sc['sn']:20s}  "
                  f"总={row[0]:3d}天  热身={row[1]}  缺口={row[2]}  缓冲={row[3]}  "
                  f"异常={row[4]}  推送={row[5]}  "
                  f"日均={row[6]}次  最高z={row[7]}")
        except Exception as e:
            print(f"  {sc['sn']}: 查询失败 {e}")

    print("\n======= 推送记录明细 =======")
    found = False
    for sc in SCENARIOS:
        t = tbl(sc['sn'])
        try:
            cursor.execute(f"""
                SELECT
                    stat_date, scratch_count, zscore, avg_zscore,
                    consec_abnormal, alert_reason
                FROM {t}
                WHERE alert_triggered = 1
                ORDER BY stat_date
            """)
            alerts = cursor.fetchall()
            if alerts:
                found = True
                for r in alerts:
                    print(f"  {sc['sn']:20s}  日期={r[0]}  "
                          f"次数={r[1]}  z={r[2]}  avgz={r[3]}  "
                          f"连续={r[4]}天  {r[5]}")
        except Exception:
            pass
    if not found:
        print("  （暂无推送记录）")

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

    print("\n=== 第三步：生成并插入每日评估数据 ===")
    for idx, sc in enumerate(SCENARIOS):
        rows = build_daily_rows(sc, seed=42 + idx)
        print(f"  [{sc['sn']}] 生成 {len(rows)} 天数据，开始插入...")
        insert_rows(conn, sc['sn'], rows)

    conn.close()

    print("\n=== 第四步：查询验证 ===")
    query_summary()

    print("\n[完成] 皮肤健康评估数据库写入完毕！")


if __name__ == "__main__":
    main()
