"""
皮肤健康评估数据库
=====================
数据库: pet_skin_health
每个设备独立一张每日评估表: skin_daily_{device_sn}
共享基线表: skin_baseline（所有设备，按 device_sn 区分）

五个场景，8 个设备（与 imu_raw_db.py 一一对应）:
  A : DEV_A_NORMAL    — 完全正常
  B : DEV_B_SICK      — 短期皮肤病后康复
  C : DEV_C_SEASON    — 季节性正常升高（温度修正后不误报）
  D : DEV_D_ALLERGY   — 持续缓慢升高（基线跟上后不误报）
  E1: DEV_E1_UNWORN   — 忘记佩戴（3天缺口）
  E2: DEV_E2_BATTERY  — 没电（5天缺口）+ 缺口后皮肤病
  E3: DEV_E3_SIGNAL   — 信号不稳定（断续丢失）
  E4: DEV_E4_LOOSE    — 项圈松动（8天无效数据）

评估算法与 demo_all.py 完全一致:
  - 热身期 3 天：只收集，不评估，不推送
  - 个体动态基线：异常天权重 0.01，正常天权重 0.05
  - 标准差保底 2.0
  - 温度修正：20 天数据后启用，系数上限 0.4
  - 动态阈值：早期 z>4.0 连续5天，过渡期 z>3.5 连续4天，稳定期 z>2.5 连续3天
  - 缺口处理：基线冻结 → 恢复后缓冲天（data_quality=5）→ 缺口≥30天重置门槛
"""

import mysql.connector
import numpy as np
import math
from datetime import date, timedelta, datetime, timezone

# ══════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════
DB_HOST     = "127.0.0.1"
DB_PORT     = 3306
DB_USER     = "root"
DB_PASSWORD = "123456"
SKIN_DB     = "pet_skin_health"

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
#  全局时间序列（与 demo_all.py 保持一致）
# ══════════════════════════════════════════════════════
np.random.seed(42)

_temperature = (22 + 13 * np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, DAYS))
                + np.random.normal(0, 1.5, DAYS))
_humidity    = (65 + 15 * np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, DAYS))
                + np.random.normal(0, 3.0, DAYS))

# 信号丢失缺口（与 demo_all.py 相同随机种子）
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
#  场景定义（与 imu_raw_db.py 完全一致）
# ══════════════════════════════════════════════════════
SCENARIOS = [
    {
        'sn':     'DEV_A_NORMAL',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
    },
    {
        'sn':     'DEV_B_SICK',
        'phases': [(0, 60, 10.0, 2.0), (60, 80, 22.0, 3.0), (80, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   (60, 80),
    },
    {
        'sn':     'DEV_C_SEASON',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.25,
        'gaps':   [],
        'sick':   None,
    },
    {
        'sn':     'DEV_D_ALLERGY',
        'phases': [(0, 60, 10.0, 2.0), (60, 120, 13.0, 2.0), (120, 180, 15.0, 2.0)],
        'tc':     0.10,
        'gaps':   [],
        'sick':   None,
    },
    {
        'sn':     'DEV_E1_UNWORN',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(35, 38, 'unworn')],
        'sick':   None,
    },
    {
        'sn':     'DEV_E2_BATTERY',
        'phases': [(0, 60, 10.0, 2.0), (60, 73, 22.0, 3.0), (73, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(40, 45, 'battery')],
        'sick':   (60, 73),
    },
    {
        'sn':     'DEV_E3_SIGNAL',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(d, d + 1, 'signal') for d in sorted(_signal_gap_days)],
        'sick':   None,
    },
    {
        'sn':     'DEV_E4_LOOSE',
        'phases': [(0, 180, 10.0, 2.0)],
        'tc':     0.10,
        'gaps':   [(50, 58, 'loose')],
        'sick':   None,
    },
]


# ══════════════════════════════════════════════════════
#  算法函数（与 demo_all.py 一致）
# ══════════════════════════════════════════════════════

def get_thresholds(valid_days: int):
    if valid_days < 1:      return None, None, None
    elif valid_days <= 11:  return 4.0, 5, 5.0   # 第4-14天
    elif valid_days <= 27:  return 3.5, 4, 4.5   # 第15-30天
    else:                   return 2.5, 3, 3.5   # 第31天起


def get_phase(valid_days: int) -> int:
    if valid_days == 0:     return 0
    elif valid_days <= 11:  return 1
    elif valid_days <= 27:  return 2
    else:                   return 3


def get_confidence(valid_days: int) -> float:
    return round(min(1.0, valid_days / 30), 2)


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


def scratch_stats(scratch_count: int, is_sick: bool):
    """生成抓挠次数相关的派生统计量"""
    avg_dur = max(500, int(np.random.normal(4500 if is_sick else 4000, 800)))
    tot_dur = scratch_count * avg_dur
    max_dur = avg_dur + max(0, int(np.random.normal(1200 if is_sick else 900, 300)))
    night_r = np.random.uniform(0.20, 0.35) if is_sick else np.random.uniform(0.05, 0.15)
    night   = max(0, int(scratch_count * night_r))
    wear    = int(np.random.uniform(1350, 1440))
    return tot_dur, avg_dur, max_dur, night, wear


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


def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def tbl_daily(sn: str) -> str:
    """device_sn → 表名（直接用 device_sn 小写）"""
    return sn.lower()


# ══════════════════════════════════════════════════════
#  核心：运行评估算法，构建每日行记录
# ══════════════════════════════════════════════════════

def build_daily_rows(sc: dict, seed: int = 42) -> list:
    """
    对一个场景运行完整的皮肤健康评估算法，返回所有日记录。
    每条记录是一个与 INSERT SQL 对应的元组。
    """
    np.random.seed(seed)

    sn        = sc['sn']
    phases    = sc['phases']
    temp_coef = sc['tc']
    gap_map   = build_gap_map(sc['gaps'])
    sick      = sc.get('sick')

    mean  = None
    std   = None
    buf_c = []
    buf_t = []
    consec      = 0
    valid_days  = 0
    gap_counter = 0
    in_gap      = False
    just_resumed = False
    recent_z_buf = []   # 用于计算 avg_z

    rows = []
    now  = now_ts()

    for i in range(DAYS):
        d    = START_DATE + timedelta(days=i)
        ts   = to_ts(d)
        temp = round(float(_temperature[i]), 1)
        humi = round(float(_humidity[i]),    1)
        is_sick_day = bool(sick and sick[0] <= i < sick[1])

        # ── 缺口天 ───────────────────────────────────────────
        if i in gap_map:
            gap_reason = gap_map[i]
            dq         = GAP_REASON_TO_DQ.get(gap_reason, DATA_QUALITY_UNWORN)
            gap_counter += 1
            in_gap       = True
            consec       = 0

            rows.append((
                ts,
                0, 0, 0, 0, 0,          # scratch stats
                temp, humi,
                safe(mean), safe(std), None, None,
                None, None,             # zscore, avg_zscore
                0,                      # consec_abnormal
                get_phase(valid_days),
                None, None, None,       # thresholds
                valid_days,
                0, 0, None,             # is_abnormal, alert, reason
                dq, 0,                  # data_quality, wear_minutes
                0, 1, 0,                # in_warmup, in_gap, just_resumed
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
        tot_dur, avg_dur, max_dur, night, wear = scratch_stats(count, is_sick_day)

        # ── 热身期 ────────────────────────────────────────────
        if i < WARMUP:
            buf_c.append(count)
            buf_t.append(temp)
            rows.append((
                ts,
                count, tot_dur, avg_dur, max_dur, night,
                temp, humi,
                None, None, None, None,
                None, None,
                0,
                0,                      # eval_phase = 热身期
                None, None, None,
                0,
                0, 0, None,
                DATA_QUALITY_NORMAL, wear,
                1, 0, 0,                # in_warmup=1
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
            rows.append((
                ts,
                count, tot_dur, avg_dur, max_dur, night,
                temp, humi,
                round(mean, 2), round(std, 2), 0.0, 0.0,
                None, None,
                0,
                get_phase(valid_days),
                safe(tz), safe(tc), safe(ta),
                valid_days,
                0, 0, None,
                DATA_QUALITY_BUFFER, wear,
                0, 0, 1,                # just_resumed=1
                now, now,
            ))
            continue

        # ── 正常评估 ──────────────────────────────────────────
        coef        = estimate_temp_coef(buf_c, buf_t)
        temp_effect = round(coef * (temp - 20), 2)
        zscore      = round(((count - mean) - temp_effect) / std, 2)
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

        # avg_z：取最近 tc 个有效 z-score（含本次）
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
            ts,
            count, tot_dur, avg_dur, max_dur, night,
            temp, humi,
            round(mean, 2), round(std, 2), coef, temp_effect,
            zscore, avg_z,
            consec,
            get_phase(valid_days),
            safe(tz), safe(tc), safe(ta),
            valid_days,
            int(is_abn), int(alert), reason,
            DATA_QUALITY_NORMAL, wear,
            0, 0, 0,
            now, now,
        ))

    return rows


# ══════════════════════════════════════════════════════
#  数据库操作
# ══════════════════════════════════════════════════════

def get_conn(database: str = None):
    cfg = dict(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD)
    if database:
        cfg['database'] = database
    return mysql.connector.connect(**cfg)


def create_database():
    conn   = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{SKIN_DB}` "
                   f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ 数据库 {SKIN_DB} 已就绪")


def create_daily_table(conn, sn: str):
    t = tbl_daily(sn)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS `{t}` (
          `stat_date_ts`        bigint        NOT NULL
                                COMMENT '统计日 UTC 零点 ms（主键）',
          `scratch_count`       smallint      NOT NULL DEFAULT 0
                                COMMENT '当日抓挠总次数',
          `scratch_duration`    int           NOT NULL DEFAULT 0
                                COMMENT '当日抓挠总时长 ms',
          `scratch_avg_dur`     int           NOT NULL DEFAULT 0
                                COMMENT '单次平均时长 ms',
          `scratch_max_dur`     int           NOT NULL DEFAULT 0
                                COMMENT '单次最长时长 ms',
          `night_scratch_count` smallint      NOT NULL DEFAULT 0
                                COMMENT '夜间抓挠次数(22:00-06:00)',
          `avg_temperature`     decimal(4,1)  DEFAULT NULL
                                COMMENT '当日平均温度 °C',
          `avg_humidity`        decimal(4,1)  DEFAULT NULL
                                COMMENT '当日平均湿度 %',
          `baseline_mean`       decimal(6,2)  DEFAULT NULL
                                COMMENT '个体基线均值 次/天',
          `baseline_std`        decimal(6,2)  DEFAULT NULL
                                COMMENT '个体基线标准差',
          `temp_coef`           decimal(5,3)  DEFAULT NULL
                                COMMENT '温度修正系数 次/°C',
          `temp_effect`         decimal(5,2)  DEFAULT NULL
                                COMMENT '当日温度效应 次',
          `zscore`              decimal(6,2)  DEFAULT NULL
                                COMMENT '温度修正后 z-score',
          `avg_zscore`          decimal(6,2)  DEFAULT NULL
                                COMMENT '近N天均值 z-score',
          `consec_abnormal`     tinyint       NOT NULL DEFAULT 0
                                COMMENT '当前连续异常天数',
          `eval_phase`          tinyint       NOT NULL DEFAULT 0
                                COMMENT '评估阶段(0:热身期 1:早期4-14天 2:过渡期15-30天 3:稳定期31天+)',
          `threshold_z`         decimal(4,2)  DEFAULT NULL
                                COMMENT '当日 z-score 门槛',
          `threshold_consec`    tinyint       DEFAULT NULL
                                COMMENT '当日连续天数门槛',
          `threshold_avgz`      decimal(4,2)  DEFAULT NULL
                                COMMENT '当日均值 z 门槛',
          `valid_days`          smallint      NOT NULL DEFAULT 0
                                COMMENT '有效数据累计天数',
          `is_abnormal`         tinyint(1)    NOT NULL DEFAULT 0
                                COMMENT '当日是否异常(1=是)',
          `alert_triggered`     tinyint(1)    NOT NULL DEFAULT 0
                                COMMENT '是否触发推送(1=是)',
          `alert_reason`        varchar(128)  DEFAULT NULL
                                COMMENT '触发原因描述',
          `data_quality`        tinyint       NOT NULL DEFAULT 0
                                COMMENT '数据质量(0:正常 1:未佩戴 2:没电 3:信号丢失 4:松动无效 5:缓冲天)',
          `wear_minutes`        smallint      NOT NULL DEFAULT 0
                                COMMENT '有效佩戴分钟数',
          `in_warmup_flag`      tinyint(1)    NOT NULL DEFAULT 0
                                COMMENT '是否热身期',
          `in_gap_flag`         tinyint(1)    NOT NULL DEFAULT 0
                                COMMENT '是否缺口天',
          `just_resumed_flag`   tinyint(1)    NOT NULL DEFAULT 0
                                COMMENT '是否恢复佩戴缓冲天',
          `created_at`          bigint        NOT NULL
                                COMMENT '记录创建时间 UTC ms',
          `updated_at`          bigint        NOT NULL
                                COMMENT '记录更新时间 UTC ms',
          PRIMARY KEY (`stat_date_ts`),
          KEY `idx_abnormal` (`is_abnormal`,     `stat_date_ts`),
          KEY `idx_alert`    (`alert_triggered`,  `stat_date_ts`),
          KEY `idx_quality`  (`data_quality`,     `stat_date_ts`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
          COMMENT='设备 {sn} 皮肤健康每日评估结果（场景模拟）';
    """)
    conn.commit()
    cursor.close()
    print(f"  ✅ 表 {t} 已就绪")


def create_baseline_table(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS `skin_baseline` (
          `device_sn`       varchar(64)   NOT NULL
                            COMMENT '设备序列号',
          `baseline_mean`   decimal(6,2)  NOT NULL
                            COMMENT '基线均值 次/天',
          `baseline_std`    decimal(6,2)  NOT NULL
                            COMMENT '基线标准差',
          `temp_coef`       decimal(5,3)  NOT NULL DEFAULT 0.000
                            COMMENT '温度修正系数 次/°C',
          `valid_days`      smallint      NOT NULL DEFAULT 0
                            COMMENT '参与计算的有效正常天数',
          `eval_phase`      tinyint       NOT NULL DEFAULT 0
                            COMMENT '评估阶段(0:热身期 1:早期 2:过渡期 3:稳定期)',
          `confidence`      decimal(4,2)  NOT NULL DEFAULT 0.00
                            COMMENT '基线置信度 0.00-1.00',
          `last_updated_ts` bigint        NOT NULL
                            COMMENT '基线最后更新时间 UTC ms',
          `created_at`      bigint        NOT NULL
                            COMMENT '首次创建时间 UTC ms',
          PRIMARY KEY (`device_sn`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
          COMMENT='各设备皮肤健康基线汇总（所有设备共用此表）';
    """)
    conn.commit()
    cursor.close()
    print("  ✅ 表 skin_baseline 已就绪")


def insert_daily_rows(conn, sn: str, rows: list):
    t   = tbl_daily(sn)
    sql = f"""
        INSERT IGNORE INTO `{t}`
          (`stat_date_ts`,
           `scratch_count`, `scratch_duration`, `scratch_avg_dur`,
           `scratch_max_dur`, `night_scratch_count`,
           `avg_temperature`, `avg_humidity`,
           `baseline_mean`, `baseline_std`, `temp_coef`, `temp_effect`,
           `zscore`, `avg_zscore`, `consec_abnormal`,
           `eval_phase`, `threshold_z`, `threshold_consec`, `threshold_avgz`,
           `valid_days`, `is_abnormal`, `alert_triggered`, `alert_reason`,
           `data_quality`, `wear_minutes`,
           `in_warmup_flag`, `in_gap_flag`, `just_resumed_flag`,
           `created_at`, `updated_at`)
        VALUES
          (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
           %s, %s, %s, %s, %s, %s, %s)
    """
    cursor = conn.cursor()
    cursor.executemany(sql, rows)
    conn.commit()
    print(f"  [{sn}] 插入 {cursor.rowcount} 条每日评估记录")
    cursor.close()


def upsert_baseline(conn, sn: str, daily_rows: list):
    """
    从刚生成的 daily_rows 中取最后30条有效正常天，计算并写入 skin_baseline。
    """
    # daily_rows 列索引:
    #   0=stat_date_ts, 1=scratch_count, 8=baseline_mean, 9=baseline_std,
    #   10=temp_coef, 19=valid_days, 22=alert_reason, 23=data_quality,
    #   25=in_warmup_flag, 26=in_gap_flag, 27=just_resumed_flag
    WARMUP_IDX   = 25
    GAP_IDX      = 26
    RESUMED_IDX  = 27
    DQ_IDX       = 23
    COUNT_IDX    = 1
    TEMP_IDX     = 6
    VALID_IDX    = 19

    normal = [
        r for r in daily_rows
        if not r[WARMUP_IDX] and not r[GAP_IDX] and not r[RESUMED_IDX]
        and r[DQ_IDX] == 0
    ]
    last30 = normal[-30:]
    if not last30:
        print(f"  [{sn}] 无有效正常天，跳过基线写入")
        return

    counts     = [r[COUNT_IDX] for r in last30]
    temps      = [float(r[TEMP_IDX]) for r in last30]
    max_valid  = max(r[VALID_IDX] for r in last30)

    mean_val   = round(float(np.mean(counts)), 2)
    std_val    = round(max(float(np.std(counts)), MIN_STD), 2)
    coef_val   = estimate_temp_coef(counts, temps)
    valid_days = min(max_valid, 30)
    phase      = (3 if valid_days > 27 else 2 if valid_days > 11 else
                  1 if valid_days > 0 else 0)
    confidence = round(min(1.0, valid_days / 30), 2)
    ts         = now_ts()

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO `skin_baseline`
          (`device_sn`, `baseline_mean`, `baseline_std`, `temp_coef`,
           `valid_days`, `eval_phase`, `confidence`,
           `last_updated_ts`, `created_at`)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
          `baseline_mean`   = VALUES(`baseline_mean`),
          `baseline_std`    = VALUES(`baseline_std`),
          `temp_coef`       = VALUES(`temp_coef`),
          `valid_days`      = VALUES(`valid_days`),
          `eval_phase`      = VALUES(`eval_phase`),
          `confidence`      = VALUES(`confidence`),
          `last_updated_ts` = VALUES(`last_updated_ts`)
    """, (sn, mean_val, std_val, coef_val,
          valid_days, phase, confidence, ts, ts))
    conn.commit()
    print(f"  [{sn}] 基线: 均值={mean_val}  标准差={std_val}  "
          f"温度系数={coef_val}  有效天={valid_days}  置信度={confidence}")
    cursor.close()


# ══════════════════════════════════════════════════════
#  查询验证
# ══════════════════════════════════════════════════════

def query_summary():
    conn   = get_conn(SKIN_DB)
    cursor = conn.cursor()

    print("\n======= 各设备每日评估概况 =======")
    for sc in SCENARIOS:
        t = tbl_daily(sc['sn'])
        try:
            cursor.execute(f"""
                SELECT
                    COUNT(*)                              AS 总天数,
                    SUM(in_warmup_flag)                   AS 热身期,
                    SUM(in_gap_flag)                      AS 缺口天,
                    SUM(just_resumed_flag)                AS 缓冲天,
                    SUM(is_abnormal)                      AS 异常天,
                    SUM(alert_triggered)                  AS 推送次数,
                    ROUND(AVG(CASE WHEN data_quality=0
                              THEN scratch_count END), 1) AS 日均抓挠,
                    ROUND(MAX(zscore), 2)                 AS 最高z_score,
                    MAX(valid_days)                       AS 最终有效天
                FROM `{t}`
            """)
            row = cursor.fetchone()
            print(f"  {sc['sn']:20s}  "
                  f"总={row[0]:3d}天  热身={row[1]}  缺口={row[2]}  缓冲={row[3]}  "
                  f"异常={row[4]}  推送={row[5]}  "
                  f"日均={row[6]}次  最高z={row[7]}  有效天={row[8]}")
        except Exception as e:
            print(f"  {sc['sn']}: 查询失败 {e}")

    print("\n======= skin_baseline 汇总 =======")
    cursor.execute("""
        SELECT device_sn, baseline_mean, baseline_std, temp_coef,
               valid_days, confidence,
               CASE eval_phase
                 WHEN 0 THEN '热身期'
                 WHEN 1 THEN '早期'
                 WHEN 2 THEN '过渡期'
                 WHEN 3 THEN '稳定期'
               END AS 阶段
        FROM skin_baseline
        ORDER BY device_sn
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:20s}  均值={row[1]}  标准差={row[2]}  "
              f"温度系数={row[3]}  有效天={row[4]}  置信度={row[5]}  {row[6]}")

    print("\n======= 推送记录明细 =======")
    found = False
    for sc in SCENARIOS:
        t = tbl_daily(sc['sn'])
        try:
            cursor.execute(f"""
                SELECT
                    FROM_UNIXTIME(stat_date_ts/1000) AS 日期,
                    scratch_count, zscore, avg_zscore,
                    consec_abnormal, alert_reason
                FROM `{t}`
                WHERE alert_triggered = 1
                ORDER BY stat_date_ts
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

if __name__ == "__main__":
    print("=== 第一步：创建数据库 ===")
    create_database()

    conn = get_conn(SKIN_DB)

    print("\n=== 第二步：建表 ===")
    for sc in SCENARIOS:
        create_daily_table(conn, sc['sn'])
    create_baseline_table(conn)

    print("\n=== 第三步：生成并插入每日评估数据 ===")
    for idx, sc in enumerate(SCENARIOS):
        rows = build_daily_rows(sc, seed=42 + idx)
        print(f"  [{sc['sn']}] 生成 {len(rows)} 天数据，开始插入...")
        insert_daily_rows(conn, sc['sn'], rows)

    print("\n=== 第四步：计算并写入基线 ===")
    for idx, sc in enumerate(SCENARIOS):
        rows = build_daily_rows(sc, seed=42 + idx)
        upsert_baseline(conn, sc['sn'], rows)

    conn.close()

    print("\n=== 第五步：查询验证 ===")
    query_summary()

    print("\n🎉 皮肤健康评估数据库完成！")
