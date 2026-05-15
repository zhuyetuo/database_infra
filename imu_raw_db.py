"""
IMU 原始数据库
=====================
数据库: pet_dog_imu
每个设备独立一张表: {device_sn}

五个场景，8 个设备:
  A : DEV_A_NORMAL    — 完全正常
  B : DEV_B_SICK      — 短期皮肤病后康复
  C : DEV_C_SEASON    — 季节性正常升高（温度系数高）
  D : DEV_D_ALLERGY   — 持续缓慢升高（过敏加重）
  E1: DEV_E1_UNWORN   — 忘记佩戴（3天缺口）
  E2: DEV_E2_BATTERY  — 没电（5天缺口）+ 缺口后皮肤病
  E3: DEV_E3_SIGNAL   — 信号不稳定（断续丢失）
  E4: DEV_E4_LOOSE    — 项圈松动（8天无效数据）

表结构：每行是一个连续行为片段的 IMU 特征摘要
  ts_start/ts_end  行为时间段 UTC ms
  behavior         1=运动 2=睡眠 3=抓挠
  ax/ay/az         加速度均值 mg
  gx/gy/gz         陀螺仪均值 deg/s
  az_rms           垂直加速度 RMS（活动强度代理）
  ax_peak          X 轴峰值（抓挠节奏代理）
  scratch_hz       主频 Hz（仅抓挠事件有值）
  confidence       行为分类置信度
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
IMU_DB      = "pet_dog_imu"

DAYS       = 180
WARMUP     = 3
START_DATE = date(2024, 1, 1)

BEHAVIOR_MOVE    = 1
BEHAVIOR_SLEEP   = 2
BEHAVIOR_SCRATCH = 3

# ══════════════════════════════════════════════════════
#  全局时间序列（与 demo_all.py 保持一致）
# ══════════════════════════════════════════════════════
np.random.seed(42)

_temperature = (22 + 13 * np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, DAYS))
                + np.random.normal(0, 1.5, DAYS))

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
#  场景定义
# ══════════════════════════════════════════════════════
# gap_specs: list of (start_day_inclusive, end_day_exclusive, reason_str)
# phases:    list of (start_day, end_day, mean_scratch, std_scratch)
# sick:      (start_day, end_day) or None
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
#  工具函数
# ══════════════════════════════════════════════════════

def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def tbl(sn: str) -> str:
    """device_sn → 表名（直接用 device_sn 小写）"""
    return sn.lower()


def scratch_count_for_day(day_idx: int, phases: list, temp: float, tc: float) -> int:
    for s, e, mean, std in phases:
        if s <= day_idx < e:
            return max(0, int(np.random.normal(mean + tc * (temp - 20), std)))
    return 0


def build_gap_map(gaps: list) -> dict:
    gm = {}
    for start, end, reason in gaps:
        for d in range(start, min(end, DAYS)):
            gm[d] = reason
    return gm


# ══════════════════════════════════════════════════════
#  IMU 特征生成
# ══════════════════════════════════════════════════════

def gen_imu(behavior: int, sick_intensity: float = 1.0) -> tuple:
    """
    根据行为类型生成 IMU 特征均值。
    sick_intensity: 1.0=正常, >1.0=发病期（抓挠幅度更大）
    返回 (ax, ay, az, gx, gy, gz, az_rms, ax_peak, scratch_hz, confidence)
    """
    if behavior == BEHAVIOR_SLEEP:
        ax  = round(np.random.normal(0,   20),  2)
        ay  = round(np.random.normal(0,   15),  2)
        az  = round(np.random.normal(980, 25),  2)
        gx  = round(np.random.normal(0,   2),   2)
        gy  = round(np.random.normal(0,   2),   2)
        gz  = round(np.random.normal(0,   1.5), 2)
        az_rms  = round(abs(az) + np.random.uniform(5, 20),   2)
        ax_peak = round(abs(ax) + np.random.uniform(10, 40),  2)
        hz  = None
        conf = round(float(np.clip(np.random.normal(0.92, 0.05), 0.70, 1.00)), 2)

    elif behavior == BEHAVIOR_MOVE:
        ax  = round(np.random.normal(40,  150), 2)
        ay  = round(np.random.normal(20,  120), 2)
        az  = round(np.random.normal(650, 280), 2)
        gx  = round(np.random.normal(0,   90),  2)
        gy  = round(np.random.normal(0,   70),  2)
        gz  = round(np.random.normal(0,   55),  2)
        az_rms  = round(abs(az) + np.random.uniform(80,  250), 2)
        ax_peak = round(abs(ax) + np.random.uniform(100, 400), 2)
        hz  = None
        conf = round(float(np.clip(np.random.normal(0.85, 0.07), 0.60, 1.00)), 2)

    else:  # SCRATCH
        si  = sick_intensity
        ax  = round(np.random.normal(180 * si,  80),  2)
        ay  = round(np.random.normal(40,         60),  2)
        az  = round(np.random.normal(800,        180), 2)
        gx  = round(np.random.normal(0,  130 * si),    2)
        gy  = round(np.random.normal(0,   90),          2)
        gz  = round(np.random.normal(0,   70),          2)
        az_rms  = round(abs(az) + np.random.uniform(150 * si, 350 * si), 2)
        ax_peak = round(abs(ax) * 2.5 + np.random.uniform(100, 300),     2)
        hz  = round(float(np.random.uniform(2.0, 5.5)), 2)
        conf = round(float(np.clip(np.random.normal(0.88, 0.06), 0.70, 1.00)), 2)

    return ax, ay, az, gx, gy, gz, az_rms, ax_peak, hz, conf


# ══════════════════════════════════════════════════════
#  每日事件生成
# ══════════════════════════════════════════════════════

def gen_day_events(day_idx: int, n_scratch: int, sick_intensity: float) -> list:
    """
    按时段生成一天的行为事件，返回行元组列表（不含 device_sn）。
    列顺序: ts_start, ts_end, behavior, ax, ay, az, gx, gy, gz,
            az_rms, ax_peak, scratch_hz, confidence
    """
    d      = START_DATE + timedelta(days=day_idx)
    day_ts = to_ts(d)
    rows   = []

    # (seg_start_sec, seg_end_sec, sleep_weight, move_weight, scratch_slots)
    s_morn = n_scratch // 3
    s_aftn = n_scratch - s_morn
    segments = [
        (0,        7 * 3600, 0.85, 0.15, 0),
        (7 * 3600, 12 * 3600, 0.10, 0.85, s_morn),
        (12 * 3600, 14 * 3600, 0.80, 0.20, 0),
        (14 * 3600, 20 * 3600, 0.10, 0.80, s_aftn),
        (20 * 3600, 24 * 3600, 0.75, 0.25, 0),
    ]

    for seg_s, seg_e, sleep_w, _, n_sc in segments:
        cursor = seg_s

        scratch_times = (
            sorted(np.random.randint(seg_s, seg_e, n_sc).tolist())
            if n_sc > 0 else []
        )
        sc_idx = 0

        while cursor < seg_e:
            if sc_idx < len(scratch_times) and cursor >= scratch_times[sc_idx]:
                dur_ms = int(np.random.uniform(1000, 8000))
                s_ts   = day_ts + scratch_times[sc_idx] * 1000
                e_ts   = s_ts + dur_ms
                imu    = gen_imu(BEHAVIOR_SCRATCH, sick_intensity)
                rows.append((s_ts, e_ts, BEHAVIOR_SCRATCH) + imu)
                cursor  = scratch_times[sc_idx] + dur_ms // 1000 + 1
                sc_idx += 1
            else:
                btype    = BEHAVIOR_SLEEP if np.random.random() < sleep_w else BEHAVIOR_MOVE
                dur_sec  = (int(np.random.uniform(600, 3600))
                            if btype == BEHAVIOR_SLEEP
                            else int(np.random.uniform(60, 900)))
                dur_sec  = min(dur_sec, seg_e - cursor)
                if dur_sec <= 0:
                    break
                s_ts = day_ts + cursor * 1000
                e_ts = s_ts + dur_sec * 1000
                imu  = gen_imu(btype, 1.0)
                rows.append((s_ts, e_ts, btype) + imu)
                cursor += dur_sec

    return rows


# ══════════════════════════════════════════════════════
#  场景数据构建
# ══════════════════════════════════════════════════════

def build_scenario_rows(sc: dict, seed: int = 42) -> list:
    np.random.seed(seed)

    gap_map    = build_gap_map(sc['gaps'])
    sick_range = sc.get('sick')
    all_rows   = []

    for i in range(DAYS):
        if i in gap_map:
            continue  # 缺口天：无 IMU 数据

        temp         = float(_temperature[i])
        n_scratch    = scratch_count_for_day(i, sc['phases'], temp, sc['tc'])
        sick_intens  = (1.8 if sick_range and sick_range[0] <= i < sick_range[1]
                        else 1.0)

        day_rows = gen_day_events(i, n_scratch, sick_intens)
        all_rows.extend(day_rows)

    return all_rows


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
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{IMU_DB}` "
                   f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ 数据库 {IMU_DB} 已就绪")


def create_table(conn, sn: str):
    t = tbl(sn)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS `{t}` (
          `id`         bigint       NOT NULL AUTO_INCREMENT
                       COMMENT '自增主键',
          `ts_start`   bigint       NOT NULL
                       COMMENT '行为开始时间 UTC ms',
          `ts_end`     bigint       NOT NULL
                       COMMENT '行为结束时间 UTC ms',
          `behavior`   tinyint      NOT NULL
                       COMMENT '行为类型(1:运动 2:睡眠 3:抓挠)',
          `ax`         decimal(8,2) NOT NULL DEFAULT 0
                       COMMENT '加速度X轴均值 mg',
          `ay`         decimal(8,2) NOT NULL DEFAULT 0
                       COMMENT '加速度Y轴均值 mg',
          `az`         decimal(8,2) NOT NULL DEFAULT 0
                       COMMENT '加速度Z轴均值 mg（重力方向）',
          `gx`         decimal(8,2) NOT NULL DEFAULT 0
                       COMMENT '陀螺仪X轴均值 deg/s',
          `gy`         decimal(8,2) NOT NULL DEFAULT 0
                       COMMENT '陀螺仪Y轴均值 deg/s',
          `gz`         decimal(8,2) NOT NULL DEFAULT 0
                       COMMENT '陀螺仪Z轴均值 deg/s',
          `az_rms`     decimal(8,2) NOT NULL DEFAULT 0
                       COMMENT '垂直加速度RMS（活动强度代理指标）',
          `ax_peak`    decimal(8,2) NOT NULL DEFAULT 0
                       COMMENT 'X轴峰值（抓挠节奏代理指标）',
          `scratch_hz` decimal(5,2) DEFAULT NULL
                       COMMENT '抓挠主频 Hz（仅抓挠事件有值）',
          `confidence` decimal(4,2) NOT NULL DEFAULT 0
                       COMMENT '行为分类置信度 0-1',
          PRIMARY KEY (`id`),
          KEY `idx_ts`       (`ts_start`),
          KEY `idx_behavior` (`behavior`, `ts_start`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
          COMMENT='设备 {sn} 原始IMU行为事件数据（场景模拟）';
    """)
    conn.commit()
    cursor.close()
    print(f"  ✅ 表 {t} 已就绪")


def insert_rows(conn, sn: str, rows: list):
    if not rows:
        print(f"  [{sn}] 无数据，跳过")
        return

    t   = tbl(sn)
    sql = f"""
        INSERT IGNORE INTO `{t}`
          (`ts_start`, `ts_end`, `behavior`,
           `ax`, `ay`, `az`, `gx`, `gy`, `gz`,
           `az_rms`, `ax_peak`, `scratch_hz`, `confidence`)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor     = conn.cursor()
    batch_size = 1000
    total      = 0
    for i in range(0, len(rows), batch_size):
        cursor.executemany(sql, rows[i: i + batch_size])
        conn.commit()
        total += cursor.rowcount
    cursor.close()
    print(f"  [{sn}] 插入 {total} 条 IMU 事件记录")


# ══════════════════════════════════════════════════════
#  查询验证
# ══════════════════════════════════════════════════════

def query_summary():
    conn   = get_conn(IMU_DB)
    cursor = conn.cursor()

    print("\n======= IMU 数据概况 =======")
    for sc in SCENARIOS:
        t = tbl(sc['sn'])
        try:
            cursor.execute(f"""
                SELECT
                    COUNT(*)                                    AS 总事件数,
                    SUM(behavior = 1)                           AS 运动,
                    SUM(behavior = 2)                           AS 睡眠,
                    SUM(behavior = 3)                           AS 抓挠,
                    ROUND(AVG(CASE WHEN behavior=3
                              THEN (ts_end - ts_start) END)/1000, 1)
                                                                AS 抓挠均时长秒,
                    ROUND(AVG(CASE WHEN behavior=3
                              THEN scratch_hz END), 2)          AS 抓挠均频率Hz,
                    ROUND(AVG(CASE WHEN behavior=3
                              THEN az_rms END), 1)              AS 抓挠均az_rms,
                    FROM_UNIXTIME(MIN(ts_start)/1000)           AS 最早,
                    FROM_UNIXTIME(MAX(ts_end)  /1000)           AS 最晚
                FROM `{t}`
            """)
            row = cursor.fetchone()
            print(f"  {sc['sn']:20s}  "
                  f"总={int(row[0]):5d}  运动={int(row[1]):4d}  睡眠={int(row[2]):4d}  抓挠={int(row[3]):4d}  "
                  f"抓挠均时长={row[4]}s  均频={row[5]}Hz  az_rms={row[6]}")
        except Exception as e:
            print(f"  {sc['sn']}: 查询失败 {e}")

    cursor.close()
    conn.close()


# ══════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== 第一步：创建数据库 ===")
    create_database()

    conn = get_conn(IMU_DB)

    print("\n=== 第二步：建表 ===")
    for sc in SCENARIOS:
        create_table(conn, sc['sn'])

    print("\n=== 第三步：生成并插入数据 ===")
    for idx, sc in enumerate(SCENARIOS):
        rows = build_scenario_rows(sc, seed=42 + idx)
        print(f"  [{sc['sn']}] 生成 {len(rows)} 条事件，开始插入...")
        insert_rows(conn, sc['sn'], rows)

    conn.close()

    print("\n=== 第四步：查询验证 ===")
    query_summary()

    print("\n🎉 IMU 数据库完成！")
