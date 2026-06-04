"""
行为识别数据库（PostgreSQL）
=====================
数据库: pet_collar，模式: pet_dog_behavior
每个设备独立一张表: {device_id}

每行 = ML 模型输出的一个行为事件
  ts_start     行为开始时间 UTC ms
  ts_end       行为结束时间 UTC ms
  behavior     1=运动 2=睡眠 3=抓挠
  duration_sec 持续时长 秒
  confidence   模型置信度 0.0-1.0
"""

import psycopg2
import numpy as np
from datetime import date, timedelta, datetime, timezone

# ══════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════
PG_HOST     = "127.0.0.1"
PG_PORT     = 5432
PG_USER     = "postgres"
PG_PASSWORD = "123456"
PG_DB       = "pet_collar"
BEH_SCHEMA  = "pet_dog_behavior"

DAYS       = 180
START_DATE = date(2024, 1, 1)

BEHAVIOR_MOVE    = 1
BEHAVIOR_SLEEP   = 2
BEHAVIOR_SCRATCH = 3

# ══════════════════════════════════════════════════════
#  全局时间序列（seed=42）
# ══════════════════════════════════════════════════════
np.random.seed(42)

_temperature = (22 + 13 * np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, DAYS))
                + np.random.normal(0, 1.5, DAYS))
_humidity    = (65 + 15 * np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, DAYS))
                + np.random.normal(0, 3.0, DAYS))

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
    {'sn': 'device_id_1',  'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'sn': 'device_id_2',  'phases': [(0, 60, 10.0, 2.0), (60, 80, 30.0, 4.0), (80, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': (60, 80)},
    {'sn': 'device_id_3',  'phases': [(0, 60, 10.0, 2.0), (60, 180, 28.0, 4.0)], 'tc': 0.10, 'gaps': [], 'sick': (60, 180)},
    {'sn': 'device_id_4',  'phases': [(0, 40, 10.0, 2.0), (40, 55, 28.0, 4.0), (55, 120, 10.0, 2.0), (120, 135, 30.0, 4.0), (135, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None, 'sick_episodes': [(40, 55), (120, 135)]},
    {'sn': 'device_id_5',  'phases': [(0, 60, 10.0, 2.0), (60, 120, 15.0, 2.0), (120, 180, 22.0, 3.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'sn': 'device_id_6',  'phases': [(0, 90, 10.0, 2.0), (90, 180, 25.0, 3.0)], 'tc': 0.10, 'gaps': [], 'sick': (90, 180)},
    {'sn': 'device_id_7',  'phases': [(0, 50, 10.0, 2.0), (50, 80, 45.0, 6.0), (80, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': (50, 80)},
    {'sn': 'device_id_8',  'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.35, 'gaps': [], 'sick': None},
    {'sn': 'device_id_9',  'phases': [(0, 30, 10.0, 2.0), (30, 90, 3.0, 1.0), (90, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'sn': 'device_id_10', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(35, 38, 'unworn')], 'sick': None},
    {'sn': 'device_id_11', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(40, 45, 'battery')], 'sick': None},
    {'sn': 'device_id_12', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(30, 65, 'battery')], 'sick': None},
    {'sn': 'device_id_13', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(d, d + 1, 'signal') for d in sorted(_signal_gap_days)], 'sick': None},
    {'sn': 'device_id_14', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(50, 58, 'loose')], 'sick': None},
    {'sn': 'device_id_15', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(88, 92, 'battery')], 'sick': None},
    {'sn': 'device_id_16', 'phases': [(0, 70, 10.0, 2.0), (70, 90, 35.0, 5.0), (90, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None, 'drift_range': (70, 90)},
    {'sn': 'device_id_17', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.30, 'gaps': [], 'sick': None},
    {'sn': 'device_id_18', 'phases': [(0, 60, 10.0, 2.0), (60, 180, 13.0, 2.0)], 'tc': 0.15, 'gaps': [], 'sick': None, 'temp_shift': (60, 5.0)},
    {'sn': 'device_id_19', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(80, 90, 'unworn')], 'sick': None},
    {'sn': 'device_id_20', 'phases': [(0, 180, 14.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'sn': 'device_id_21', 'phases': [(0, 180, 15.0, 4.0)], 'tc': 0.10, 'gaps': [], 'sick': None, 'warmup': 7},
    {'sn': 'device_id_22', 'phases': [(0, 180, 5.0, 1.0)],  'tc': 0.05, 'gaps': [], 'sick': None},
    {'sn': 'device_id_23', 'phases': [(0, 180, 20.0, 3.0)], 'tc': 0.12, 'gaps': [], 'sick': None},
    {'sn': 'device_id_24', 'phases': [(0, 180, 4.0, 1.0)],  'tc': 0.08, 'gaps': [], 'sick': None},
]


# ══════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════

def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def tbl(sn: str) -> str:
    return f"{BEH_SCHEMA}.{sn.lower()}"


def is_sick_day(day_idx: int, sc: dict) -> bool:
    episodes = sc.get('sick_episodes')
    if episodes:
        return any(s <= day_idx < e for s, e in episodes)
    sick = sc.get('sick')
    if sick:
        return sick[0] <= day_idx < sick[1]
    return False


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
#  行为事件生成
# ══════════════════════════════════════════════════════

def gen_confidence(behavior: int, mixed_period: bool) -> float:
    if mixed_period:
        return round(np.random.uniform(0.65, 0.85), 3)
    return round(np.random.uniform(0.85, 0.98), 3)


def gen_day_events(day_idx: int, n_scratch: int, is_sick: bool) -> list:
    d      = START_DATE + timedelta(days=day_idx)
    day_ts = to_ts(d)
    rows   = []

    s_morn = n_scratch // 3
    s_aftn = n_scratch - s_morn
    segments = [
        (0,         7 * 3600,  0.85, 0.15, 0,      False),
        (7 * 3600,  12 * 3600, 0.10, 0.85, s_morn, True),
        (12 * 3600, 14 * 3600, 0.80, 0.20, 0,      False),
        (14 * 3600, 20 * 3600, 0.10, 0.80, s_aftn, True),
        (20 * 3600, 24 * 3600, 0.75, 0.25, 0,      False),
    ]

    for seg_s, seg_e, sleep_w, _, n_sc, mixed in segments:
        cursor = seg_s
        scratch_times = (
            sorted(np.random.randint(seg_s, seg_e, n_sc).tolist())
            if n_sc > 0 else []
        )
        sc_idx = 0

        while cursor < seg_e:
            if sc_idx < len(scratch_times) and cursor >= scratch_times[sc_idx]:
                dur_ms  = int(np.random.uniform(1000, 8000))
                s_ts    = day_ts + scratch_times[sc_idx] * 1000
                e_ts    = s_ts + dur_ms
                dur_sec = dur_ms / 1000.0
                conf    = gen_confidence(BEHAVIOR_SCRATCH, mixed)
                rows.append((s_ts, e_ts, BEHAVIOR_SCRATCH, round(dur_sec, 2), conf))
                cursor  = scratch_times[sc_idx] + dur_ms // 1000 + 1
                sc_idx += 1
            else:
                btype = BEHAVIOR_SLEEP if np.random.random() < sleep_w else BEHAVIOR_MOVE
                dur_sec = (int(np.random.uniform(600, 3600))
                           if btype == BEHAVIOR_SLEEP
                           else int(np.random.uniform(60, 900)))
                dur_sec = min(dur_sec, seg_e - cursor)
                if dur_sec <= 0:
                    break
                s_ts = day_ts + cursor * 1000
                e_ts = s_ts + dur_sec * 1000
                conf = gen_confidence(btype, mixed)
                rows.append((s_ts, e_ts, btype, float(dur_sec), conf))
                cursor += dur_sec

    return rows


def build_scenario_rows(sc: dict, seed: int = 42) -> list:
    np.random.seed(seed)
    gap_map  = build_gap_map(sc['gaps'])
    all_rows = []

    for i in range(DAYS):
        if i in gap_map:
            continue
        temp      = float(_temperature[i])
        n_scratch = scratch_count_for_day(i, sc['phases'], temp, sc['tc'])
        sick      = is_sick_day(i, sc)
        day_rows  = gen_day_events(i, n_scratch, sick)
        all_rows.extend(day_rows)

    return all_rows


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
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {BEH_SCHEMA}")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[OK] 模式 {BEH_SCHEMA} 已就绪")


def create_table(conn, sn: str):
    t = tbl(sn)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {t} (
          id           BIGSERIAL     PRIMARY KEY,
          ts_start     BIGINT        NOT NULL,
          ts_end       BIGINT        NOT NULL,
          behavior     SMALLINT      NOT NULL,
          duration_sec NUMERIC(10,2) NOT NULL,
          confidence   NUMERIC(5,3)  NOT NULL
        )
    """)
    cursor.execute(f"CREATE INDEX IF NOT EXISTS {sn}_idx_ts ON {t} (ts_start)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS {sn}_idx_beh ON {t} (behavior, ts_start)")
    conn.commit()
    cursor.close()
    print(f"  [OK] 表 {t} 已就绪")


def insert_rows(conn, sn: str, rows: list):
    if not rows:
        print(f"  [{sn}] 无数据，跳过")
        return

    t   = tbl(sn)
    sql = f"""
        INSERT INTO {t} (ts_start, ts_end, behavior, duration_sec, confidence)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """
    cursor     = conn.cursor()
    batch_size = 1000
    total      = 0
    for i in range(0, len(rows), batch_size):
        cursor.executemany(sql, rows[i: i + batch_size])
        conn.commit()
        total += cursor.rowcount
    cursor.close()
    print(f"  [{sn}] 插入 {total} 条行为事件记录")


# ══════════════════════════════════════════════════════
#  查询验证
# ══════════════════════════════════════════════════════

def query_summary():
    conn   = get_conn()
    cursor = conn.cursor()

    print("\n======= 行为数据概况 =======")
    for sc in SCENARIOS:
        t = tbl(sc['sn'])
        try:
            cursor.execute(f"""
                SELECT
                    COUNT(*)                                               AS total,
                    SUM(CASE WHEN behavior=1 THEN 1 ELSE 0 END)           AS move_cnt,
                    SUM(CASE WHEN behavior=2 THEN 1 ELSE 0 END)           AS sleep_cnt,
                    SUM(CASE WHEN behavior=3 THEN 1 ELSE 0 END)           AS scratch_cnt,
                    ROUND(AVG(confidence)::numeric, 3)                    AS avg_conf,
                    to_char(to_timestamp(MIN(ts_start)/1000), 'YYYY-MM-DD HH24:MI') AS earliest,
                    to_char(to_timestamp(MAX(ts_end)/1000),   'YYYY-MM-DD HH24:MI') AS latest
                FROM {t}
            """)
            row = cursor.fetchone()
            print(f"  {sc['sn']:20s}  总={int(row[0]):6d}  "
                  f"运动={row[1]}  睡眠={row[2]}  抓挠={row[3]}  "
                  f"均置信={row[4]}  {row[5]} ~ {row[6]}")
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

    print("\n=== 第三步：生成并插入数据 ===")
    for idx, sc in enumerate(SCENARIOS):
        rows = build_scenario_rows(sc, seed=42 + idx)
        print(f"  [{sc['sn']}] 生成 {len(rows)} 条事件，开始插入...")
        insert_rows(conn, sc['sn'], rows)

    conn.close()

    print("\n=== 第四步：查询验证 ===")
    query_summary()

    print("\n[完成] 行为数据库写入完毕！")


if __name__ == "__main__":
    main()
