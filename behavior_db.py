"""
行为识别数据库（MySQL）
=====================
数据库: pet_dog_behavior
每个设备独立一张表: d_{device_id}

每行 = ML 模型输出的一个行为事件
  ts_start     行为开始时间 UTC ms
  ts_end       行为结束时间 UTC ms
  behavior     1=运动 2=睡眠 3=抓挠
  duration_sec 持续时长 秒
  confidence   模型置信度 0.0-1.0
  local_start  本地时间字符串
  local_end    本地时间字符串
  user_timezone 时区字符串
"""

import os
import pymysql
import pymysql.cursors
import numpy as np
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
    {'device_id': 1001, 'sn': 'SIM-D001', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'device_id': 1002, 'sn': 'SIM-D002', 'phases': [(0, 60, 10.0, 2.0), (60, 80, 30.0, 4.0), (80, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': (60, 80)},
    {'device_id': 1003, 'sn': 'SIM-D003', 'phases': [(0, 60, 10.0, 2.0), (60, 180, 28.0, 4.0)], 'tc': 0.10, 'gaps': [], 'sick': (60, 180)},
    {'device_id': 1004, 'sn': 'SIM-D004', 'phases': [(0, 40, 10.0, 2.0), (40, 55, 28.0, 4.0), (55, 120, 10.0, 2.0), (120, 135, 30.0, 4.0), (135, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None, 'sick_episodes': [(40, 55), (120, 135)]},
    {'device_id': 1005, 'sn': 'SIM-D005', 'phases': [(0, 60, 10.0, 2.0), (60, 120, 15.0, 2.0), (120, 180, 22.0, 3.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'device_id': 1006, 'sn': 'SIM-D006', 'phases': [(0, 90, 10.0, 2.0), (90, 180, 25.0, 3.0)], 'tc': 0.10, 'gaps': [], 'sick': (90, 180)},
    {'device_id': 1007, 'sn': 'SIM-D007', 'phases': [(0, 50, 10.0, 2.0), (50, 80, 45.0, 6.0), (80, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': (50, 80)},
    {'device_id': 1008, 'sn': 'SIM-D008', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.35, 'gaps': [], 'sick': None},
    {'device_id': 1009, 'sn': 'SIM-D009', 'phases': [(0, 30, 10.0, 2.0), (30, 90, 3.0, 1.0), (90, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'device_id': 1010, 'sn': 'SIM-D010', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(35, 38, 'unworn')], 'sick': None},
    {'device_id': 1011, 'sn': 'SIM-D011', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(40, 45, 'battery')], 'sick': None},
    {'device_id': 1012, 'sn': 'SIM-D012', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(30, 65, 'battery')], 'sick': None},
    {'device_id': 1013, 'sn': 'SIM-D013', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(d, d + 1, 'signal') for d in sorted(_signal_gap_days)], 'sick': None},
    {'device_id': 1014, 'sn': 'SIM-D014', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(50, 58, 'loose')], 'sick': None},
    {'device_id': 1015, 'sn': 'SIM-D015', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(88, 92, 'battery')], 'sick': None},
    {'device_id': 1016, 'sn': 'SIM-D016', 'phases': [(0, 70, 10.0, 2.0), (70, 90, 35.0, 5.0), (90, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None, 'drift_range': (70, 90)},
    {'device_id': 1017, 'sn': 'SIM-D017', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.30, 'gaps': [], 'sick': None},
    {'device_id': 1018, 'sn': 'SIM-D018', 'phases': [(0, 60, 10.0, 2.0), (60, 180, 13.0, 2.0)], 'tc': 0.15, 'gaps': [], 'sick': None, 'temp_shift': (60, 5.0)},
    {'device_id': 1019, 'sn': 'SIM-D019', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(80, 90, 'unworn')], 'sick': None},
    {'device_id': 1020, 'sn': 'SIM-D020', 'phases': [(0, 180, 14.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'device_id': 1021, 'sn': 'SIM-D021', 'phases': [(0, 180, 15.0, 4.0)], 'tc': 0.10, 'gaps': [], 'sick': None, 'warmup': 7},
    {'device_id': 1022, 'sn': 'SIM-D022', 'phases': [(0, 180, 5.0, 1.0)],  'tc': 0.05, 'gaps': [], 'sick': None},
    {'device_id': 1023, 'sn': 'SIM-D023', 'phases': [(0, 180, 20.0, 3.0)], 'tc': 0.12, 'gaps': [], 'sick': None},
    {'device_id': 1024, 'sn': 'SIM-D024', 'phases': [(0, 180, 4.0, 1.0)],  'tc': 0.08, 'gaps': [], 'sick': None},
]


# ══════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════

def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def ms_to_local_str(ts_ms: int, user_timezone: str = 'UTC') -> str:
    """Convert ms timestamp to local time string (YYYY-MM-DD HH:MM:SS) in UTC."""
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def tbl(device_id: int) -> str:
    return f"`{BEH_SCHEMA}`.`d_{device_id}`"


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
                local_start = ms_to_local_str(s_ts)
                local_end   = ms_to_local_str(e_ts)
                rows.append((s_ts, e_ts, BEHAVIOR_SCRATCH, round(dur_sec, 2), conf, local_start, local_end, 'UTC'))
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
                local_start = ms_to_local_str(s_ts)
                local_end   = ms_to_local_str(e_ts)
                rows.append((s_ts, e_ts, btype, float(dur_sec), conf, local_start, local_end, 'UTC'))
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
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{BEH_SCHEMA}` CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[OK] 模式 {BEH_SCHEMA} 已就绪")


def create_table(conn, device_id: int):
    t = tbl(device_id)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {t} (
          id           BIGINT        NOT NULL AUTO_INCREMENT PRIMARY KEY,
          ts_start     BIGINT        NOT NULL,
          ts_end       BIGINT        NOT NULL,
          behavior     SMALLINT      NOT NULL,
          duration_sec DECIMAL(10,2) NOT NULL,
          confidence   DECIMAL(5,3)  NOT NULL,
          local_start  VARCHAR(24)   DEFAULT NULL,
          local_end    VARCHAR(24)   DEFAULT NULL,
          user_timezone VARCHAR(32)  DEFAULT NULL,
          UNIQUE KEY uq_ts_start (ts_start)
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
          (ts_start, ts_end, behavior, duration_sec, confidence, local_start, local_end, user_timezone)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor     = conn.cursor()
    batch_size = 1000
    total      = 0
    for i in range(0, len(rows), batch_size):
        cursor.executemany(sql, rows[i: i + batch_size])
        conn.commit()
        total += cursor.rowcount
    cursor.close()
    print(f"  [device_id={device_id}] 插入 {total} 条行为事件记录")


# ══════════════════════════════════════════════════════
#  查询验证
# ══════════════════════════════════════════════════════

def query_summary():
    conn   = get_conn(database=BEH_SCHEMA)
    cursor = conn.cursor()

    print("\n======= 行为数据概况 =======")
    for sc in SCENARIOS:
        t = tbl(sc['device_id'])
        try:
            cursor.execute(f"""
                SELECT
                    COUNT(*)                                               AS total,
                    SUM(CASE WHEN behavior=1 THEN 1 ELSE 0 END)           AS move_cnt,
                    SUM(CASE WHEN behavior=2 THEN 1 ELSE 0 END)           AS sleep_cnt,
                    SUM(CASE WHEN behavior=3 THEN 1 ELSE 0 END)           AS scratch_cnt,
                    ROUND(AVG(confidence), 3)                             AS avg_conf,
                    DATE_FORMAT(FROM_UNIXTIME(MIN(ts_start)/1000), '%Y-%m-%d %H:%i') AS earliest,
                    DATE_FORMAT(FROM_UNIXTIME(MAX(ts_end)/1000),   '%Y-%m-%d %H:%i') AS latest
                FROM {t}
            """)
            row = cursor.fetchone()
            print(f"  {sc['sn']:12s} (id={sc['device_id']})  总={int(row['total']):6d}  "
                  f"运动={row['move_cnt']}  睡眠={row['sleep_cnt']}  抓挠={row['scratch_cnt']}  "
                  f"均置信={row['avg_conf']}  {row['earliest']} ~ {row['latest']}")
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

    conn = get_conn(database=BEH_SCHEMA)

    print("\n=== 第二步：建表 ===")
    for sc in SCENARIOS:
        create_table(conn, sc['device_id'])

    print("\n=== 第三步：生成并插入数据 ===")
    for idx, sc in enumerate(SCENARIOS):
        rows = build_scenario_rows(sc, seed=42 + idx)
        print(f"  [{sc['sn']}] 生成 {len(rows)} 条事件，开始插入...")
        insert_rows(conn, sc['device_id'], rows)

    conn.close()

    print("\n=== 第四步：查询验证 ===")
    query_summary()

    print("\n[完成] 行为数据库写入完毕！")


if __name__ == "__main__":
    main()
