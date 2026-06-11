"""
环境传感器数据库（MySQL）
=====================
数据库: pet_dog_environment
每个设备独立一张表: d_{device_id}

每行 = 一天一条传感器采样记录
  ts           当天 UTC 零点 ms
  env_temp     环境温度 °C
  env_humidity 环境湿度 %
  neck_temp    脖颈温度 °C（炎症期偏高）
  local_date   日期字符串 YYYY-MM-DD
  user_timezone 时区字符串
  created_at   创建时间 ms
  updated_at   更新时间 ms
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
ENV_SCHEMA  = "pet_dog_environment"

DAYS       = 180
START_DATE = date(2024, 1, 1)

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


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def tbl(device_id: int) -> str:
    return f"`{ENV_SCHEMA}`.`d_{device_id}`"


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


# ══════════════════════════════════════════════════════
#  环境数据生成
# ══════════════════════════════════════════════════════

def build_env_rows(sc: dict, seed: int = 42) -> list:
    np.random.seed(seed)
    gap_map    = build_gap_map(sc['gaps'])
    temp_shift = sc.get('temp_shift')
    rows       = []
    now        = now_ms()

    for i in range(DAYS):
        d  = START_DATE + timedelta(days=i)
        ts = to_ts(d)
        local_date = d.strftime('%Y-%m-%d')

        env_temp = round(float(_temperature[i]), 1)
        env_humi = round(float(_humidity[i]), 1)

        if temp_shift and i >= temp_shift[0]:
            env_temp = round(env_temp + temp_shift[1], 1)

        if i in gap_map:
            rows.append((ts, env_temp, env_humi, None, local_date, 'UTC', now, now))
            continue

        sick = is_sick_day(i, sc)
        if sick:
            neck_temp = round(38.5 + np.random.uniform(0.0, 0.8), 2)
        else:
            neck_temp = round(37.5 + np.random.uniform(-0.3, 0.3), 2)

        rows.append((ts, env_temp, env_humi, neck_temp, local_date, 'UTC', now, now))

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
    conn   = get_conn(database=None)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{ENV_SCHEMA}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[OK] 模式 {ENV_SCHEMA} 已就绪")


def create_table(conn, device_id: int):
    t = tbl(device_id)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {t} (
          ts            BIGINT       NOT NULL,
          env_temp      DECIMAL(5,2) DEFAULT NULL,
          env_humidity  DECIMAL(5,1) DEFAULT NULL,
          neck_temp     DECIMAL(5,2) DEFAULT NULL,
          local_date    VARCHAR(12)  DEFAULT NULL,
          user_timezone VARCHAR(32)  DEFAULT NULL,
          created_at    BIGINT       NOT NULL,
          updated_at    BIGINT       NOT NULL,
          PRIMARY KEY (ts)
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
          (ts, env_temp, env_humidity, neck_temp, local_date, user_timezone, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE updated_at=VALUES(updated_at)
    """
    cursor = conn.cursor()
    cursor.executemany(sql, rows)
    conn.commit()
    print(f"  [device_id={device_id}] 插入 {cursor.rowcount} 条环境数据记录")
    cursor.close()


# ══════════════════════════════════════════════════════
#  查询验证
# ══════════════════════════════════════════════════════

def query_summary():
    conn   = get_conn(database=ENV_SCHEMA)
    cursor = conn.cursor()

    print("\n======= 环境数据概况 =======")
    for sc in SCENARIOS:
        t = tbl(sc['device_id'])
        try:
            cursor.execute(f"""
                SELECT
                    COUNT(*)                                             AS total,
                    SUM(CASE WHEN neck_temp IS NULL THEN 1 ELSE 0 END)  AS gap_days,
                    ROUND(AVG(neck_temp), 2)                            AS avg_neck,
                    ROUND(AVG(env_temp), 1)                             AS avg_env_temp,
                    ROUND(AVG(env_humidity), 1)                         AS avg_humi,
                    DATE_FORMAT(FROM_UNIXTIME(MIN(ts)/1000), '%Y-%m-%d') AS earliest,
                    DATE_FORMAT(FROM_UNIXTIME(MAX(ts)/1000), '%Y-%m-%d') AS latest
                FROM {t}
            """)
            row = cursor.fetchone()
            print(f"  {sc['sn']:12s} (id={sc['device_id']})  总={int(row['total']):3d}天  缺口={row['gap_days']}天  "
                  f"脖颈均温={row['avg_neck']}°C  环境均温={row['avg_env_temp']}°C  "
                  f"均湿={row['avg_humi']}%  {row['earliest']} ~ {row['latest']}")
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

    conn = get_conn(database=ENV_SCHEMA)

    print("\n=== 第二步：建表 ===")
    for sc in SCENARIOS:
        create_table(conn, sc['device_id'])

    print("\n=== 第三步：生成并插入数据 ===")
    for idx, sc in enumerate(SCENARIOS):
        rows = build_env_rows(sc, seed=42 + idx)
        print(f"  [{sc['sn']}] 生成 {len(rows)} 条记录，开始插入...")
        insert_rows(conn, sc['device_id'], rows)

    conn.close()

    print("\n=== 第四步：查询验证 ===")
    query_summary()

    print("\n[完成] 环境数据库写入完毕！")


if __name__ == "__main__":
    main()
