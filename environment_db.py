"""
环境传感器数据库（PostgreSQL）
=====================
数据库: pet_collar，模式: pet_dog_environment
每个设备独立一张表: {device_sn}

每行 = 一天一条传感器采样记录
  ts           当天 UTC 零点 ms
  neck_temp    脖颈温度 °C（炎症期偏高）
  env_temp     环境温度 °C（全局共享序列）
  env_humidity 环境湿度 %（全局共享序列）
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
    {'sn': 'device_sn_1',  'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_2',  'phases': [(0, 60, 10.0, 2.0), (60, 80, 30.0, 4.0), (80, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': (60, 80)},
    {'sn': 'device_sn_3',  'phases': [(0, 60, 10.0, 2.0), (60, 180, 28.0, 4.0)], 'tc': 0.10, 'gaps': [], 'sick': (60, 180)},
    {'sn': 'device_sn_4',  'phases': [(0, 40, 10.0, 2.0), (40, 55, 28.0, 4.0), (55, 120, 10.0, 2.0), (120, 135, 30.0, 4.0), (135, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None, 'sick_episodes': [(40, 55), (120, 135)]},
    {'sn': 'device_sn_5',  'phases': [(0, 60, 10.0, 2.0), (60, 120, 15.0, 2.0), (120, 180, 22.0, 3.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_6',  'phases': [(0, 90, 10.0, 2.0), (90, 180, 25.0, 3.0)], 'tc': 0.10, 'gaps': [], 'sick': (90, 180)},
    {'sn': 'device_sn_7',  'phases': [(0, 50, 10.0, 2.0), (50, 80, 45.0, 6.0), (80, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': (50, 80)},
    {'sn': 'device_sn_8',  'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.35, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_9',  'phases': [(0, 30, 10.0, 2.0), (30, 90, 3.0, 1.0), (90, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_10', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(35, 38, 'unworn')], 'sick': None},
    {'sn': 'device_sn_11', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(40, 45, 'battery')], 'sick': None},
    {'sn': 'device_sn_12', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(30, 65, 'battery')], 'sick': None},
    {'sn': 'device_sn_13', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(d, d + 1, 'signal') for d in sorted(_signal_gap_days)], 'sick': None},
    {'sn': 'device_sn_14', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(50, 58, 'loose')], 'sick': None},
    {'sn': 'device_sn_15', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(88, 92, 'battery')], 'sick': None},
    {'sn': 'device_sn_16', 'phases': [(0, 70, 10.0, 2.0), (70, 90, 35.0, 5.0), (90, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None, 'drift_range': (70, 90)},
    {'sn': 'device_sn_17', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.30, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_18', 'phases': [(0, 60, 10.0, 2.0), (60, 180, 13.0, 2.0)], 'tc': 0.15, 'gaps': [], 'sick': None, 'temp_shift': (60, 5.0)},
    {'sn': 'device_sn_19', 'phases': [(0, 180, 10.0, 2.0)], 'tc': 0.10, 'gaps': [(80, 90, 'unworn')], 'sick': None},
    {'sn': 'device_sn_20', 'phases': [(0, 180, 14.0, 2.0)], 'tc': 0.10, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_21', 'phases': [(0, 180, 15.0, 4.0)], 'tc': 0.10, 'gaps': [], 'sick': None, 'warmup': 7},
    {'sn': 'device_sn_22', 'phases': [(0, 180, 5.0, 1.0)],  'tc': 0.05, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_23', 'phases': [(0, 180, 20.0, 3.0)], 'tc': 0.12, 'gaps': [], 'sick': None},
    {'sn': 'device_sn_24', 'phases': [(0, 180, 4.0, 1.0)],  'tc': 0.08, 'gaps': [], 'sick': None},
]


# ══════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════

def to_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)


def tbl(sn: str) -> str:
    return f"{ENV_SCHEMA}.{sn.lower()}"


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

    for i in range(DAYS):
        d  = START_DATE + timedelta(days=i)
        ts = to_ts(d)

        env_temp = round(float(_temperature[i]), 1)
        env_humi = round(float(_humidity[i]), 1)

        if temp_shift and i >= temp_shift[0]:
            env_temp = round(env_temp + temp_shift[1], 1)

        if i in gap_map:
            rows.append((ts, None, env_temp, env_humi))
            continue

        sick = is_sick_day(i, sc)
        if sick:
            neck_temp = round(38.5 + np.random.uniform(0.0, 0.8), 2)
        else:
            neck_temp = round(37.5 + np.random.uniform(-0.3, 0.3), 2)

        rows.append((ts, neck_temp, env_temp, env_humi))

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
    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {ENV_SCHEMA}")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"[OK] 模式 {ENV_SCHEMA} 已就绪")


def create_table(conn, sn: str):
    t = tbl(sn)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {t} (
          id           BIGSERIAL    PRIMARY KEY,
          ts           BIGINT       NOT NULL,
          neck_temp    NUMERIC(5,2) DEFAULT NULL,
          env_temp     NUMERIC(5,1) NOT NULL,
          env_humidity NUMERIC(5,1) NOT NULL,
          UNIQUE (ts)
        )
    """)
    cursor.execute(f"CREATE INDEX IF NOT EXISTS {sn}_idx_ts ON {t} (ts)")
    conn.commit()
    cursor.close()
    print(f"  [OK] 表 {t} 已就绪")


def insert_rows(conn, sn: str, rows: list):
    if not rows:
        print(f"  [{sn}] 无数据，跳过")
        return

    t   = tbl(sn)
    sql = f"""
        INSERT INTO {t} (ts, neck_temp, env_temp, env_humidity)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """
    cursor = conn.cursor()
    cursor.executemany(sql, rows)
    conn.commit()
    print(f"  [{sn}] 插入 {cursor.rowcount} 条环境数据记录")
    cursor.close()


# ══════════════════════════════════════════════════════
#  查询验证
# ══════════════════════════════════════════════════════

def query_summary():
    conn   = get_conn()
    cursor = conn.cursor()

    print("\n======= 环境数据概况 =======")
    for sc in SCENARIOS:
        t = tbl(sc['sn'])
        try:
            cursor.execute(f"""
                SELECT
                    COUNT(*)                                             AS total,
                    SUM(CASE WHEN neck_temp IS NULL THEN 1 ELSE 0 END)  AS gap_days,
                    ROUND(AVG(neck_temp)::numeric, 2)                   AS avg_neck,
                    ROUND(AVG(env_temp)::numeric, 1)                    AS avg_env_temp,
                    ROUND(AVG(env_humidity)::numeric, 1)                AS avg_humi,
                    to_char(to_timestamp(MIN(ts)/1000), 'YYYY-MM-DD')   AS earliest,
                    to_char(to_timestamp(MAX(ts)/1000), 'YYYY-MM-DD')   AS latest
                FROM {t}
            """)
            row = cursor.fetchone()
            print(f"  {sc['sn']:20s}  总={int(row[0]):3d}天  缺口={row[1]}天  "
                  f"脖颈均温={row[2]}°C  环境均温={row[3]}°C  "
                  f"均湿={row[4]}%  {row[5]} ~ {row[6]}")
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
        rows = build_env_rows(sc, seed=42 + idx)
        print(f"  [{sc['sn']}] 生成 {len(rows)} 条记录，开始插入...")
        insert_rows(conn, sc['sn'], rows)

    conn.close()

    print("\n=== 第四步：查询验证 ===")
    query_summary()

    print("\n[完成] 环境数据库写入完毕！")


if __name__ == "__main__":
    main()
