import mysql.connector
import numpy as np
import math
from datetime import date, timedelta, datetime, timezone

# ======================================
# 数据库连接配置
# ======================================
DB_CONFIG = {
    "host":     "127.0.0.1",
    "port":     3306,
    "user":     "root",
    "password": "123456",
    "database": "pet_device"
}

# ======================================
# 1. 建表
# ======================================
def create_table():
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS `pet_behavior_record` (
          `device_id`       varchar(64)  NOT NULL
                            COMMENT '设备序列号',
          `behavior_type`   tinyint      NOT NULL
                            COMMENT '行为种类（0:未知 1:运动 2:睡眠 3:抓挠）',
          `behavior_detail` varchar(64)  DEFAULT NULL
                            COMMENT '行为细分（第二阶段启用，如 MOVE_WALK / SCRATCH_FRONT_PAW）',
          `start_time`      bigint       NOT NULL
                            COMMENT '行为开始时间 UTC 毫秒时间戳',
          `end_time`        bigint       NOT NULL
                            COMMENT '行为结束时间 UTC 毫秒时间戳',
          `confidence`      decimal(4,2) NOT NULL DEFAULT '0.00'
                            COMMENT '置信度（0.00-1.00）',
          PRIMARY KEY (`device_id`, `start_time`),
          KEY `idx_behavior_type` (`behavior_type`),
          KEY `idx_device_time`   (`device_id`, `start_time`, `end_time`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
          COMMENT='宠物行为推理结果表';
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ 建表成功（或已存在）")


# ======================================
# 2. 工具函数
# ======================================

np.random.seed(42)

# 行为类型
BEHAVIOR_UNKNOWN = 0
BEHAVIOR_MOVE    = 1
BEHAVIOR_SLEEP   = 2
BEHAVIOR_SCRATCH = 3

# 行为细分（第二阶段，当前 None）
BEHAVIOR_DETAIL = {
    BEHAVIOR_MOVE:    ['MOVE_WALK', 'MOVE_RUN', 'MOVE_PLAY'],
    BEHAVIOR_SLEEP:   ['SLEEP_DEEP', 'SLEEP_LIGHT'],
    BEHAVIOR_SCRATCH: ['SCRATCH_FRONT_PAW', 'SCRATCH_HIND_PAW', 'SCRATCH_FACE'],
}


def now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def date_to_ts(d: date) -> int:
    """date → 当天 UTC 零点毫秒时间戳"""
    dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def safe_conf(v: float) -> float:
    return round(min(1.0, max(0.0, v)), 2)


# ======================================
# 3. 生成伪数据
# ======================================

def gen_day_behaviors(device_id: str, day: date, scratch_count: int):
    """
    为某只狗某天生成一天的行为记录
    策略：
      - 全天 86400 秒按时段分配行为
      - 凌晨 00:00-07:00  以睡眠为主
      - 上午 07:00-12:00  运动 + 少量抓挠
      - 下午 12:00-14:00  午睡
      - 下午 14:00-20:00  运动 + 抓挠
      - 晚上 20:00-24:00  睡眠为主
      - scratch_count 控制当天抓挠次数
    """
    day_ts  = date_to_ts(day)   # 当天零点 UTC ms
    records = []

    # 时间段定义（秒偏移，行为权重）
    # (start_sec, end_sec, sleep_w, move_w, scratch_slots)
    segments = [
        (0,      7*3600,  0.85, 0.15, 0),           # 凌晨：主要睡眠
        (7*3600, 12*3600, 0.10, 0.85, scratch_count // 3),   # 上午：运动
        (12*3600,14*3600, 0.80, 0.20, 0),            # 午睡
        (14*3600,20*3600, 0.10, 0.80, scratch_count - scratch_count // 3),  # 下午：运动+抓挠
        (20*3600,24*3600, 0.75, 0.25, 0),            # 晚上：睡眠
    ]

    cursor_sec = 0  # 当前时间指针（秒）

    for seg_start, seg_end, sleep_w, move_w, n_scratch in segments:
        cursor_sec = seg_start
        seg_dur    = seg_end - seg_start

        # 在本时段内随机插入 n_scratch 次抓挠
        scratch_times = sorted(
            np.random.randint(seg_start, seg_end, n_scratch).tolist()
        ) if n_scratch > 0 else []

        scratch_idx = 0

        while cursor_sec < seg_end:
            # 判断是否在抓挠时间点
            if scratch_idx < len(scratch_times) and \
               cursor_sec >= scratch_times[scratch_idx]:

                # 抓挠行为：持续 1-8 秒
                duration = int(np.random.uniform(1000, 8000))
                s_ts = day_ts + scratch_times[scratch_idx] * 1000
                e_ts = s_ts + duration
                conf = safe_conf(np.random.normal(0.88, 0.06))

                records.append((
                    device_id,
                    BEHAVIOR_SCRATCH,
                    None,   # behavior_detail 第二阶段启用
                    s_ts,
                    e_ts,
                    conf,
                ))
                cursor_sec = scratch_times[scratch_idx] + duration // 1000 + 1
                scratch_idx += 1

            else:
                # 随机选睡眠或运动
                btype = BEHAVIOR_SLEEP \
                        if np.random.random() < sleep_w \
                        else BEHAVIOR_MOVE

                # 持续时长
                if btype == BEHAVIOR_SLEEP:
                    duration_sec = int(np.random.uniform(600, 3600))   # 10min-1h
                else:
                    duration_sec = int(np.random.uniform(60, 900))     # 1min-15min

                # 不超过本时段
                duration_sec = min(duration_sec, seg_end - cursor_sec)
                if duration_sec <= 0:
                    break

                s_ts = day_ts + cursor_sec * 1000
                e_ts = s_ts   + duration_sec * 1000
                conf = safe_conf(np.random.normal(0.85, 0.07))

                records.append((
                    device_id,
                    btype,
                    None,
                    s_ts,
                    e_ts,
                    conf,
                ))
                cursor_sec += duration_sec

    return records


def build_rows():
    """
    四只设备，对应皮肤健康表的四个场景，天数一致
    DEV_001_NORMAL  30天  正常抓挠 8-12次/天
    DEV_002_SICK    35天  第16-28天抓挠暴增 18-25次
    DEV_003_SEASON  25天  夏天，抓挠略多 12-18次
    DEV_004_GAP     10天  第4-6天没数据（缺口）
    """
    all_rows = []

    # DEV_001_NORMAL：正常
    for i in range(30):
        d       = date(2026, 1, 1) + timedelta(days=i)
        n_scratch = max(0, int(np.random.normal(10, 2)))
        all_rows += gen_day_behaviors('DEV_001_NORMAL', d, n_scratch)

    # DEV_002_SICK：皮肤病
    for i in range(35):
        d = date(2026, 1, 5) + timedelta(days=i)
        if   i < 15: n_scratch = max(0, int(np.random.normal(9,  2)))
        elif i < 28: n_scratch = max(0, int(np.random.normal(9 + (i-14)*1.2, 2.5)))
        else:        n_scratch = max(0, int(np.random.normal(max(9, 22-(i-27)*2.5), 2)))
        all_rows += gen_day_behaviors('DEV_002_SICK', d, n_scratch)

    # DEV_003_SEASON：夏天季节性
    for i in range(25):
        d         = date(2026, 6, 15) + timedelta(days=i)
        doy       = d.timetuple().tm_yday
        base_temp = 22 + 13 * math.sin((doy - 80) / 365 * 2 * math.pi)
        n_scratch = max(0, int(np.random.normal(10 + 0.25 * (base_temp - 20), 2)))
        all_rows += gen_day_behaviors('DEV_003_SEASON', d, n_scratch)

    # DEV_004_GAP：带缺口（第4-6天跳过）
    gap_days = {3, 4, 5}
    for i in range(10):
        if i in gap_days:
            continue   # 缺口天：没有行为数据
        d         = date(2026, 2, 1) + timedelta(days=i)
        n_scratch = max(0, int(np.random.normal(11, 2)))
        all_rows += gen_day_behaviors('DEV_004_GAP', d, n_scratch)

    return all_rows


# ======================================
# 4. 插入数据
# ======================================
def insert_data(rows):
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    sql = """
        INSERT IGNORE INTO `pet_behavior_record`
          (`device_id`, `behavior_type`, `behavior_detail`,
           `start_time`, `end_time`, `confidence`)
        VALUES (%s, %s, %s, %s, %s, %s)
    """

    # 分批插入，避免单次太大
    batch_size = 500
    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i: i + batch_size]
        cursor.executemany(sql, batch)
        conn.commit()
        total += cursor.rowcount

    print(f"✅ 插入成功，共 {total} 条记录")
    cursor.close()
    conn.close()


# ======================================
# 5. 查询验证
# ======================================
def query_data():
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("\n======= 各设备行为记录概况 =======")
    cursor.execute("""
        SELECT
            device_id,
            COUNT(*)                                      AS 总记录数,
            SUM(behavior_type = 1)                        AS 运动次数,
            SUM(behavior_type = 2)                        AS 睡眠次数,
            SUM(behavior_type = 3)                        AS 抓挠次数,
            ROUND(AVG(CASE WHEN behavior_type = 3
                      THEN (end_time - start_time) END) / 1000, 1)
                                                          AS 抓挠平均时长秒,
            ROUND(AVG(confidence), 2)                     AS 平均置信度,
            FROM_UNIXTIME(MIN(start_time) / 1000)         AS 最早记录,
            FROM_UNIXTIME(MAX(end_time)   / 1000)         AS 最晚记录
        FROM pet_behavior_record
        GROUP BY device_id
        ORDER BY device_id
    """)
    for row in cursor.fetchall():
        print(row)

    print("\n======= DEV_002_SICK 每日抓挠次数（验证发病趋势）=======")
    cursor.execute("""
        SELECT
            DATE(FROM_UNIXTIME(start_time / 1000))  AS 日期,
            COUNT(*)                                 AS 当日抓挠次数,
            ROUND(AVG(end_time - start_time) / 1000, 1) AS 平均时长秒
        FROM pet_behavior_record
        WHERE device_id = 'DEV_002_SICK'
          AND behavior_type = 3
        GROUP BY DATE(FROM_UNIXTIME(start_time / 1000))
        ORDER BY 日期
    """)
    for row in cursor.fetchall():
        print(row)

    print("\n======= 抓挠记录明细（DEV_001 前10条）=======")
    cursor.execute("""
        SELECT
            device_id,
            FROM_UNIXTIME(start_time / 1000)            AS 开始时间,
            FROM_UNIXTIME(end_time   / 1000)            AS 结束时间,
            ROUND((end_time - start_time) / 1000, 2)    AS 持续秒,
            confidence
        FROM pet_behavior_record
        WHERE device_id = 'DEV_001_NORMAL'
          AND behavior_type = 3
        ORDER BY start_time
        LIMIT 10
    """)
    for row in cursor.fetchall():
        print(row)

    print("\n======= DEV_004_GAP 缺口验证（第4-6天无数据）=======")
    cursor.execute("""
        SELECT
            DATE(FROM_UNIXTIME(start_time / 1000)) AS 日期,
            COUNT(*)                                AS 记录数
        FROM pet_behavior_record
        WHERE device_id = 'DEV_004_GAP'
        GROUP BY DATE(FROM_UNIXTIME(start_time / 1000))
        ORDER BY 日期
    """)
    for row in cursor.fetchall():
        print(row)

    cursor.close()
    conn.close()


# ======================================
# 主运行
# ======================================
if __name__ == "__main__":
    print("=== 第一步：建表 ===")
    create_table()

    print("\n=== 第二步：生成并插入数据 ===")
    rows = build_rows()
    print(f"生成 {len(rows)} 条记录，开始插入...")
    insert_data(rows)

    print("\n=== 第三步：查询验证 ===")
    query_data()

    print("\n🎉 完成！")