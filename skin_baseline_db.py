import mysql.connector
import numpy as np
import math
from datetime import datetime, timezone

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
        CREATE TABLE IF NOT EXISTS `pet_skin_baseline` (
          `device_id`       varchar(64)   NOT NULL
                            COMMENT '设备序列号',
          `baseline_mean`   decimal(6,2)  NOT NULL
                            COMMENT '抓挠次数基线均值（次/天）',
          `baseline_std`    decimal(6,2)  NOT NULL
                            COMMENT '基线标准差',
          `temp_coef`       decimal(5,3)  NOT NULL DEFAULT 0.000
                            COMMENT '温度修正系数（次/°C），不足20天数据时为0',
          `valid_days`      smallint      NOT NULL DEFAULT 0
                            COMMENT '参与计算的有效正常天数',
          `eval_phase`      tinyint       NOT NULL DEFAULT 0
                            COMMENT '当前评估阶段(0:热身期 1:早期4-14天 2:过渡期15-30天 3:稳定期31天+)',
          `confidence`      decimal(4,2)  NOT NULL DEFAULT 0.00
                            COMMENT '基线置信度(0.00-1.00)，valid_days/30线性增长，30天后封顶1.00',
          `last_updated_ts` bigint        NOT NULL
                            COMMENT '基线最后更新时间 UTC 毫秒时间戳',
          `created_at`      bigint        NOT NULL
                            COMMENT '首次创建时间 UTC 毫秒时间戳',
          PRIMARY KEY (`device_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci
          COMMENT='宠物抓挠个体基线表（每只狗一条，滚动更新）';
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ 建表成功（或已存在）")


# ======================================
# 2. 工具函数
# ======================================

np.random.seed(42)


def now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def get_phase(valid_days: int) -> int:
    if valid_days == 0:    return 0
    elif valid_days <= 11: return 1
    elif valid_days <= 27: return 2
    else:                  return 3


def get_confidence(valid_days: int) -> float:
    return round(min(1.0, valid_days / 30), 2)


def calc_temp_coef(counts: list, temps: list) -> float:
    """
    用正常天的原始次数和温度做最小二乘，估计温度系数
    不足20天返回0，系数限制在[0.0, 0.4]
    """
    if len(counts) < 20:
        return 0.0
    x = np.array(temps)
    y = np.array(counts)
    coef = (np.sum((x - x.mean()) * (y - y.mean()))
            / (np.sum((x - x.mean()) ** 2) + 1e-8))
    return round(float(np.clip(coef, 0.0, 0.4)), 3)


# ======================================
# 3. 从 pet_skin_health_daily 读取数据
#    重新计算基线并写入 pet_skin_baseline
# ======================================

def compute_and_upsert_baseline(device_id: str, conn):
    """
    从 pet_skin_health_daily 取这只狗过去30天的正常天数据
    重新计算 baseline_mean / baseline_std / temp_coef
    然后 INSERT ... ON DUPLICATE KEY UPDATE
    """
    cursor = conn.cursor()

    # 取过去30天：data_quality=0（正常天）且 is_abnormal=0（非异常天）
    cursor.execute("""
        SELECT scratch_count, avg_temperature, valid_days
        FROM pet_skin_health_daily
        WHERE device_id = %s
          AND data_quality = 0
          AND is_abnormal  = 0
          AND in_warmup_flag = 0
        ORDER BY stat_date_ts DESC
        LIMIT 30
    """, (device_id,))
    rows = cursor.fetchall()

    if not rows:
        print(f"  [{device_id}] 无有效数据，跳过")
        cursor.close()
        return

    counts = [r[0] for r in rows]
    temps  = [float(r[1]) if r[1] is not None else 20.0 for r in rows]
    max_valid = max(r[2] for r in rows)   # 取最大有效天数作为当前 valid_days

    mean_val   = round(float(np.mean(counts)), 2)
    std_val    = round(max(float(np.std(counts)), 2.0), 2)
    coef_val   = calc_temp_coef(counts, temps)
    valid_days = min(max_valid, 30)
    phase      = get_phase(valid_days)
    confidence = get_confidence(valid_days)
    ts         = now_ts()

    cursor.execute("""
        INSERT INTO `pet_skin_baseline`
          (`device_id`, `baseline_mean`, `baseline_std`, `temp_coef`,
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
    """, (
        device_id, mean_val, std_val, coef_val,
        valid_days, phase, confidence,
        ts, ts
    ))

    conn.commit()
    print(f"  [{device_id}] 基线更新完成  "
          f"均值={mean_val}  标准差={std_val}  "
          f"温度系数={coef_val}  有效天={valid_days}  "
          f"置信度={confidence}  阶段={phase}")
    cursor.close()


# ======================================
# 4. 直接插入伪数据（不依赖 daily 表）
#    用于独立测试本文件
# ======================================

def insert_seed_data():
    """
    四只设备的基线伪数据
    模拟不同积累程度的基线状态
    """
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    ts     = now_ts()

    # (device_id, mean, std, temp_coef, valid_days)
    devices = [
        ('DEV_001_NORMAL', 9.8,  2.1, 0.08,  30),   # 稳定期，基线成熟
        ('DEV_002_SICK',   9.2,  2.3, 0.10,  22),   # 过渡期，刚康复
        ('DEV_003_SEASON', 7.9,  2.0, 0.21,  25),   # 过渡期，温度系数已学习
        ('DEV_004_GAP',    10.5, 2.0, 0.00,   4),   # 早期，缺口导致数据少
    ]

    sql = """
        INSERT INTO `pet_skin_baseline`
          (`device_id`, `baseline_mean`, `baseline_std`, `temp_coef`,
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
    """

    rows = []
    for sn, mean, std, coef, vd in devices:
        rows.append((
            sn, mean, std, coef,
            vd, get_phase(vd), get_confidence(vd),
            ts, ts
        ))

    cursor.executemany(sql, rows)
    conn.commit()
    print(f"✅ 插入/更新成功，共 {cursor.rowcount} 条记录")
    cursor.close()
    conn.close()


# ======================================
# 5. 从 daily 表重新计算并更新基线
#    （需要 pet_skin_health_daily 已有数据）
# ======================================

def refresh_all_baselines():
    """
    遍历 pet_skin_health_daily 里所有设备，
    重新计算基线并写入 pet_skin_baseline
    注意：pet_skin_health_daily 需要有 in_warmup_flag 字段
          如果没有，可以用 valid_days > 0 代替
    """
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 先检查 daily 表是否有数据
    cursor.execute("SELECT COUNT(*) FROM pet_skin_health_daily")
    cnt = cursor.fetchone()[0]
    if cnt == 0:
        print("⚠️  pet_skin_health_daily 暂无数据，跳过重新计算")
        print("    请先运行 skin_health_db.py 插入数据")
        cursor.close()
        conn.close()
        return

    # 检查是否有 in_warmup_flag 字段，没有就用 valid_days > 0 替代
    cursor.execute("SHOW COLUMNS FROM pet_skin_health_daily LIKE 'in_warmup_flag'")
    has_flag = cursor.fetchone() is not None

    # 取所有设备
    cursor.execute("SELECT DISTINCT device_id FROM pet_skin_health_daily")
    device_list = [r[0] for r in cursor.fetchall()]
    cursor.close()

    print(f"找到 {len(device_list)} 个设备，开始重新计算基线...")

    for sn in device_list:
        c2 = conn.cursor()

        # 如果没有 in_warmup_flag，用 valid_days > 0 替代热身期过滤
        if has_flag:
            where_extra = "AND in_warmup_flag = 0"
        else:
            where_extra = "AND valid_days > 0"

        c2.execute(f"""
            SELECT scratch_count, avg_temperature, valid_days
            FROM pet_skin_health_daily
            WHERE device_id = %s
              AND data_quality = 0
              AND is_abnormal  = 0
              {where_extra}
            ORDER BY stat_date_ts DESC
            LIMIT 30
        """, (sn,))
        rows = c2.fetchall()
        c2.close()

        if not rows:
            print(f"  [{sn}] 无有效正常天数据，跳过")
            continue

        counts     = [r[0] for r in rows]
        temps      = [float(r[1]) if r[1] is not None else 20.0 for r in rows]
        max_valid  = max(r[2] for r in rows)

        mean_val   = round(float(np.mean(counts)), 2)
        std_val    = round(max(float(np.std(counts)), 2.0), 2)
        coef_val   = calc_temp_coef(counts, temps)
        valid_days = min(max_valid, 30)
        phase      = get_phase(valid_days)
        confidence = get_confidence(valid_days)
        ts         = now_ts()

        c3 = conn.cursor()
        c3.execute("""
            INSERT INTO `pet_skin_baseline`
              (`device_id`, `baseline_mean`, `baseline_std`, `temp_coef`,
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
        """, (
            sn, mean_val, std_val, coef_val,
            valid_days, phase, confidence,
            ts, ts
        ))
        conn.commit()
        c3.close()

        print(f"  [{sn}] 均值={mean_val}  标准差={std_val}  "
              f"温度系数={coef_val}  有效天={valid_days}  "
              f"置信度={confidence}  阶段={phase}")

    conn.close()


# ======================================
# 6. 查询验证
# ======================================
def query_data():
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("\n======= 所有设备基线状态 =======")
    cursor.execute("""
        SELECT
            device_id,
            baseline_mean,
            baseline_std,
            temp_coef,
            valid_days,
            CASE eval_phase
                WHEN 0 THEN '热身期'
                WHEN 1 THEN '早期(4-14天)'
                WHEN 2 THEN '过渡期(15-30天)'
                WHEN 3 THEN '稳定期(31天+)'
            END                                       AS 评估阶段,
            confidence                                AS 置信度,
            FROM_UNIXTIME(last_updated_ts / 1000)     AS 最后更新时间
        FROM pet_skin_baseline
        ORDER BY device_id
    """)
    for row in cursor.fetchall():
        print(row)

    print("\n======= 置信度说明 =======")
    cursor.execute("""
        SELECT
            device_id,
            valid_days,
            confidence,
            CASE
                WHEN confidence < 0.30 THEN '基线建立中，仅供参考'
                WHEN confidence < 0.70 THEN '基线初步可信'
                WHEN confidence < 1.00 THEN '基线较为可信'
                ELSE                        '基线完全可信'
            END AS 可信程度
        FROM pet_skin_baseline
        ORDER BY confidence DESC
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

    print("\n=== 第二步：插入伪数据 ===")
    insert_seed_data()

    print("\n=== 第三步：尝试从 daily 表重新计算基线 ===")
    print("（需要先运行 skin_health_db.py，若无数据则跳过）")
    refresh_all_baselines()

    print("\n=== 第四步：查询验证 ===")
    query_data()

    print("\n🎉 完成！")