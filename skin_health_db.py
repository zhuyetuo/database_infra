import pymysql
import pymysql.cursors
import numpy as np
import math
from datetime import date, timedelta, datetime, timezone

# ======================================
# 数据库连接配置
# ======================================
MYSQL_HOST     = "127.0.0.1"
MYSQL_PORT     = 3306
MYSQL_USER     = "appuser"
MYSQL_PASSWORD = "123456"
MYSQL_DB       = "pet_collar"
MYSQL_SCHEMA   = "pet_device"

# ======================================
# 1. 建表
# ======================================
def get_conn():
    return pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, database=MYSQL_DB,
        cursorclass=pymysql.cursors.DictCursor,
        charset="utf8mb4",
    )


def create_table():
    conn   = get_conn()
    cursor = conn.cursor()

    cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_SCHEMA}` CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci")

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS `{MYSQL_SCHEMA}`.`pet_skin_health_daily` (
          device_id           VARCHAR(64)   NOT NULL,
          stat_date_ts        BIGINT        NOT NULL,
          scratch_count       SMALLINT      NOT NULL DEFAULT 0,
          scratch_duration    INT           NOT NULL DEFAULT 0,
          scratch_avg_dur     INT           NOT NULL DEFAULT 0,
          scratch_max_dur     INT           NOT NULL DEFAULT 0,
          night_scratch_count SMALLINT      NOT NULL DEFAULT 0,
          avg_temperature     DECIMAL(4,1)  DEFAULT NULL,
          avg_humidity        DECIMAL(4,1)  DEFAULT NULL,
          baseline_mean       DECIMAL(6,2)  DEFAULT NULL,
          baseline_std        DECIMAL(6,2)  DEFAULT NULL,
          temp_coef           DECIMAL(5,3)  DEFAULT NULL,
          temp_effect         DECIMAL(5,2)  DEFAULT NULL,
          zscore              DECIMAL(6,2)  DEFAULT NULL,
          avg_zscore          DECIMAL(6,2)  DEFAULT NULL,
          consec_abnormal     SMALLINT      NOT NULL DEFAULT 0,
          eval_phase          SMALLINT      NOT NULL DEFAULT 0,
          threshold_z         DECIMAL(4,2)  DEFAULT NULL,
          threshold_consec    SMALLINT      DEFAULT NULL,
          threshold_avgz      DECIMAL(4,2)  DEFAULT NULL,
          valid_days          SMALLINT      NOT NULL DEFAULT 0,
          is_abnormal         SMALLINT      NOT NULL DEFAULT 0,
          alert_triggered     SMALLINT      NOT NULL DEFAULT 0,
          alert_reason        VARCHAR(128)  DEFAULT NULL,
          data_quality        SMALLINT      NOT NULL DEFAULT 0,
          wear_minutes        SMALLINT      NOT NULL DEFAULT 0,
          created_at          BIGINT        NOT NULL,
          updated_at          BIGINT        NOT NULL,
          PRIMARY KEY (device_id, stat_date_ts),
          INDEX idx_skin_alert    (device_id, alert_triggered, stat_date_ts),
          INDEX idx_skin_abnormal (device_id, is_abnormal, stat_date_ts),
          INDEX idx_skin_quality  (device_id, data_quality, stat_date_ts)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("建表成功（或已存在）")


# ======================================
# 2. 工具函数
# ======================================

np.random.seed(42)

WARMUP   = 3
MIN_STD  = 2.0
NORMAL_W = 0.05
ABNORM_W = 0.01


def to_ts(d: date) -> int:
    """date → 当天 UTC 零点毫秒时间戳"""
    dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def now_ts() -> int:
    """当前 UTC 毫秒时间戳"""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def get_thresholds(valid_days):
    if valid_days < 1:     return None, None, None
    elif valid_days <= 11: return 4.0, 5, 5.0
    elif valid_days <= 27: return 3.5, 4, 4.5
    else:                  return 2.5, 3, 3.5


def get_phase(valid_days):
    if valid_days == 0:    return 0
    elif valid_days <= 11: return 1
    elif valid_days <= 27: return 2
    else:                  return 3


def get_temp(d: date) -> float:
    doy  = d.timetuple().tm_yday
    base = 22 + 13 * math.sin((doy - 80) / 365 * 2 * math.pi)
    return round(base + np.random.normal(0, 1.5), 1)


def get_humi(d: date) -> float:
    doy  = d.timetuple().tm_yday
    base = 65 + 15 * math.sin((doy - 80) / 365 * 2 * math.pi)
    return round(base + np.random.normal(0, 3), 1)


def safe(val):
    """None 或 NaN 统一返回 None"""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


# ======================================
# 3. 生成伪数据
# ======================================

def build_rows():
    rows = []
    now  = now_ts()

    # ── DEV_001_NORMAL：完全正常（30天）────────────────
    sn = 'DEV_001_NORMAL'
    mean_cur = 10.0; std_cur = 2.0
    valid_days = 0;  consec = 0

    for i in range(30):
        d     = date(2026, 1, 1) + timedelta(days=i)
        ts    = to_ts(d)
        temp  = get_temp(d)
        humi  = get_humi(d)
        count = max(0, int(np.random.normal(10, 2)))
        d_avg = max(500, int(np.random.normal(4000, 800)))
        d_tot = count * d_avg
        d_max = d_avg + int(np.random.normal(1000, 300))
        night = max(0, int(count * np.random.uniform(0.05, 0.15)))

        in_warmup = (i < WARMUP)
        if not in_warmup:
            valid_days += 1
            tz, tc, ta = get_thresholds(valid_days)
            coef       = 0.08 if valid_days >= 20 else 0.0
            t_eff      = round(coef * (temp - 20), 2)
            bl_mean    = round(mean_cur, 2)
            bl_std     = round(std_cur, 2)
            zs         = round(((count - mean_cur) - t_eff) / std_cur, 2)
            is_abn     = zs > tz
            if is_abn:
                consec += 1
                mean_cur = mean_cur * (1 - ABNORM_W) + count * ABNORM_W
            else:
                consec = 0
                mean_cur = mean_cur * (1 - NORMAL_W) + count * NORMAL_W
            avg_z  = round(zs * 0.6 + np.random.normal(0, 0.3), 2)
            alert  = (consec >= tc) and (avg_z >= ta)
            reason = f'连续{consec}天z>{tz}，均值z={avg_z}' if alert else None
        else:
            tz=tc=ta=coef=t_eff=bl_mean=bl_std=zs=avg_z=None
            is_abn=alert=False; consec=0; reason=None

        rows.append((
            sn, ts,
            count, d_tot, d_avg, d_max, night,
            temp, humi,
            safe(bl_mean), safe(bl_std), safe(coef), safe(t_eff),
            safe(zs), safe(avg_z), consec,
            get_phase(valid_days), safe(tz), safe(tc), safe(ta),
            valid_days, int(is_abn), int(alert), reason,
            0, int(np.random.uniform(1380, 1440)),
            now, now
        ))

    # ── DEV_002_SICK：皮肤病发作后康复（35天）──────────
    sn = 'DEV_002_SICK'
    mean_cur = 9.0; std_cur = 2.0
    valid_days = 0;  consec = 0

    for i in range(35):
        d     = date(2026, 1, 5) + timedelta(days=i)
        ts    = to_ts(d)
        temp  = get_temp(d)
        humi  = get_humi(d)

        if   i < 15: true_m = 9.0
        elif i < 28: true_m = 9.0 + (i - 14) * 1.2
        else:        true_m = max(9.0, 22 - (i - 27) * 2.5)

        count = max(0, int(np.random.normal(true_m, 2.5)))
        d_avg = max(500, int(np.random.normal(
                    4500 if 15 <= i < 28 else 3800, 800)))
        d_tot = count * d_avg
        d_max = d_avg + int(np.random.normal(1200, 400))
        night = max(0, int(count * (0.25 if 15 <= i < 28 else 0.08)))

        in_warmup = (i < WARMUP)
        if not in_warmup:
            valid_days += 1
            tz, tc, ta = get_thresholds(valid_days)
            coef       = 0.10 if valid_days >= 20 else 0.0
            t_eff      = round(coef * (temp - 20), 2)
            bl_mean    = round(mean_cur, 2)
            bl_std     = round(std_cur, 2)
            zs         = round(((count - mean_cur) - t_eff) / max(std_cur, MIN_STD), 2)
            is_abn     = zs > tz
            if is_abn:
                consec += 1
                mean_cur = mean_cur * (1 - ABNORM_W) + count * ABNORM_W
            else:
                consec = 0
                mean_cur = mean_cur * (1 - NORMAL_W) + count * NORMAL_W
            avg_z  = round(zs * 0.7 + np.random.normal(0, 0.2), 2)
            alert  = (consec >= tc) and (avg_z >= ta)
            reason = f'连续{consec}天z>{tz}，均值z={avg_z}，抓挠{count}次/天' \
                     if alert else None
        else:
            tz=tc=ta=coef=t_eff=bl_mean=bl_std=zs=avg_z=None
            is_abn=alert=False; consec=0; reason=None

        rows.append((
            sn, ts,
            count, d_tot, d_avg, d_max, night,
            temp, humi,
            safe(bl_mean), safe(bl_std), safe(coef), safe(t_eff),
            safe(zs), safe(avg_z), consec,
            get_phase(valid_days), safe(tz), safe(tc), safe(ta),
            valid_days, int(is_abn), int(alert), reason,
            0, int(np.random.uniform(1350, 1440)),
            now, now
        ))

    # ── DEV_003_SEASON：季节性升高（25天，夏天）─────────
    sn = 'DEV_003_SEASON'
    mean_cur = 10.0; std_cur = 2.0
    valid_days = 0;  consec = 0

    for i in range(25):
        d     = date(2026, 6, 15) + timedelta(days=i)
        ts    = to_ts(d)
        temp  = get_temp(d)
        humi  = get_humi(d)
        count = max(0, int(np.random.normal(10 + 0.25 * (temp - 20), 2)))
        d_avg = max(500, int(np.random.normal(3800, 600)))
        d_tot = count * d_avg
        d_max = d_avg + int(np.random.normal(900, 300))
        night = max(0, int(count * 0.07))

        in_warmup = (i < WARMUP)
        if not in_warmup:
            valid_days   += 1
            tz, tc, ta    = get_thresholds(valid_days)
            learned_coef  = min(0.22, valid_days * 0.01) \
                            if valid_days >= 20 else 0.0
            t_eff         = round(learned_coef * (temp - 20), 2)
            bl_mean       = round(mean_cur, 2)
            bl_std        = round(std_cur, 2)
            zs            = round(((count - mean_cur) - t_eff) / max(std_cur, MIN_STD), 2)
            is_abn        = zs > tz
            if is_abn:
                consec += 1
                mean_cur = mean_cur * (1 - ABNORM_W) + count * ABNORM_W
            else:
                consec = 0
                mean_cur = mean_cur * (1 - NORMAL_W) + count * NORMAL_W
            avg_z  = round(zs * 0.65 + np.random.normal(0, 0.2), 2)
            alert  = (consec >= tc) and (avg_z >= ta)
        else:
            tz=tc=ta=learned_coef=t_eff=bl_mean=bl_std=zs=avg_z=None
            is_abn=alert=False; consec=0

        rows.append((
            sn, ts,
            count, d_tot, d_avg, d_max, night,
            temp, humi,
            safe(bl_mean), safe(bl_std), safe(learned_coef), safe(t_eff),
            safe(zs), safe(avg_z), consec,
            get_phase(valid_days), safe(tz), safe(tc), safe(ta),
            valid_days, int(is_abn), int(alert), None,
            0, int(np.random.uniform(1380, 1440)),
            now, now
        ))

    # ── DEV_004_GAP：带缺口（10天，第4-6天没电）────────
    sn = 'DEV_004_GAP'
    mean_cur = 11.0; std_cur = 2.0
    valid_days = 0;  consec = 0
    gap_days   = {3, 4, 5}
    buffer_day = 6

    for i in range(10):
        d     = date(2026, 2, 1) + timedelta(days=i)
        ts    = to_ts(d)
        temp  = get_temp(d)
        humi  = get_humi(d)

        # 缺口天（没电）
        if i in gap_days:
            rows.append((
                sn, ts,
                0, 0, 0, 0, 0,
                temp, humi,
                round(mean_cur, 2), round(std_cur, 2),
                None, None, None, None, 0,
                get_phase(valid_days), None, None, None,
                valid_days, 0, 0, None,
                2, 0,   # data_quality=2 没电
                now, now
            ))
            continue

        count = max(0, int(np.random.normal(mean_cur, std_cur)))
        d_avg = max(500, int(np.random.normal(4200, 700)))
        d_tot = count * d_avg
        d_max = d_avg + int(np.random.normal(1000, 300))
        night = max(0, int(count * 0.08))

        # 缓冲天
        if i == buffer_day:
            mean_cur = mean_cur * (1 - NORMAL_W) + count * NORMAL_W
            rows.append((
                sn, ts,
                count, d_tot, d_avg, d_max, night,
                temp, humi,
                round(mean_cur, 2), round(std_cur, 2),
                0.0, 0.0, None, None, 0,
                get_phase(valid_days), None, None, None,
                valid_days, 0, 0, None,
                5, int(np.random.uniform(1300, 1440)),  # data_quality=5 缓冲天
                now, now
            ))
            continue

        in_warmup = (i < WARMUP)
        if not in_warmup:
            valid_days += 1
            tz, tc, ta = get_thresholds(valid_days)
            bl_mean    = round(mean_cur, 2)
            bl_std     = round(std_cur, 2)
            zs         = round((count - mean_cur) / max(std_cur, MIN_STD), 2)
            is_abn     = zs > tz
            if is_abn:
                consec += 1
                mean_cur = mean_cur * (1 - ABNORM_W) + count * ABNORM_W
            else:
                consec = 0
                mean_cur = mean_cur * (1 - NORMAL_W) + count * NORMAL_W
            avg_z = round(zs * 0.6, 2)
            alert = (consec >= tc) and (avg_z >= ta)
            coef  = 0.0
            t_eff = 0.0
        else:
            tz=tc=ta=bl_mean=bl_std=zs=avg_z=coef=t_eff=None
            is_abn=alert=False; consec=0

        rows.append((
            sn, ts,
            count, d_tot, d_avg, d_max, night,
            temp, humi,
            safe(bl_mean), safe(bl_std), safe(coef), safe(t_eff),
            safe(zs), safe(avg_z), consec,
            get_phase(valid_days), safe(tz), safe(tc), safe(ta),
            valid_days, int(is_abn), int(alert), None,
            0, int(np.random.uniform(1350, 1440)),
            now, now
        ))

    return rows


# ======================================
# 4. 插入数据
# ======================================
def insert_data(rows):
    conn   = get_conn()
    cursor = conn.cursor()

    sql = f"""
        INSERT INTO `{MYSQL_SCHEMA}`.`pet_skin_health_daily`
          (device_id, stat_date_ts,
           scratch_count, scratch_duration, scratch_avg_dur,
           scratch_max_dur, night_scratch_count,
           avg_temperature, avg_humidity,
           baseline_mean, baseline_std, temp_coef, temp_effect,
           zscore, avg_zscore, consec_abnormal,
           eval_phase, threshold_z, threshold_consec, threshold_avgz,
           valid_days, is_abnormal, alert_triggered, alert_reason,
           data_quality, wear_minutes,
           created_at, updated_at)
        VALUES
          (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
           %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
           %s, %s)
        ON DUPLICATE KEY UPDATE device_id=device_id
    """

    cursor.executemany(sql, rows)
    conn.commit()
    print(f"插入成功，共 {cursor.rowcount} 条记录")
    cursor.close()
    conn.close()


# ======================================
# 5. 查询验证
# ======================================
def query_data():
    conn   = get_conn()
    cursor = conn.cursor()

    print("\n======= 各设备记录概况 =======")
    cursor.execute(f"""
        SELECT
            device_id,
            COUNT(*)                                    AS 总天数,
            SUM(is_abnormal)                            AS 异常天,
            SUM(alert_triggered)                        AS 推送次数,
            SUM(CASE WHEN data_quality > 0 THEN 1 ELSE 0 END) AS 缺口或缓冲天,
            ROUND(AVG(scratch_count), 1)       AS 日均抓挠次数,
            MAX(scratch_count)                          AS 最多单日次数,
            ROUND(MAX(zscore), 2)              AS 最高z_score,
            to_char(to_timestamp(MIN(stat_date_ts)/1000), 'YYYY-MM-DD') AS 最早统计日,
            to_char(to_timestamp(MAX(stat_date_ts)/1000), 'YYYY-MM-DD') AS 最晚统计日
        FROM `{MYSQL_SCHEMA}`.`pet_skin_health_daily`
        GROUP BY device_id
        ORDER BY device_id
    """)
    for row in cursor.fetchall():
        print(row)

    print("\n======= 推送记录明细 =======")
    cursor.execute(f"""
        SELECT
            device_id,
            to_char(to_timestamp(stat_date_ts / 1000), 'YYYY-MM-DD') AS 统计日期,
            scratch_count, zscore, avg_zscore,
            consec_abnormal, alert_reason
        FROM `{MYSQL_SCHEMA}`.`pet_skin_health_daily`
        WHERE alert_triggered = 1
        ORDER BY device_id, stat_date_ts
    """)
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            print(row)
    else:
        print("（暂无推送记录）")

    print("\n======= 缺口与缓冲天 =======")
    cursor.execute(f"""
        SELECT
            device_id,
            to_char(to_timestamp(stat_date_ts / 1000), 'YYYY-MM-DD') AS 统计日期,
            CASE data_quality
                WHEN 1 THEN '未佩戴'
                WHEN 2 THEN '没电'
                WHEN 3 THEN '信号丢失'
                WHEN 4 THEN '松动无效'
                WHEN 5 THEN '缓冲天'
            END AS 原因,
            wear_minutes
        FROM `{MYSQL_SCHEMA}`.`pet_skin_health_daily`
        WHERE data_quality > 0
        ORDER BY device_id, stat_date_ts
    """)
    for row in cursor.fetchall():
        print(row)

    print("\n======= DEV_002_SICK 发病期间详情 =======")
    ts_start = int(datetime(2026, 1, 15, tzinfo=timezone.utc).timestamp() * 1000)
    ts_end   = int(datetime(2026, 2,  5, tzinfo=timezone.utc).timestamp() * 1000)
    cursor.execute(f"""
        SELECT
            to_char(to_timestamp(stat_date_ts / 1000), 'YYYY-MM-DD') AS 统计日期,
            scratch_count, baseline_mean,
            zscore, consec_abnormal, is_abnormal, alert_triggered
        FROM `{MYSQL_SCHEMA}`.`pet_skin_health_daily`
        WHERE device_id = 'DEV_002_SICK'
          AND stat_date_ts BETWEEN %s AND %s
        ORDER BY stat_date_ts
    """, (ts_start, ts_end))
    for row in cursor.fetchall():
        print(row)

    print("\n======= 动态阈值变化（DEV_001 前20天）=======")
    cursor.execute(f"""
        SELECT
            to_char(to_timestamp(stat_date_ts / 1000), 'YYYY-MM-DD') AS 统计日期,
            valid_days, eval_phase,
            threshold_z, threshold_consec, threshold_avgz
        FROM `{MYSQL_SCHEMA}`.`pet_skin_health_daily`
        WHERE device_id = 'DEV_001_NORMAL'
        ORDER BY stat_date_ts
        LIMIT 20
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