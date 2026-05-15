import mysql.connector
from datetime import datetime

# 数据库连接配置
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "pet_device"
}

# ======================================
# 1. 插入测试数据
# ======================================
def insert_test_data():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 插入设备
    device_id = "test_dev_001"
    try:
        cursor.execute(
            "INSERT INTO device_info (device_id, device_name) VALUES (%s, %s)",
            (device_id, "测试智能项圈")
        )
        print("✅ 设备插入成功")
    except Exception as e:
        print("ℹ️ 设备已存在，跳过插入")

    # 插入推理结果
    cursor.execute(
        "INSERT INTO imu_infer_result (device_id, label, score) VALUES (%s, %s, %s)",
        (device_id, 1, 0.92)
    )
    conn.commit()
    print("✅ 推理结果插入成功")

    cursor.close()
    conn.close()

# ======================================
# 2. 查询数据（验证是否成功）
# ======================================
def query_data():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("\n======= 查询设备列表 =======")
    cursor.execute("SELECT id, device_id, device_name, create_time FROM device_info")
    for row in cursor.fetchall():
        print(row)

    print("\n======= 查询推理结果 =======")
    cursor.execute("SELECT device_id, label, score, create_time FROM imu_infer_result")
    for row in cursor.fetchall():
        print(row)

    cursor.close()
    conn.close()

# ======================================
# 主运行
# ======================================
if __name__ == "__main__":
    print("=== 开始插入 MySQL 测试数据 ===")
    insert_test_data()

    print("\n=== 开始查询 MySQL 数据 ===")
    query_data()
    print("\n🎉 MySQL 测试完成！")