#!/bin/bash
# 重置数据库并重新写入所有模拟数据

set -e

MYSQL_CMD="docker exec local-mysql8 mysql -uroot -p123456"

echo "=== 1. 重启 MySQL 容器 ==="
docker restart local-mysql8

echo "=== 2. 等待 MySQL 就绪 ==="
until $MYSQL_CMD -e "SELECT 1" > /dev/null 2>&1; do
    echo "  等待中..."
    sleep 2
done
echo "  MySQL 已就绪"

echo "=== 3. 删除旧库 ==="
$MYSQL_CMD -e "
    DROP DATABASE IF EXISTS pet_dog_imu;
    DROP DATABASE IF EXISTS pet_dog_environment;
    DROP DATABASE IF EXISTS pet_dog_behavior;
    DROP DATABASE IF EXISTS pet_dog_skin_assessment;
    DROP DATABASE IF EXISTS pet_dog_scratch_baseline;
    DROP DATABASE IF EXISTS pet_dog_skin;
    DROP DATABASE IF EXISTS pet_imu;
    DROP DATABASE IF EXISTS pet_skin_health;
" 2>/dev/null || true
echo "  旧库已清除"

echo "=== 4. 写入模拟数据 ==="
python imu_raw_db.py
python environment_db.py
python behavior_db.py
python skin_assessment_db.py
python scratch_baseline_db.py

echo ""
echo "✅ 全部完成"
