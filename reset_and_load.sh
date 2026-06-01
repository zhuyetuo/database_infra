#!/bin/bash
set -e

PG_CMD="docker exec local-postgres16 psql -U postgres -d pet_collar"
TD_URL="http://127.0.0.1:6041/rest/sql"
TD_AUTH="-u root:taosdata"

echo "=== 1. 重启容器 ==="
docker restart local-postgres16 local-tdengine3

echo "=== 2. 等待 PostgreSQL 就绪 ==="
until docker exec local-postgres16 pg_isready -U postgres -d pet_collar > /dev/null 2>&1; do
    echo "  等待中..."
    sleep 2
done
echo "  PostgreSQL 已就绪"

echo "=== 3. 等待 TDengine 就绪 ==="
until curl -sf $TD_URL -d "SELECT 1" $TD_AUTH > /dev/null 2>&1; do
    echo "  等待中..."
    sleep 2
done
echo "  TDengine 已就绪"

echo "=== 4. 删除旧数据 ==="
$PG_CMD -c "
    DROP SCHEMA IF EXISTS pet_dog_environment    CASCADE;
    DROP SCHEMA IF EXISTS pet_dog_behavior       CASCADE;
    DROP SCHEMA IF EXISTS pet_dog_skin_assessment CASCADE;
    DROP SCHEMA IF EXISTS pet_dog_scratch_baseline CASCADE;
    DROP SCHEMA IF EXISTS pet_device             CASCADE;
" 2>/dev/null || true

curl -sf $TD_URL -d "DROP DATABASE IF EXISTS pet_dog_imu" $TD_AUTH > /dev/null || true
echo "  旧数据已清除"

echo "=== 5. 写入模拟数据 ==="
python imu_raw_db.py
python environment_db.py
python behavior_db.py
python skin_assessment_db.py
python scratch_baseline_db.py

echo ""
echo "✅ 全部完成"
