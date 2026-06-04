# database_infra

本地 PostgreSQL 16 + TDengine 3 测试环境，仅用于宠物项圈皮肤健康监测算法的数据模拟与验证。

---

## 目录结构

```
database_infra/
├── docker-compose.yml        # PostgreSQL + TDengine 容器配置
├── .env                      # 环境变量（PostgreSQL 密码、默认库名）
├── postgres-data/            # PostgreSQL 数据持久化目录（自动创建）
├── tdengine-data/            # TDengine 数据持久化目录（自动创建）
├── imu_raw_db.py             # 创建 pet_dog_imu 库（TDengine）+ 生成 IMU 原始事件数据
├── environment_db.py         # 创建 pet_dog_environment 模式（PostgreSQL）+ 生成环境传感器数据
├── behavior_db.py            # 创建 pet_dog_behavior 模式（PostgreSQL）+ 生成行为识别事件数据
├── skin_assessment_db.py     # 创建 pet_dog_skin_assessment 模式（PostgreSQL）+ 生成皮肤健康评估数据
├── scratch_baseline_db.py    # 创建 pet_dog_scratch_baseline 模式（PostgreSQL）+ 生成抓挠基线快照数据
├── skin_health_db.py         # 创建 pet_device 模式（PostgreSQL）+ 生成综合皮肤健康日报数据
├── visualize_db.py           # 从数据库读取数据，生成 6 张可视化图表
└── charts/                   # 可视化输出图片目录
```

---

## 快速开始

### 1. 启动容器

```bash
docker-compose up -d
```

等待容器健康检查通过（约 30 秒）：

```bash
docker ps
# STATUS 显示 (healthy) 即可连接
```

### 2. 写入模拟数据

按顺序执行（各脚本可独立运行，互不依赖）：

```bash
python imu_raw_db.py            # 写入 pet_dog_imu（IMU 原始事件，TDengine）
python environment_db.py        # 写入 pet_dog_environment（环境传感器，PostgreSQL）
python behavior_db.py           # 写入 pet_dog_behavior（行为识别事件，PostgreSQL）
python skin_assessment_db.py    # 写入 pet_dog_skin_assessment（皮肤健康评估，PostgreSQL）
python scratch_baseline_db.py   # 写入 pet_dog_scratch_baseline（抓挠基线快照，PostgreSQL）
```

或使用重置脚本（会重启容器并清除旧数据）：

```bash
bash reset_and_load.sh
```

### 3. 生成可视化图表

```bash
pip install matplotlib pandas psycopg2-binary requests   # 首次需安装
python visualize_db.py          # 图表输出到 ./charts/
```

---

## 容器管理

> 新版 Docker 已将 `docker-compose` 合并为 `docker compose`（无连字符），两种写法均可。

### 启动

```bash
docker compose up -d          # 后台启动所有容器
docker compose ps             # 查看运行状态（STATUS = healthy 即可用）
docker compose logs -f        # 实时查看日志（Ctrl+C 退出）
docker compose logs postgres  # 只看 PostgreSQL 日志
docker compose logs tdengine  # 只看 TDengine 日志
```

### 停止

```bash
docker compose stop           # 停止容器，保留数据（下次 up -d 可恢复）
docker compose restart        # 重启所有容器
```

### 删除

```bash
docker compose down           # 停止并删除容器（数据卷保留，数据不丢失）
docker compose down -v        # 停止并删除容器 + 数据卷（数据完全清除）
docker compose down --rmi all # 同上，同时删除镜像
```

### 单独操作某个容器

```bash
docker compose stop postgres
docker compose stop tdengine
docker compose start postgres
docker compose start tdengine
docker compose restart postgres
```

### 其他常用

```bash
docker compose pull           # 拉取最新镜像（不重建容器）
docker compose up -d --build  # 强制重建后启动
```

| 操作 | 数据是否保留 |
|------|-------------|
| `stop` / `start` | ✅ 保留 |
| `down` | ✅ 保留（卷未删除） |
| `down -v` | ❌ 全部清除 |

---

## 清空数据

### TDengine — 删除全部子表数据

```bash
# 逐表清空（保留表结构）
for sn in $(seq 1 24); do
  for suffix in imu env neck; do
    curl -sf http://127.0.0.1:6041/rest/sql \
      -d "DELETE FROM pet_collar_raw.device_sn_${sn}_${suffix}" \
      -u root:taosdata
  done
done
```

```bash
# 或直接删除整个数据库（含所有表）再重建
curl -sf http://127.0.0.1:6041/rest/sql \
  -d 'DROP DATABASE IF EXISTS pet_collar_raw' \
  -u root:taosdata
```

### PostgreSQL — 删除全部子表数据

```bash
# 逐表清空（保留表结构）
docker exec -i local-postgres16 psql -U postgres -d pet_collar <<'EOF'
DO $$ DECLARE t TEXT; BEGIN
  FOR t IN SELECT tablename FROM pg_tables WHERE schemaname = 'pet_dog_behavior'
  LOOP EXECUTE 'TRUNCATE TABLE pet_dog_behavior.' || t; END LOOP;
END $$;
DO $$ DECLARE t TEXT; BEGIN
  FOR t IN SELECT tablename FROM pg_tables WHERE schemaname = 'pet_dog_environment'
  LOOP EXECUTE 'TRUNCATE TABLE pet_dog_environment.' || t; END LOOP;
END $$;
DO $$ DECLARE t TEXT; BEGIN
  FOR t IN SELECT tablename FROM pg_tables WHERE schemaname = 'pet_dog_skin_assessment'
  LOOP EXECUTE 'TRUNCATE TABLE pet_dog_skin_assessment.' || t; END LOOP;
END $$;
DO $$ DECLARE t TEXT; BEGIN
  FOR t IN SELECT tablename FROM pg_tables WHERE schemaname = 'pet_dog_scratch_baseline'
  LOOP EXECUTE 'TRUNCATE TABLE pet_dog_scratch_baseline.' || t; END LOOP;
END $$;
EOF
```

```bash
# 或直接删除整个 schema（含所有表）再重建
docker exec -i local-postgres16 psql -U postgres -d pet_collar <<'EOF'
DROP SCHEMA IF EXISTS pet_dog_behavior CASCADE;
DROP SCHEMA IF EXISTS pet_dog_environment CASCADE;
DROP SCHEMA IF EXISTS pet_dog_skin_assessment CASCADE;
DROP SCHEMA IF EXISTS pet_dog_scratch_baseline CASCADE;
DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA public;
EOF
```

---

## 连接信息

### PostgreSQL

```
host:     127.0.0.1
port:     5432
user:     postgres
password: 123456
database: pet_collar
```

### TDengine

```
host:     127.0.0.1
port:     6041 (REST API)
port:     6030 (native)
user:     root
password: taosdata
database: pet_dog_imu
```

连接 TDengine REST API 示例：

```bash
curl -sf http://127.0.0.1:6041/rest/sql \
  -d 'SHOW DATABASES' \
  -u root:taosdata
```

**TDengine Explorer（Web UI）**

访问地址：`http://<宿主机IP>:6060/explorer`（例：`http://192.168.2.140:6060/explorer`）

> **首次注册须知**：注册过程必须保证 Explorer 可连接互联网，否则无法注册成功。注册成功后可内网使用，无需再连接互联网。后续登录请使用数据库用户名密码登录（默认：user=`root`，password=`taosdata`）。

---

## 数据库说明

### 架构设计

- **TDengine** (`pet_dog_imu`): 存储高频 IMU 6轴传感器原始事件数据，利用时序数据库的超级表机制按设备分子表
- **PostgreSQL** (`pet_collar`): 存储所有经过算法处理的结构化数据，按功能域分模式（schema）

### 1. TDengine — `pet_dog_imu`（IMU 原始事件）

超级表 `imu_events`，每个设备一张子表（`device_sn_1` ~ `device_sn_24`），每行为一次连续行为片段的 IMU 特征摘要。

共 24 张子表，每张表约 2000~4000 行（180 天 × 每天若干事件）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `ts` | TIMESTAMP | 行为开始时间（UTC ms，主键） |
| `ts_end` | BIGINT | 行为结束时间（UTC ms） |
| `ax` | FLOAT | 加速度 X 轴均值（mg） |
| `ay` | FLOAT | 加速度 Y 轴均值（mg） |
| `az` | FLOAT | 加速度 Z 轴均值（mg，重力方向） |
| `gx` | FLOAT | 陀螺仪 X 轴均值（deg/s） |
| `gy` | FLOAT | 陀螺仪 Y 轴均值（deg/s） |
| `gz` | FLOAT | 陀螺仪 Z 轴均值（deg/s） |
| `device_sn` | BINARY(32) | 设备序列号（TAG） |

### 2. PostgreSQL 模式 `pet_dog_environment`（环境传感器）

每个设备一张独立表，每行为一天一条传感器采样记录。

共 24 张表，每张表 180 行（每天一条）。缺口天的 `neck_temp` 为 NULL。

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | BIGSERIAL | 自增主键 |
| `ts` | BIGINT | 采样日 UTC 零点（ms） |
| `neck_temp` | NUMERIC(5,2) | 脖颈温度 °C（炎症期偏高；缺口天为 NULL） |
| `env_temp` | NUMERIC(5,1) | 环境温度 °C（全局共享序列） |
| `env_humidity` | NUMERIC(5,1) | 环境湿度 %（全局共享序列） |

### 3. PostgreSQL 模式 `pet_dog_behavior`（行为识别事件）

每个设备一张独立表，每行为 ML 模型输出的一次行为识别事件。

共 24 张表，每张表约 2000~4000 行。缺口天无数据。

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | BIGSERIAL | 自增主键 |
| `ts_start` | BIGINT | 行为开始时间（UTC ms） |
| `ts_end` | BIGINT | 行为结束时间（UTC ms） |
| `behavior` | SMALLINT | 行为类型：1=运动  2=睡眠  3=抓挠 |
| `duration_sec` | NUMERIC(10,2) | 持续时长（秒） |
| `confidence` | NUMERIC(5,3) | 模型置信度（0.000~1.000） |

### 4. PostgreSQL 模式 `pet_dog_skin_assessment`（皮肤健康评估）

每个设备一张独立表，每行为一天的评估结果，由 EWMA 动态基线 + z-score 算法生成。

共 24 张表，每张表 180 行。

| 字段 | 类型 | 说明 |
|------|------|------|
| `stat_date` | DATE | 统计日期（主键） |
| `scratch_count` | INT | 当日抓挠总次数 |
| `baseline_mean` | DECIMAL(6,2) | 个体动态基线均值（次/天） |
| `baseline_std` | DECIMAL(6,2) | 个体基线标准差 |
| `zscore` | DECIMAL(6,2) | 温度修正后 z-score |
| `avg_zscore` | DECIMAL(6,2) | 近 N 天均值 z-score |
| `consec_abnormal` | INT | 当前连续异常天数 |
| `eval_phase` | SMALLINT | 评估阶段：0=热身期  1=早期  2=过渡期  3=稳定期 |
| `threshold_z` | DECIMAL(4,2) | 当日 z-score 门槛 |
| `threshold_consec` | SMALLINT | 当日连续天数门槛 |
| `is_abnormal` | SMALLINT | 当日是否异常（0=正常  1=异常） |
| `alert_triggered` | SMALLINT | 是否触发报警（0=无  1=报警） |
| `alert_reason` | VARCHAR(256) | 报警原因描述 |
| `data_quality` | SMALLINT | 数据质量：0=正常  1=未佩戴  2=没电  3=信号丢失  4=松动  5=缓冲天 |
| `wear_minutes` | INT | 有效佩戴分钟数 |

### 5. PostgreSQL 模式 `pet_dog_scratch_baseline`（抓挠基线快照）

每个设备一张独立表，每行为当天算法运行后的基线状态快照。缺口天不保存快照。

共 24 张表，每张表约 150~177 行（扣除热身期和缺口天）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `stat_date` | DATE | 统计日期（主键） |
| `baseline_mean` | DECIMAL(6,2) | 基线均值（次/天） |
| `baseline_std` | DECIMAL(6,2) | 基线标准差 |
| `temp_coef` | DECIMAL(5,3) | 温度修正系数（次/°C） |
| `confidence` | DECIMAL(4,2) | 基线置信度（0.00~1.00，有效天/30） |
| `valid_days` | INT | 参与计算的有效正常天数 |

---

## 场景说明

共 24 个设备，覆盖健康状态、设备质量、环境变化、个体差异四大类场景。

### 健康场景（device_sn_1 ~ device_sn_9）

| 设备 | 场景名称 | 说明 |
|------|---------|------|
| device_sn_1 | 完全正常 | 180 天抓挠均值约 10 次/天，基线稳定，无报警 |
| device_sn_2 | 急性皮肤病后康复 | 第 60~80 天发病（均值 30 次/天），之后恢复正常；测试报警触发与康复 |
| device_sn_3 | 慢性皮肤病（不恢复） | 第 60 天起持续高抓挠（28 次/天），全程不恢复；测试持续异常检测 |
| device_sn_4 | 复发（两次发病） | 第 40~55 天、第 120~135 天各发病一次；测试多次发病检测 |
| device_sn_5 | 渐进性过敏（基线追上） | 三阶段渐进升高（10→15→22 次/天），基线逐步跟上；测试慢性过敏识别 |
| device_sn_6 | 食物过敏（突发触发） | 第 90 天突然发病，之后持续高位（25 次/天）；测试突发型报警 |
| device_sn_7 | 跳蚤/螨虫侵扰 | 第 50~80 天极高抓挠（45 次/天），康复后恢复；测试极端异常检测 |
| device_sn_8 | 季节性过敏（高温度系数） | 温度系数 0.35，高温时抓挠自然升高；测试温度修正防误报 |
| device_sn_9 | 术后恢复 | 第 30~90 天活动受限（3 次/天），其余正常；测试低活动期基线学习 |

### 设备/数据质量场景（device_sn_10 ~ device_sn_16）

| 设备 | 场景名称 | 说明 |
|------|---------|------|
| device_sn_10 | 忘记佩戴（短缺口） | 第 35~37 天未佩戴（3 天缺口）；测试短缺口后基线恢复 |
| device_sn_11 | 电池耗尽 | 第 40~44 天没电（5 天缺口）；测试中等缺口处理 |
| device_sn_12 | 长期缺口（>30 天） | 第 30~64 天缺口（35 天）；触发有效天重置，测试基线重建 |
| device_sn_13 | 信号不稳定 | 第 20~80 天间歇性信号丢失；缺口随机分布，测试断续数据处理 |
| device_sn_14 | 松动项圈 | 第 50~57 天数据无效（8 天）；data_quality=4，测试无效数据标记 |
| device_sn_15 | 设备更换 | 第 88~91 天缺口，模拟更换设备；缺口后基线状态重置 |
| device_sn_16 | 传感器漂移 | 第 70~90 天传感器异常偏高（35 次/天），之后恢复；data_quality=3 |

### 环境场景（device_sn_17 ~ device_sn_20）

| 设备 | 场景名称 | 说明 |
|------|---------|------|
| device_sn_17 | 季节转换（明显温度效应） | 温度系数 0.30，整体随季节温度波动；测试高温度系数处理 |
| device_sn_18 | 搬家（环境突变） | 第 60 天起环境温度整体 +5°C；抓挠略升（13 次/天），基线重新适应 |
| device_sn_19 | 出行旅游（短缺口+不同环境） | 第 80~89 天缺口（外出旅行）；测试旅行期间数据中断 |
| device_sn_20 | 高湿度环境 | 整体抓挠基线偏高（14 次/天）；测试个体差异化基线建立 |

### 个体类型场景（device_sn_21 ~ device_sn_24）

| 设备 | 场景名称 | 说明 |
|------|---------|------|
| device_sn_21 | 幼犬（基线建立慢） | 热身期延长至 7 天；高活跃度（15 次/天）；测试幼犬个体化配置 |
| device_sn_22 | 老年犬（低活动） | 极低抓挠基线（5 次/天），温度系数 0.05；测试低活动个体的灵敏度 |
| device_sn_23 | 高活跃度犬 | 高抓挠基线（20 次/天），温度系数 0.12；测试高活跃个体的异常判断 |
| device_sn_24 | 低活跃度犬（敏感） | 极低基线（4 次/天），温度系数 0.08；小波动即可触发 z-score 异常 |

---

## 算法说明

所有评估脚本共享相同的算法逻辑：

- **全局随机种子**：环境温湿度数组使用 `seed=42`；信号缺口生成使用 `seed=7`；每个场景的行为数据使用 `seed=42+场景序号`
- **热身期**：默认前 3 天（device_sn_21 为 7 天），只收集数据，不输出评估结果
- **动态基线**：EWMA 更新，正常天权重 0.05，异常天权重 0.01，标准差保底 2.0
- **温度修正**：积累 20 天数据后启用，系数由线性回归估计，上限 0.4
- **动态阈值**：
  - 早期（第 4~14 天）：z > 4.0，连续 5 天，均值 z > 5.0
  - 过渡期（第 15~30 天）：z > 3.5，连续 4 天，均值 z > 4.5
  - 稳定期（第 31 天+）：z > 2.5，连续 3 天，均值 z > 3.5
- **缺口处理**：缺口期间基线冻结；恢复后第一天为缓冲天（`data_quality=5`）；缺口 ≥ 30 天则重置有效天计数
