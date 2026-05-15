# mysql_infra

本地 MySQL 8.0 测试环境，仅用于算法数据模拟与验证。

---

## 目录结构

```
mysql_infra/
├── docker-compose.yml      # MySQL 容器配置
├── .env                    # 环境变量（密码、默认库名）
├── mysql-data/             # 数据持久化目录（自动创建）
├── mysql-init/             # 初始化 SQL 脚本目录（可选）
├── imu_raw_db.py           # 创建 pet_dog_imu 库 + 生成 IMU 原始数据
├── skin_assessment_db.py   # 创建 pet_dog_skin 库 + 生成健康评估数据
├── visualize_db.py         # 从数据库读取数据，生成 6 张可视化图表
├── charts/                 # 可视化输出图片
└── ...                     # 旧版单表脚本（skin_health_db.py 等）
```

---

## 快速开始

### 1. 启动容器

```bash
docker-compose up -d
```

等待容器健康检查通过（约 20 秒）：

```bash
docker ps
# STATUS 显示 (healthy) 即可连接
```

### 2. 写入模拟数据

```bash
python imu_raw_db.py           # 写入 pet_dog_imu（IMU 原始事件数据）
python skin_assessment_db.py   # 写入 pet_dog_skin（健康评估 + 基线）
```

### 3. 生成可视化图表

```bash
pip install matplotlib pandas   # 首次需安装
python visualize_db.py          # 图表输出到 ./charts/
```

---

## 容器管理

| 操作 | 命令 |
|------|------|
| 启动容器（后台） | `docker-compose up -d` |
| 查看运行状态 | `docker-compose ps` 或 `docker ps` |
| 查看实时日志 | `docker-compose logs -f` |
| 停止容器（保留数据） | `docker-compose stop` |
| 停止并删除容器 | `docker-compose down` |
| **停止并删除容器 + 数据卷** | `docker-compose down -v` |

---

## 删除数据

### 只清空数据库表（容器保持运行）

```bash
# 进入容器
docker exec -it local-mysql8 mysql -uroot -p123456

# 删除整个数据库（在 MySQL 内执行）
DROP DATABASE pet_dog_imu;
DROP DATABASE pet_dog_skin;
exit
```

重新写入：

```bash
python imu_raw_db.py
python skin_assessment_db.py
```

### 删除容器 + 所有持久化数据（彻底重置）

```bash
docker-compose down          # 停止并删除容器
rm -rf ./mysql-data          # 删除本地数据目录
docker-compose up -d         # 重新启动（全新状态）
```

---

## 数据库说明

### `pet_dog_imu` — IMU 原始数据库

每个设备一张表 `imu_raw_{device_sn}`，记录行为事件级 IMU 特征：

| 字段 | 说明 |
|------|------|
| `ts_start / ts_end` | 行为时间段（UTC ms） |
| `behavior` | 1=运动 2=睡眠 3=抓挠 |
| `ax/ay/az` | 加速度均值（mg） |
| `gx/gy/gz` | 陀螺仪均值（deg/s） |
| `az_rms` | 垂直加速度 RMS（活动强度） |
| `scratch_hz` | 抓挠主频 Hz（仅抓挠事件） |

### `pet_dog_skin` — 皮肤健康评估库

- `skin_daily_{device_sn}`：每日评估结果（z-score、基线、报警等）
- `skin_baseline`：各设备个体基线汇总

### 五个场景 / 八个设备

| 设备 | 场景 | 说明 |
|------|------|------|
| device_sn_1 | A | 完全正常，180 天 |
| device_sn_2 | B | 第 60-80 天皮肤病爆发后康复 |
| device_sn_3 | C | 季节性升高，温度系数 0.25，温度修正后不误报 |
| device_sn_4 | D | 过敏缓慢加重，基线追上后不误报 |
| device_sn_5 | E1 | 忘记佩戴（第 35-37 天缺口） |
| device_sn_6 | E2 | 没电缺口（第 40-44 天）+ 缺口后皮肤病 |
| device_sn_7 | E3 | 信号不稳定（第 20-80 天间歇丢失） |
| device_sn_8 | E4 | 项圈松动（第 50-57 天数据无效） |

---

## 连接信息

```
host:     127.0.0.1
port:     3306
user:     root
password: 123456
```
