"""
PostgreSQL 应用层种子数据初始化
=====================================
schema: pet_device

表：
  pet_device.user                — 用户账号
  pet_device.device_bind_history — 设备绑定历史

运行：python pg_seed.py
"""

import os
import hashlib
import psycopg2
from datetime import datetime, timezone, timedelta

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

PG_HOST     = os.environ.get("PG_HOST",     "127.0.0.1")
PG_PORT     = int(os.environ.get("PG_PORT", "5432"))
PG_USER     = os.environ.get("PG_USER",     "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "123456")
PG_DB       = os.environ.get("PG_DB",       "pet_collar")

SCHEMA = "pet_device"

# ══════════════════════════════════════════════════════
#  种子数据
# ══════════════════════════════════════════════════════

# 12 个用户，主要欧美时区
USERS = [
    {"id": 1,  "email": "james.wilson@example.com",  "first_name": "James",   "last_name": "Wilson",  "country": "US", "language": "en", "timezone": "America/New_York"},
    {"id": 2,  "email": "emma.johnson@example.com",  "first_name": "Emma",    "last_name": "Johnson", "country": "US", "language": "en", "timezone": "America/Chicago"},
    {"id": 3,  "email": "michael.brown@example.com", "first_name": "Michael", "last_name": "Brown",   "country": "US", "language": "en", "timezone": "America/Los_Angeles"},
    {"id": 4,  "email": "lisa.anderson@example.com", "first_name": "Lisa",    "last_name": "Anderson","country": "US", "language": "en", "timezone": "America/Denver"},
    {"id": 5,  "email": "sarah.davis@example.com",   "first_name": "Sarah",   "last_name": "Davis",   "country": "CA", "language": "en", "timezone": "America/Toronto"},
    {"id": 6,  "email": "oliver.smith@example.com",  "first_name": "Oliver",  "last_name": "Smith",   "country": "GB", "language": "en", "timezone": "Europe/London"},
    {"id": 7,  "email": "thomas.baker@example.com",  "first_name": "Thomas",  "last_name": "Baker",   "country": "GB", "language": "en", "timezone": "Europe/London"},
    {"id": 8,  "email": "sophie.martin@example.com", "first_name": "Sophie",  "last_name": "Martin",  "country": "FR", "language": "fr", "timezone": "Europe/Paris"},
    {"id": 9,  "email": "anna.mueller@example.com",  "first_name": "Anna",    "last_name": "Müller",  "country": "DE", "language": "de", "timezone": "Europe/Berlin"},
    {"id": 10, "email": "luca.rossi@example.com",    "first_name": "Luca",    "last_name": "Rossi",   "country": "IT", "language": "it", "timezone": "Europe/Rome"},
    {"id": 11, "email": "carlos.garcia@example.com", "first_name": "Carlos",  "last_name": "García",  "country": "ES", "language": "es", "timezone": "Europe/Madrid"},
    {"id": 12, "email": "nina.berg@example.com",     "first_name": "Nina",    "last_name": "Berg",    "country": "NL", "language": "nl", "timezone": "Europe/Amsterdam"},
]

# 24 个设备绑定分配（device_id → user_id）
# 每个设备绑定一只宠物，pet_id 与 device_id 一一对应
BINDINGS = [
    # user 1 (US/NY)        — 2 台
    {"device_id": 1,  "user_id": 1,  "pet_id": 1},
    {"device_id": 2,  "user_id": 1,  "pet_id": 2},
    # user 2 (US/Chicago)   — 3 台
    {"device_id": 3,  "user_id": 2,  "pet_id": 3},
    {"device_id": 4,  "user_id": 2,  "pet_id": 4},
    {"device_id": 5,  "user_id": 2,  "pet_id": 5},
    # user 3 (US/LA)        — 2 台
    {"device_id": 6,  "user_id": 3,  "pet_id": 6},
    {"device_id": 7,  "user_id": 3,  "pet_id": 7},
    # user 4 (US/Denver)    — 1 台
    {"device_id": 8,  "user_id": 4,  "pet_id": 8},
    # user 5 (CA/Toronto)   — 2 台
    {"device_id": 9,  "user_id": 5,  "pet_id": 9},
    {"device_id": 10, "user_id": 5,  "pet_id": 10},
    # user 6 (GB/London)    — 3 台
    {"device_id": 11, "user_id": 6,  "pet_id": 11},
    {"device_id": 12, "user_id": 6,  "pet_id": 12},
    {"device_id": 13, "user_id": 6,  "pet_id": 13},
    # user 7 (GB/London)    — 2 台
    {"device_id": 14, "user_id": 7,  "pet_id": 14},
    {"device_id": 15, "user_id": 7,  "pet_id": 15},
    # user 8 (FR/Paris)     — 2 台
    {"device_id": 16, "user_id": 8,  "pet_id": 16},
    {"device_id": 17, "user_id": 8,  "pet_id": 17},
    # user 9 (DE/Berlin)    — 2 台
    {"device_id": 18, "user_id": 9,  "pet_id": 18},
    {"device_id": 19, "user_id": 9,  "pet_id": 19},
    # user 10 (IT/Rome)     — 1 台
    {"device_id": 20, "user_id": 10, "pet_id": 20},
    # user 11 (ES/Madrid)   — 2 台
    {"device_id": 21, "user_id": 11, "pet_id": 21},
    {"device_id": 22, "user_id": 11, "pet_id": 22},
    # user 12 (NL/Amsterdam)— 2 台
    {"device_id": 23, "user_id": 12, "pet_id": 23},
    {"device_id": 24, "user_id": 12, "pet_id": 24},
]

BIND_START = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ══════════════════════════════════════════════════════
#  DDL
# ══════════════════════════════════════════════════════

DDL_SCHEMA = f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"

DDL_USER = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.user (
    id                    BIGSERIAL     PRIMARY KEY,
    email                 VARCHAR(255)  NOT NULL,
    password_hash         VARCHAR(255),
    first_name            VARCHAR(64),
    last_name             VARCHAR(64),
    phone                 VARCHAR(64),
    avatar                VARCHAR(255),
    country               VARCHAR(10),
    language              VARCHAR(10),
    timezone              VARCHAR(32),
    account_status        SMALLINT      NOT NULL DEFAULT 1,
    failed_login_attempts BIGINT        NOT NULL DEFAULT 0,
    last_login_ip         VARCHAR(45),
    last_login_at         TIMESTAMP(3),
    created_at            TIMESTAMP(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    deleted_at            TIMESTAMP(3),
    CONSTRAINT uni_user_email UNIQUE (email)
)
"""

DDL_BIND_HISTORY = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.device_bind_history (
    id                BIGSERIAL    PRIMARY KEY,
    device_id         BIGINT       NOT NULL,
    user_id           BIGINT       NOT NULL,
    pet_id            BIGINT       NOT NULL,
    bind_time         TIMESTAMP(3) NOT NULL,
    unbind_time       TIMESTAMP(3),
    unbind_mode       SMALLINT     DEFAULT 1,
    bind_status       SMALLINT     NOT NULL DEFAULT 1,
    created_at        TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    device_token_hash VARCHAR(64)
)
"""

DDL_INDEXES = [
    f"CREATE INDEX IF NOT EXISTS idx_dbh_device_id   ON {SCHEMA}.device_bind_history (device_id)",
    f"CREATE INDEX IF NOT EXISTS idx_dbh_user_id     ON {SCHEMA}.device_bind_history (user_id)",
    f"CREATE INDEX IF NOT EXISTS idx_dbh_pet_id      ON {SCHEMA}.device_bind_history (pet_id)",
    f"CREATE INDEX IF NOT EXISTS idx_dbh_bind_status ON {SCHEMA}.device_bind_history (bind_status)",
    f"CREATE INDEX IF NOT EXISTS idx_user_deleted_at ON {SCHEMA}.user (deleted_at)",
]


# ══════════════════════════════════════════════════════
#  数据库操作
# ══════════════════════════════════════════════════════

def pg_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER,
        password=PG_PASSWORD, dbname=PG_DB
    )


def init_tables(cur):
    cur.execute(DDL_SCHEMA)
    cur.execute(DDL_USER)
    cur.execute(DDL_BIND_HISTORY)
    for idx in DDL_INDEXES:
        cur.execute(idx)
    print(f"[OK] schema & 表结构已就绪")


def seed_users(cur):
    inserted = 0
    for u in USERS:
        pw_hash = hashlib.sha256(f"demo_{u['email']}".encode()).hexdigest()
        cur.execute(
            f"""
            INSERT INTO {SCHEMA}.user
                (id, email, password_hash, first_name, last_name,
                 country, language, timezone, account_status)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,1)
            ON CONFLICT (email) DO NOTHING
            """,
            (u["id"], u["email"], pw_hash,
             u["first_name"], u["last_name"],
             u["country"], u["language"], u["timezone"]),
        )
        inserted += cur.rowcount
    # 让 BIGSERIAL 从 max(id)+1 开始，避免冲突
    cur.execute(f"SELECT setval('{SCHEMA}.user_id_seq', (SELECT MAX(id) FROM {SCHEMA}.user))")
    print(f"[OK] user: 插入 {inserted} 条（共 {len(USERS)} 条）")


def seed_bindings(cur):
    inserted = 0
    for b in BINDINGS:
        token_hash = hashlib.sha256(f"token_device_{b['device_id']}".encode()).hexdigest()[:64]
        cur.execute(
            f"""
            INSERT INTO {SCHEMA}.device_bind_history
                (device_id, user_id, pet_id, bind_time, bind_status, device_token_hash)
            VALUES (%s,%s,%s,%s,1,%s)
            ON CONFLICT DO NOTHING
            """,
            (b["device_id"], b["user_id"], b["pet_id"],
             BIND_START, token_hash),
        )
        inserted += cur.rowcount
    print(f"[OK] device_bind_history: 插入 {inserted} 条（共 {len(BINDINGS)} 条）")


# ══════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print(f"  schema : {SCHEMA}")
    print(f"  用户数 : {len(USERS)}")
    print(f"  设备数 : {len(BINDINGS)}")
    print("=" * 55)

    conn = pg_conn()
    cur  = conn.cursor()

    print("\n[1] 初始化表结构...")
    init_tables(cur)

    print("\n[2] 写入用户数据...")
    seed_users(cur)

    print("\n[3] 写入设备绑定数据...")
    seed_bindings(cur)

    conn.commit()
    cur.close()
    conn.close()

    print("\n[完成]")
    print()
    print("  用户分布：")
    from collections import Counter
    tz_count = Counter(u["timezone"] for u in USERS)
    for tz, n in sorted(tz_count.items()):
        print(f"    {tz:<30}  {n} 人")
    print()
    print("  设备分布（每用户设备数）：")
    dev_count = Counter(b["user_id"] for b in BINDINGS)
    for uid, n in sorted(dev_count.items()):
        u = next(x for x in USERS if x["id"] == uid)
        print(f"    user_{uid:<2} {u['first_name']:<10} ({u['timezone']:<28})  {n} 台")


if __name__ == "__main__":
    main()
