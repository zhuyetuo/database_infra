"""
删表重建脚本
===========
1. 删除所有模拟数据库（MySQL）
2. 删除 TDengine 中 device72 的子表
3. 按顺序重新运行各初始化脚本

运行：python drop_recreate.py
"""

import os
import sys
import subprocess

# ── 加载配置 ──────────────────────────────────────────────────────────────
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

import pymysql
import requests

MYSQL_HOST     = os.environ.get("MYSQL_HOST", "192.168.33.253")
MYSQL_PORT     = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_ROOT_PW  = os.environ.get("MYSQL_ROOT_PASSWORD", "Hicc-mysql-2026")

TD_HOST = os.environ.get("TD_HOST", "192.168.33.253")
TD_PORT = int(os.environ.get("TD_PORT", "6041"))
TD_USER = os.environ.get("TD_USER", "root")
TD_PASS = os.environ.get("TD_PASS", "taosdata")
TD_DB   = os.environ.get("TD_DB",   "hiccpet_device")

# 所有要删除的 MySQL 数据库
MYSQL_DBS_TO_DROP = [
    "pet_device",
    "pet_dog_behavior",
    "pet_dog_environment",
    "pet_dog_skin_assessment",
    "pet_dog_scratch_baseline",
    "pet_dog_wear_event",
    "pet_dog_daily_summary",
]

# TDengine 中 device72 的子表
TD_TABLES_TO_DROP = [
    f"{TD_DB}.imu_ea_cb_3e_cf_00_11",
    f"{TD_DB}.env_ea_cb_3e_cf_00_11",
    f"{TD_DB}.bodytemp_ea_cb_3e_cf_00_11",
]

# ── MySQL 删库 ────────────────────────────────────────────────────────────
def drop_mysql_databases():
    print("\n[1] 删除 MySQL 数据库...")
    conn = pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT,
        user="root", password=MYSQL_ROOT_PW,
        charset="utf8mb4",
    )
    cur = conn.cursor()
    for db in MYSQL_DBS_TO_DROP:
        cur.execute(f"DROP DATABASE IF EXISTS `{db}`")
        print(f"    DROP DATABASE {db}  ✓")
    conn.commit()
    cur.close()
    conn.close()
    print("  MySQL 清理完成")

# ── TDengine 删子表 ───────────────────────────────────────────────────────
def td_exec(sql: str):
    url = f"http://{TD_HOST}:{TD_PORT}/rest/sql"
    resp = requests.post(url, data=sql.encode(), auth=(TD_USER, TD_PASS), timeout=30)
    body = resp.json()
    if body.get("code", 0) != 0:
        msg = body.get("desc", body)
        # 表不存在时忽略
        if "Table does not exist" in str(msg) or "doesn't exist" in str(msg):
            return
        raise RuntimeError(f"TDengine error: {msg}\nSQL: {sql}")

def drop_tdengine_tables():
    print("\n[2] 删除 TDengine device72 子表...")
    for tbl in TD_TABLES_TO_DROP:
        td_exec(f"DROP TABLE IF EXISTS {tbl}")
        print(f"    DROP TABLE {tbl}  ✓")
    print("  TDengine 清理完成")

# ── 重新运行初始化脚本 ────────────────────────────────────────────────────
SCRIPTS = [
    ("mysql_seed.py",          "pet_device 用户/设备绑定"),
    ("behavior_db.py",         "pet_dog_behavior（24设备）"),
    ("environment_db.py",      "pet_dog_environment（24设备）"),
    ("scratch_baseline_db.py", "pet_dog_scratch_baseline（24设备）"),
    ("skin_assessment_db.py",  "pet_dog_skin_assessment（24设备）"),
]

def run_scripts():
    here = os.path.dirname(os.path.abspath(__file__))
    print("\n[3] 重新初始化各脚本...")
    for script, desc in SCRIPTS:
        path = os.path.join(here, script)
        print(f"\n  → {script}  ({desc})")
        result = subprocess.run(
            [sys.executable, path],
            cwd=here,
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"  ✗ {script} 退出码 {result.returncode}，请检查输出")
            sys.exit(1)
        print(f"  ✓ {script} 完成")

# ── 主程序 ────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  删表重建脚本")
    print(f"  MySQL  : {MYSQL_HOST}:{MYSQL_PORT}")
    print(f"  TDengine: {TD_HOST}:{TD_PORT}")
    print("=" * 55)
    print("\n即将删除以下 MySQL 数据库：")
    for db in MYSQL_DBS_TO_DROP:
        print(f"  - {db}")
    print("\n即将删除以下 TDengine 子表：")
    for t in TD_TABLES_TO_DROP:
        print(f"  - {t}")

    ans = input("\n确认删除？(yes/no): ").strip().lower()
    if ans != "yes":
        print("已取消")
        sys.exit(0)

    drop_mysql_databases()
    drop_tdengine_tables()

    print("\n" + "=" * 55)
    print("  删除完成")
    print("=" * 55)

    ans2 = input("\n是否立即重建基础数据（mysql_seed / behavior / environment / scratch_baseline / skin_assessment）？(yes/no): ").strip().lower()
    if ans2 == "yes":
        run_scripts()
        print("\n" + "=" * 55)
        print("  基础数据重建完成")
        print("  请单独运行 device72_db.py 生成 device72 的 181 天数据")
        print("=" * 55)
    else:
        print("\n跳过重建。可手动运行各脚本：")
        for script, desc in SCRIPTS:
            print(f"  python {script}   # {desc}")
        print("  python device72_db.py   # device72 181天数据")

if __name__ == "__main__":
    main()
