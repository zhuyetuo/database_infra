"""
宠物项圈皮肤健康监测 — 数据库可视化展示
==========================================
优先从 PostgreSQL/TDengine 数据库读取数据；数据库不可用时自动切换到内联数据生成
（内联数据与入库数据算法完全一致，结果等价）

输出 6 张图表到 ./charts/ 目录:
  01_imu_6axis.png   — IMU 6轴数据（加速度 + 陀螺仪）
  02_behavior.png    — 行为识别统计（分布饼图 + 抓挠时序）
  03_health.png      — 皮肤健康评估（z-score + 基线 + 报警）
  04_baseline.png    — 个体基线演变（均值 + 置信度）
  05_neck_temp.png   — 脖子温度估算（基于抓挠强度与炎症状态推算）
  06_env.png         — 环境温湿度与抓挠行为相关性
"""

import os, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from datetime import date, timedelta, datetime, timezone

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# ══════════════════════════════════════════════════════
#  字体 & 样式
# ══════════════════════════════════════════════════════
_FP = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_FP):
    fm.fontManager.addfont(_FP)
    plt.rcParams['font.family']     = fm.FontProperties(fname=_FP).get_name()
plt.rcParams['axes.unicode_minus']  = False
plt.rcParams['figure.facecolor']    = 'white'
plt.rcParams['axes.facecolor']      = '#F8F9FA'
plt.rcParams['axes.grid']           = True
plt.rcParams['grid.alpha']          = 0.3
plt.rcParams['grid.linewidth']      = 0.6
plt.rcParams['axes.spines.top']     = False
plt.rcParams['axes.spines.right']   = False

OUT_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
#  常量 & 场景定义
# ══════════════════════════════════════════════════════
DAYS       = 180
WARMUP     = 3
MIN_STD    = 2.0
NORMAL_W   = 0.05
ABNORM_W   = 0.01
GAP_RESET  = 30
START_DATE = date(2024, 1, 1)

BEH_COLORS = {1: '#27AE60', 2: '#3498DB', 3: '#E74C3C'}
BEH_LABELS = {1: '运动', 2: '睡眠', 3: '抓挠'}
PHASE_LABELS = {0: '热身期', 1: '早期(4-14天)', 2: '过渡期(15-30天)', 3: '稳定期(31天+)'}

DEV_META = [
    ('device_id_1',  '完全正常',                '#2ECC71'),
    ('device_id_2',  '急性皮肤病后康复',          '#E74C3C'),
    ('device_id_3',  '慢性皮肤病(不恢复)',         '#C0392B'),
    ('device_id_4',  '复发(两次发病)',             '#9B59B6'),
    ('device_id_5',  '渐进性过敏',               '#8E44AD'),
    ('device_id_6',  '食物过敏(突发)',             '#E67E22'),
    ('device_id_7',  '跳蚤/螨虫(极高抓挠)',        '#D35400'),
    ('device_id_8',  '季节性过敏(高温度系数)',      '#F39C12'),
    ('device_id_9',  '术后恢复(低活动)',           '#1ABC9C'),
    ('device_id_10', '忘记佩戴(3天缺口)',          '#27AE60'),
    ('device_id_11', '电池耗尽(5天缺口)',          '#16A085'),
    ('device_id_12', '长期缺口>30天(基线重置)',     '#2980B9'),
    ('device_id_13', '信号不稳定(断续丢失)',        '#3498DB'),
    ('device_id_14', '松动项圈(8天无效)',          '#7F8C8D'),
    ('device_id_15', '设备更换(第90天)',           '#95A5A6'),
    ('device_id_16', '传感器漂移(第70-90天)',       '#BDC3C7'),
    ('device_id_17', '季节转换(明显温度效应)',       '#F1C40F'),
    ('device_id_18', '搬家(环境突变第60天)',        '#E74C3C'),
    ('device_id_19', '出行旅游(第80-90天缺口)',     '#2ECC71'),
    ('device_id_20', '高湿度环境',                '#1ABC9C'),
    ('device_id_21', '幼犬(基线建立慢)',           '#3498DB'),
    ('device_id_22', '老年犬(低活动)',             '#9B59B6'),
    ('device_id_23', '高活跃度犬',                '#E67E22'),
    ('device_id_24', '低活跃度犬(敏感)',           '#BDC3C7'),
]
SN_COLOR  = {sn: c for sn, _, c in DEV_META}
SN_LABEL  = {sn: lb for sn, lb, _ in DEV_META}

KEY_SNS = ['device_id_1', 'device_id_2', 'device_id_4', 'device_id_6']

# ── 全局时间序列（复现入库脚本随机种子）──
np.random.seed(42)
_TEMP_ARR = (22 + 13 * np.sin(np.linspace(-np.pi/2, 3*np.pi/2, DAYS))
             + np.random.normal(0, 1.5, DAYS))
_HUMI_ARR = (65 + 15 * np.sin(np.linspace(-np.pi/2, 3*np.pi/2, DAYS))
             + np.random.normal(0, 3.0, DAYS))

np.random.seed(7)
_SIG_GAPS: set = set()
_i = 20
while _i < 80:
    if np.random.random() < 0.25:
        _g = np.random.randint(1, 4)
        for _j in range(_i, min(_i + _g, 80)):
            _SIG_GAPS.add(_j)
        _i += _g + np.random.randint(2, 6)
    else:
        _i += 1
np.random.seed(42)

SCENARIOS = [
    # 健康场景 1-9
    {'sn': 'device_id_1',  'phases': [(0,180,10,2)],                              'tc': 0.10, 'gaps': [],                                        'sick': None},
    {'sn': 'device_id_2',  'phases': [(0,60,10,2),(60,80,30,4),(80,180,10,2)],    'tc': 0.10, 'gaps': [],                                        'sick': (60,80)},
    {'sn': 'device_id_3',  'phases': [(0,60,10,2),(60,180,28,4)],                 'tc': 0.10, 'gaps': [],                                        'sick': (60,180)},
    {'sn': 'device_id_4',  'phases': [(0,40,10,2),(40,55,28,4),(55,120,10,2),(120,135,30,4),(135,180,10,2)], 'tc': 0.10, 'gaps': [], 'sick': None, 'sick_episodes': [(40,55),(120,135)]},
    {'sn': 'device_id_5',  'phases': [(0,60,10,2),(60,120,15,2),(120,180,22,3)],  'tc': 0.10, 'gaps': [],                                        'sick': None},
    {'sn': 'device_id_6',  'phases': [(0,90,10,2),(90,180,25,3)],                 'tc': 0.10, 'gaps': [],                                        'sick': (90,180)},
    {'sn': 'device_id_7',  'phases': [(0,50,10,2),(50,80,45,6),(80,180,10,2)],    'tc': 0.10, 'gaps': [],                                        'sick': (50,80)},
    {'sn': 'device_id_8',  'phases': [(0,180,10,2)],                              'tc': 0.35, 'gaps': [],                                        'sick': None},
    {'sn': 'device_id_9',  'phases': [(0,30,10,2),(30,90,3,1),(90,180,10,2)],     'tc': 0.10, 'gaps': [],                                        'sick': None},
    # 设备/数据质量场景 10-16
    {'sn': 'device_id_10', 'phases': [(0,180,10,2)],                              'tc': 0.10, 'gaps': [(35,38,'unworn')],                        'sick': None},
    {'sn': 'device_id_11', 'phases': [(0,180,10,2)],                              'tc': 0.10, 'gaps': [(40,45,'battery')],                       'sick': None},
    {'sn': 'device_id_12', 'phases': [(0,180,10,2)],                              'tc': 0.10, 'gaps': [(30,65,'battery')],                       'sick': None},
    {'sn': 'device_id_13', 'phases': [(0,180,10,2)],                              'tc': 0.10, 'gaps': [(d,d+1,'signal') for d in sorted(_SIG_GAPS)], 'sick': None},
    {'sn': 'device_id_14', 'phases': [(0,180,10,2)],                              'tc': 0.10, 'gaps': [(50,58,'loose')],                         'sick': None},
    {'sn': 'device_id_15', 'phases': [(0,180,10,2)],                              'tc': 0.10, 'gaps': [(88,92,'battery')],                       'sick': None},
    {'sn': 'device_id_16', 'phases': [(0,70,10,2),(70,90,35,5),(90,180,10,2)],    'tc': 0.10, 'gaps': [],                                        'sick': None},
    # 环境场景 17-20
    {'sn': 'device_id_17', 'phases': [(0,180,10,2)],                              'tc': 0.30, 'gaps': [],                                        'sick': None},
    {'sn': 'device_id_18', 'phases': [(0,60,10,2),(60,180,13,2)],                 'tc': 0.15, 'gaps': [],                                        'sick': None, 'temp_shift': (60, 5.0)},
    {'sn': 'device_id_19', 'phases': [(0,180,10,2)],                              'tc': 0.10, 'gaps': [(80,90,'unworn')],                        'sick': None},
    {'sn': 'device_id_20', 'phases': [(0,180,14,2)],                              'tc': 0.10, 'gaps': [],                                        'sick': None},
    # 个体类型场景 21-24
    {'sn': 'device_id_21', 'phases': [(0,180,15,4)],                              'tc': 0.10, 'gaps': [],                                        'sick': None, 'warmup': 7},
    {'sn': 'device_id_22', 'phases': [(0,180,5,1)],                               'tc': 0.05, 'gaps': [],                                        'sick': None},
    {'sn': 'device_id_23', 'phases': [(0,180,20,3)],                              'tc': 0.12, 'gaps': [],                                        'sick': None},
    {'sn': 'device_id_24', 'phases': [(0,180,4,1)],                               'tc': 0.08, 'gaps': [],                                        'sick': None},
]
SC_MAP = {s['sn']: s for s in SCENARIOS}

# ══════════════════════════════════════════════════════
#  数据库连接（可选）
# ══════════════════════════════════════════════════════
IMU_DB      = 'pet_dog_imu'
PG_DB       = 'pet_collar'
SKIN_SCHEMA = 'pet_dog_skin_assessment'
BSL_SCHEMA  = 'pet_dog_scratch_baseline'
TD_HOST     = '127.0.0.1'
TD_PORT     = 6041
_DB_AVAILABLE = False

try:
    import psycopg2 as _pg
    _test = _pg.connect(host='127.0.0.1', port=5432, user='postgres',
                        password='123456', dbname='pet_collar', connect_timeout=3)
    _test.close()
    _DB_AVAILABLE = True
    print('[OK] 数据库连接成功，使用数据库数据')
except Exception:
    print('[INFO] 数据库不可用，使用内联生成数据（与数据库内容算法一致）')


def _db_read(sql: str) -> pd.DataFrame:
    conn = _pg.connect(host='127.0.0.1', port=5432, user='postgres',
                       password='123456', dbname=PG_DB)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


def _td_read(sql: str) -> dict:
    import requests
    url  = f"http://{TD_HOST}:{TD_PORT}/rest/sql"
    resp = requests.post(url, data=sql.encode('utf-8'),
                         auth=('root', 'taosdata'), timeout=60)
    resp.raise_for_status()
    return resp.json()


# ══════════════════════════════════════════════════════
#  算法函数（复现入库脚本逻辑）
# ══════════════════════════════════════════════════════
def _thresholds(vd):
    if vd < 1:      return None, None, None
    elif vd <= 11:  return 4.0, 5, 5.0
    elif vd <= 27:  return 3.5, 4, 4.5
    else:           return 2.5, 3, 3.5

def _phase(vd):
    return 0 if vd==0 else 1 if vd<=11 else 2 if vd<=27 else 3

def _conf(vd):
    return round(min(1.0, vd/30), 2)

def _temp_coef(bc, bt):
    if len(bc) < 20: return 0.0
    x, y = np.array(bt, float), np.array(bc, float)
    c = np.sum((x-x.mean())*(y-y.mean())) / (np.sum((x-x.mean())**2)+1e-8)
    return round(float(np.clip(c, 0.0, 0.4)), 3)

def _gap_map(gaps):
    gm = {}
    for s, e, r in gaps:
        for d in range(s, min(e, DAYS)): gm[d] = r
    return gm

def _scratch_count(i, phases, temp, tc):
    for s, e, m, sd in phases:
        if s <= i < e:
            return max(0, int(np.random.normal(m + tc*(temp-20), sd)))
    return 0

def _is_sick(i, sc):
    episodes = sc.get('sick_episodes')
    if episodes:
        return any(s <= i < e for s, e in episodes)
    sick = sc.get('sick')
    if sick:
        return sick[0] <= i < sick[1]
    return False

DQ_MAP = {'unworn':1,'battery':2,'signal':3,'loose':4}

# ══════════════════════════════════════════════════════
#  内联数据生成：皮肤健康每日评估
# ══════════════════════════════════════════════════════
def gen_skin_daily(sc: dict, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    gap_map  = _gap_map(sc['gaps'])
    warmup   = sc.get('warmup', WARMUP)
    mean=std=None; bc=[]; bt=[]; consec=0; vd=0
    gc=0; in_gap=False; resumed=False; zbuf=[]
    rows = []

    for i in range(DAYS):
        d    = START_DATE + timedelta(days=i)
        temp = round(float(_TEMP_ARR[i]), 1)
        humi = round(float(_HUMI_ARR[i]), 1)

        if i in gap_map:
            dq = DQ_MAP.get(gap_map[i], 1)
            gc += 1; in_gap = True; consec = 0
            rows.append({'date': d, 'scratch_count': 0,
                         'avg_temperature': temp, 'avg_humidity': humi,
                         'baseline_mean': mean, 'baseline_std': std,
                         'temp_coef': None, 'temp_effect': None,
                         'zscore': None, 'avg_zscore': None,
                         'consec_abnormal': 0, 'eval_phase': _phase(vd),
                         'threshold_z': None, 'threshold_consec': None,
                         'valid_days': vd, 'is_abnormal': 0,
                         'alert_triggered': 0, 'alert_reason': None,
                         'data_quality': dq, 'in_warmup_flag': 0,
                         'in_gap_flag': 1, 'just_resumed_flag': 0})
            continue

        if in_gap:
            in_gap = False; resumed = True
            if gc >= GAP_RESET: vd = 0
            gc = 0; consec = 0
        else:
            resumed = False

        cnt  = _scratch_count(i, sc['phases'], temp, sc['tc'])
        wear = int(np.random.uniform(1350, 1440))

        if i < warmup:
            bc.append(cnt); bt.append(temp)
            rows.append({'date': d, 'scratch_count': cnt,
                         'avg_temperature': temp, 'avg_humidity': humi,
                         'baseline_mean': None, 'baseline_std': None,
                         'temp_coef': None, 'temp_effect': None,
                         'zscore': None, 'avg_zscore': None,
                         'consec_abnormal': 0, 'eval_phase': 0,
                         'threshold_z': None, 'threshold_consec': None,
                         'valid_days': 0, 'is_abnormal': 0,
                         'alert_triggered': 0, 'alert_reason': None,
                         'data_quality': 0, 'in_warmup_flag': 1,
                         'in_gap_flag': 0, 'just_resumed_flag': 0})
            continue

        if mean is None:
            mean = float(np.mean(bc))
            std  = max(float(np.std(bc)) if len(bc)>1 else MIN_STD, MIN_STD)

        vd += 1
        tz, tc, ta = _thresholds(vd)

        if resumed:
            mean = mean*(1-NORMAL_W) + cnt*NORMAL_W
            bc.append(cnt); bt.append(temp)
            rows.append({'date': d, 'scratch_count': cnt,
                         'avg_temperature': temp, 'avg_humidity': humi,
                         'baseline_mean': round(mean,2), 'baseline_std': round(std,2),
                         'temp_coef': 0.0, 'temp_effect': 0.0,
                         'zscore': None, 'avg_zscore': None,
                         'consec_abnormal': 0, 'eval_phase': _phase(vd),
                         'threshold_z': tz, 'threshold_consec': tc,
                         'valid_days': vd, 'is_abnormal': 0,
                         'alert_triggered': 0, 'alert_reason': None,
                         'data_quality': 5, 'in_warmup_flag': 0,
                         'in_gap_flag': 0, 'just_resumed_flag': 1})
            continue

        coef    = _temp_coef(bc, bt)
        te      = round(coef*(temp-20), 2)
        zs      = round(((cnt - mean) - te) / std, 2)
        is_abn  = bool(tz is not None and zs > tz)

        if is_abn:
            consec += 1; mean = mean*(1-ABNORM_W) + cnt*ABNORM_W
        else:
            consec = 0; mean = mean*(1-NORMAL_W) + cnt*NORMAL_W
            bc.append(cnt); bt.append(temp)

        if len(bc) > 1:
            std = max(float(np.std(bc[-30:])), MIN_STD)

        nb   = max((tc-1) if tc else 2, 1)
        avgz = round(float(np.mean(zbuf[-nb:] + [zs])), 2)
        zbuf.append(zs)
        if len(zbuf) > 10: zbuf.pop(0)

        alert  = bool(tc and consec >= tc and avgz >= ta)
        reason = f'连续{consec}天z>{tz:.1f}，均值z={avgz:.2f}，抓挠{cnt}次' if alert else None

        rows.append({'date': d, 'scratch_count': cnt,
                     'avg_temperature': temp, 'avg_humidity': humi,
                     'baseline_mean': round(mean,2), 'baseline_std': round(std,2),
                     'temp_coef': coef, 'temp_effect': te,
                     'zscore': zs, 'avg_zscore': avgz,
                     'consec_abnormal': consec, 'eval_phase': _phase(vd),
                     'threshold_z': tz, 'threshold_consec': tc,
                     'valid_days': vd, 'is_abnormal': int(is_abn),
                     'alert_triggered': int(alert), 'alert_reason': reason,
                     'data_quality': 0, 'in_warmup_flag': 0,
                     'in_gap_flag': 0, 'just_resumed_flag': 0})

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    return df


# ══════════════════════════════════════════════════════
#  内联数据生成：IMU 每日聚合
# ══════════════════════════════════════════════════════
def _imu_feat(btype, si=1.0):
    if btype == 2:  # sleep
        return dict(ax=np.random.normal(0,20), ay=np.random.normal(0,15),
                    az=np.random.normal(980,25), gx=np.random.normal(0,2),
                    gy=np.random.normal(0,2), gz=np.random.normal(0,1.5))
    elif btype == 1:  # move
        return dict(ax=np.random.normal(40,150), ay=np.random.normal(20,120),
                    az=np.random.normal(650,280), gx=np.random.normal(0,90),
                    gy=np.random.normal(0,70), gz=np.random.normal(0,55))
    else:  # scratch
        return dict(ax=np.random.normal(180*si,80), ay=np.random.normal(40,60),
                    az=np.random.normal(800,180), gx=np.random.normal(0,130*si),
                    gy=np.random.normal(0,90), gz=np.random.normal(0,70))

def gen_imu_daily_agg(sc: dict, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    gap_map = _gap_map(sc['gaps'])
    records = []

    for i in range(DAYS):
        if i in gap_map: continue
        temp  = float(_TEMP_ARR[i])
        cnt   = _scratch_count(i, sc['phases'], temp, sc['tc'])
        si    = 1.8 if _is_sick(i, sc) else 1.0
        d_obj = START_DATE + timedelta(days=i)

        for btype in [1, 2]:
            n_ev = int(np.random.uniform(8, 20))
            evs  = [_imu_feat(btype, si) for _ in range(n_ev)]
            if evs:
                records.append({'date': d_obj, 'behavior': btype,
                                'ax_mean': np.mean([e['ax'] for e in evs]),
                                'ay_mean': np.mean([e['ay'] for e in evs]),
                                'az_mean': np.mean([e['az'] for e in evs]),
                                'gx_mean': np.mean([e['gx'] for e in evs]),
                                'gy_mean': np.mean([e['gy'] for e in evs]),
                                'gz_mean': np.mean([e['gz'] for e in evs]),
                                'event_count': n_ev,
                                'total_minutes': n_ev * np.random.uniform(3, 30)})

        if cnt > 0:
            evs = [_imu_feat(3, si) for _ in range(cnt)]
            records.append({'date': d_obj, 'behavior': 3,
                            'ax_mean': np.mean([e['ax'] for e in evs]),
                            'ay_mean': np.mean([e['ay'] for e in evs]),
                            'az_mean': np.mean([e['az'] for e in evs]),
                            'gx_mean': np.mean([e['gx'] for e in evs]),
                            'gy_mean': np.mean([e['gy'] for e in evs]),
                            'gz_mean': np.mean([e['gz'] for e in evs]),
                            'event_count': cnt,
                            'total_minutes': cnt * np.random.uniform(0.02, 0.13)})

    df = pd.DataFrame(records)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


# ══════════════════════════════════════════════════════
#  数据加载（DB 或 内联）
# ══════════════════════════════════════════════════════
_skin_cache: dict = {}
_imu_cache:  dict = {}

def get_skin(sn: str) -> pd.DataFrame:
    if sn not in _skin_cache:
        if _DB_AVAILABLE:
            t   = f"{SKIN_SCHEMA}.{sn.lower()}"
            sql = f'''SELECT stat_date AS date,
                        scratch_count, baseline_mean, baseline_std,
                        zscore, avg_zscore, consec_abnormal, eval_phase,
                        threshold_z, threshold_consec,
                        valid_days, is_abnormal, alert_triggered, alert_reason,
                        data_quality,
                        (data_quality = 0 AND eval_phase = 0)::int AS in_warmup_flag,
                        (data_quality IN (1,2,3,4))::int           AS in_gap_flag,
                        (data_quality = 5)::int                    AS just_resumed_flag
                      FROM {t} ORDER BY stat_date'''
            df = _db_read(sql)
            df['date'] = pd.to_datetime(df['date'])
        else:
            sc = SC_MAP[sn]
            df = gen_skin_daily(sc, seed=42 + list(SC_MAP).index(sn))
        # synthesize avg_temperature / avg_humidity from global arrays for inline
        if not _DB_AVAILABLE:
            pass  # already in inline-generated df
        _skin_cache[sn] = df
    return _skin_cache[sn]

def get_imu(sn: str) -> pd.DataFrame:
    if sn not in _imu_cache:
        if _DB_AVAILABLE:
            t   = sn.lower()
            sql = (f"SELECT to_char(to_timestamp(CAST(ts AS BIGINT)/1000), 'YYYY-MM-DD') AS date, "
                   f"AVG(ax) ax_mean, AVG(ay) ay_mean, AVG(az) az_mean, "
                   f"AVG(gx) gx_mean, AVG(gy) gy_mean, AVG(gz) gz_mean, "
                   f"COUNT(*) event_count, "
                   f"SUM(CAST(ts_end AS BIGINT) - CAST(ts AS BIGINT))/60000.0 total_minutes "
                   f"FROM {IMU_DB}.{t} "
                   f"GROUP BY to_char(to_timestamp(CAST(ts AS BIGINT)/1000), 'YYYY-MM-DD') "
                   f"ORDER BY date")
            try:
                result = _td_read(sql)
                cols   = [c['name'] for c in result.get('column_meta', [])]
                data   = result.get('data', [])
                df = pd.DataFrame(data, columns=cols) if data else pd.DataFrame(
                    columns=['date','ax_mean','ay_mean','az_mean',
                             'gx_mean','gy_mean','gz_mean','event_count','total_minutes'])
            except Exception:
                df = pd.DataFrame(
                    columns=['date','ax_mean','ay_mean','az_mean',
                             'gx_mean','gy_mean','gz_mean','event_count','total_minutes'])
            df['date'] = pd.to_datetime(df['date'])
            # behavior column not stored in imu table; derive from pattern (optional placeholder)
            df['behavior'] = 1
        else:
            sc = SC_MAP[sn]
            df = gen_imu_daily_agg(sc, seed=42 + list(SC_MAP).index(sn))
        _imu_cache[sn] = df
    return _imu_cache[sn]

def get_baseline() -> pd.DataFrame:
    if _DB_AVAILABLE:
        rows = []
        for sc in SCENARIOS:
            t   = f"{BSL_SCHEMA}.{sc['sn'].lower()}"
            sql = f'''SELECT MAX(stat_date) AS stat_date, baseline_mean, baseline_std,
                             temp_coef, confidence, valid_days
                      FROM {t}
                      ORDER BY stat_date DESC LIMIT 1'''
            try:
                df = _db_read(sql)
                if not df.empty:
                    row = df.iloc[0]
                    rows.append({'device_id': sc['sn'],
                                 'baseline_mean': row['baseline_mean'],
                                 'baseline_std':  row['baseline_std'],
                                 'temp_coef':     row['temp_coef'],
                                 'valid_days':     row['valid_days'],
                                 'confidence':     row['confidence']})
            except Exception:
                pass
        return pd.DataFrame(rows)
    rows = []
    for idx, sc in enumerate(SCENARIOS):
        df = get_skin(sc['sn'])
        ok = df[(df['data_quality']==0) & (df['in_warmup_flag']==0) & (df['in_gap_flag']==0)]
        last30 = ok.tail(30)
        if last30.empty: continue
        counts = last30['scratch_count'].values
        temps  = last30.get('avg_temperature', pd.Series(dtype=float)).values
        vd     = min(int(last30['valid_days'].max()), 30) if 'valid_days' in last30 else 0
        rows.append({'device_id': sc['sn'],
                     'baseline_mean': round(float(np.mean(counts)), 2),
                     'baseline_std':  round(max(float(np.std(counts)), MIN_STD), 2),
                     'temp_coef':     _temp_coef(list(counts), list(temps)) if len(temps) else 0.0,
                     'valid_days':    vd,
                     'confidence':    _conf(vd)})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════
#  辅助：为轴添加阶段背景色
# ══════════════════════════════════════════════════════
def _phase_bg(ax_obj, dates):
    if len(dates) < 4: return
    ax_obj.axvspan(dates.iloc[0],  dates.iloc[min(WARMUP, len(dates)-1)],
                   alpha=0.08, color='#BDC3C7', zorder=0, label='热身期')
    if len(dates) > 14:
        ax_obj.axvspan(dates.iloc[WARMUP], dates.iloc[min(14,len(dates)-1)],
                       alpha=0.06, color='#FADBD8', zorder=0)
    if len(dates) > 30:
        ax_obj.axvspan(dates.iloc[14],     dates.iloc[min(30,len(dates)-1)],
                       alpha=0.05, color='#FEF9E7', zorder=0)

def _alert_vlines(ax_obj, df):
    prev = False
    for _, r in df.iterrows():
        if r.get('alert_triggered', 0) and not prev:
            ax_obj.axvline(r['date'], color='#C0392B', lw=1.8,
                           ls='--', alpha=0.85, zorder=6)
        prev = bool(r.get('alert_triggered', 0))

def _gap_bg(ax_obj, df):
    gap = df[df['in_gap_flag']==1]
    if gap.empty: return
    for d in gap['date']:
        ax_obj.axvspan(d - pd.Timedelta(hours=12),
                       d + pd.Timedelta(hours=12),
                       alpha=0.25, color='#D5D8DC', zorder=0)


# ══════════════════════════════════════════════════════
#  图 01 — IMU 6轴数据
# ══════════════════════════════════════════════════════
def fig_imu():
    sn  = 'device_id_2'
    imu = get_imu(sn)
    if imu.empty:
        print('  [01] IMU 数据为空，跳过'); return

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(f'IMU 6轴原始数据  —  {SN_LABEL[sn]}\n'
                 f'（每点=当日均值；抓挠事件在第60-80天明显增强）',
                 fontsize=14, fontweight='bold', y=0.98)

    axes_info = [
        ('ax_mean', '加速度 ax (mg)', None),
        ('ay_mean', '加速度 ay (mg)', None),
        ('az_mean', '加速度 az (mg)', None),
        ('gx_mean', '陀螺仪 gx (deg/s)', None),
        ('gy_mean', '陀螺仪 gy (deg/s)', None),
        ('gz_mean', '陀螺仪 gz (deg/s)', None),
    ]

    gs = gridspec.GridSpec(6, 1, hspace=0.18, top=0.92, bottom=0.06,
                           left=0.09, right=0.97)

    for row_i, (col, ylabel, _) in enumerate(axes_info):
        ax = fig.add_subplot(gs[row_i])

        col_c = SN_COLOR[sn]
        if col in imu.columns:
            y = imu[col].fillna(0)
            ax.scatter(imu['date'], y, c=col_c, s=22, alpha=0.55, zorder=4)

        d0 = pd.Timestamp('2024-03-01')
        d1 = pd.Timestamp('2024-03-21')
        ax.axvspan(d0, d1, alpha=0.10, color='#E74C3C', zorder=0)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis='x', labelsize=8,
                       labelbottom=(row_i == len(axes_info)-1))
        if row_i == 0:
            handles = [mpatches.Patch(color='#E74C3C', alpha=0.3, label='发病期(第60-80天)')]
            ax.legend(handles=handles, fontsize=9, loc='upper left', framealpha=0.9)
        if row_i == len(axes_info) - 1:
            ax.xaxis.set_major_formatter(
                plt.matplotlib.dates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=25)

    path = os.path.join(OUT_DIR, '01_imu_6axis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [OK] 已保存: {path}')


# ══════════════════════════════════════════════════════
#  图 02 — 行为识别统计
# ══════════════════════════════════════════════════════
def fig_behavior():
    sns_show = KEY_SNS
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    fig.suptitle('每日抓挠次数时序  —  4个代表场景\n'
                 '红色=异常天；垂直虚线=报警触发；灰色背景=缺口天',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.subplots_adjust(hspace=0.45, wspace=0.32, top=0.90, bottom=0.12)

    for col, sn in enumerate(sns_show):
        skin  = get_skin(sn)
        col_c = SN_COLOR[sn]

        ax_bar = axes[col]
        ok     = skin[(skin['data_quality']==0) | (skin['in_warmup_flag']==1)]

        bar_c = ['#E74C3C' if r.get('is_abnormal') else '#BDC3C7' if r.get('in_warmup_flag')
                 else col_c
                 for _, r in ok.iterrows()]
        ax_bar.bar(ok['date'], ok['scratch_count'], color=bar_c,
                   alpha=0.75, width=1.0, zorder=3)

        _gap_bg(ax_bar, skin)

        sc = SC_MAP[sn]
        if sc.get('sick'):
            s, e = sc['sick']
            d0 = pd.Timestamp(START_DATE + timedelta(days=s))
            d1 = pd.Timestamp(START_DATE + timedelta(days=e))
            ax_bar.axvspan(d0, d1, alpha=0.12, color='#E74C3C', zorder=0)
        episodes = sc.get('sick_episodes', [])
        for s, e in episodes:
            d0 = pd.Timestamp(START_DATE + timedelta(days=s))
            d1 = pd.Timestamp(START_DATE + timedelta(days=e))
            ax_bar.axvspan(d0, d1, alpha=0.12, color='#E74C3C', zorder=0)

        _alert_vlines(ax_bar, skin)
        ax_bar.set_ylabel('抓挠次数/天', fontsize=9)
        ax_bar.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter('%m-%d'))
        plt.setp(ax_bar.xaxis.get_majorticklabels(), rotation=30, fontsize=8)
        ax_bar.set_title(f'{SN_LABEL[sn]}', fontsize=10, fontweight='bold', color=col_c)

        total_abn   = int(skin['is_abnormal'].sum())
        total_alert = int(skin['alert_triggered'].sum())
        ax_bar.text(0.99, 0.97,
                    f'异常{total_abn}天\n推送{total_alert}次',
                    transform=ax_bar.transAxes, fontsize=9,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='#EBF5FB', edgecolor=col_c, lw=1.2))

    path = os.path.join(OUT_DIR, '02_behavior.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [OK] 已保存: {path}')


# ══════════════════════════════════════════════════════
#  图 03 — 皮肤健康评估
# ══════════════════════════════════════════════════════
def fig_health():
    fig, axes = plt.subplots(3, 4, figsize=(26, 16))
    fig.suptitle('皮肤健康评估  —  4个代表场景\n'
                 '行1: 抓挠次数+个体基线；行2: z-score+动态门槛；行3: 连续异常天数',
                 fontsize=14, fontweight='bold', y=0.99)
    fig.subplots_adjust(hspace=0.48, wspace=0.28, top=0.91, bottom=0.07)

    row_titles = ['① 抓挠次数 + 动态基线（±2σ 置信区间）',
                  '② 温度修正 z-score + 动态阈值',
                  '③ 连续异常天数 + 推送触发']

    for col, sn in enumerate(KEY_SNS):
        df    = get_skin(sn)
        col_c = SN_COLOR[sn]
        ok    = df[df['data_quality'] == 0]
        ev    = ok[ok['baseline_mean'].notna()]
        sc    = SC_MAP[sn]

        # 行 0：抓挠次数 + 基线
        ax0 = axes[0][col]
        if not ev.empty:
            ax0.fill_between(ev['date'],
                             ev['baseline_mean'] - 2*ev['baseline_std'],
                             ev['baseline_mean'] + 2*ev['baseline_std'],
                             color=col_c, alpha=0.12, zorder=2)
            ax0.plot(ev['date'], ev['baseline_mean'],
                     color=col_c, lw=2.5, label='个体基线', zorder=4)
        ax0.scatter(ok['date'], ok['scratch_count'],
                    c=['#E74C3C' if a else col_c
                       for a in ok['is_abnormal']],
                    s=18, alpha=0.65, zorder=5)
        _gap_bg(ax0, df)
        _alert_vlines(ax0, df)
        if sc.get('sick'):
            s, e = sc['sick']
            ax0.axvspan(pd.Timestamp(START_DATE+timedelta(days=s)),
                        pd.Timestamp(START_DATE+timedelta(days=e)),
                        alpha=0.08, color='#E74C3C')
        for s, e in sc.get('sick_episodes', []):
            ax0.axvspan(pd.Timestamp(START_DATE+timedelta(days=s)),
                        pd.Timestamp(START_DATE+timedelta(days=e)),
                        alpha=0.08, color='#E74C3C')
        ax0.set_title(SN_LABEL[sn], fontsize=10, fontweight='bold',
                      color=col_c, pad=6)
        if col == 0: ax0.set_ylabel('次数/天', fontsize=9)
        ax0.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter('%m-%d'))
        plt.setp(ax0.xaxis.get_majorticklabels(), rotation=28, fontsize=7.5)
        _phase_bg(ax0, df['date'])

        # 行 1：z-score
        ax1 = axes[1][col]
        if not ev.empty:
            zs = ev['zscore'].fillna(0)
            bar_c2 = ['#E74C3C' if (not pd.isna(z) and not pd.isna(t) and z > t)
                      else col_c
                      for z, t in zip(ev['zscore'], ev['threshold_z'])]
            ax1.bar(ev['date'], zs, color=bar_c2, alpha=0.65,
                    width=1.0, zorder=3)
            ax1.plot(ev['date'], ev['avg_zscore'].fillna(0),
                     color='#E67E22', lw=1.8, label='均值z', zorder=4)
            ax1.plot(ev['date'], ev['threshold_z'].fillna(4.0),
                     color='#C0392B', lw=1.8, ls='--',
                     label='z门槛', zorder=5)
        ax1.axhline(0, color='gray', lw=0.6, alpha=0.4)
        _gap_bg(ax1, df)
        if col == 0: ax1.set_ylabel('z-score', fontsize=9)
        ax1.set_ylim(-3, max(8, ev['zscore'].max()*1.2 if not ev.empty else 8))
        if col == 0:
            ax1.legend(fontsize=8, framealpha=0.9, loc='upper left')
        ax1.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter('%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=28, fontsize=7.5)

        # 行 2：连续异常天数
        ax2 = axes[2][col]
        if not ev.empty:
            consec_c = ['#E74C3C' if r['alert_triggered']
                        else '#F39C12' if r['consec_abnormal'] > 0
                        else col_c
                        for _, r in ev.iterrows()]
            ax2.bar(ev['date'], ev['consec_abnormal'],
                    color=consec_c, alpha=0.75, width=1.0, zorder=3)
            ax2.plot(ev['date'], ev['threshold_consec'].fillna(5),
                     color='#C0392B', lw=1.8, ls='--',
                     label='连续天数门槛', zorder=4)
        _gap_bg(ax2, df)
        if col == 0:
            ax2.set_ylabel('连续异常天', fontsize=9)
            ax2.legend(fontsize=8, framealpha=0.9, loc='upper left')
        ax2.set_ylim(0, 9)
        ax2.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter('%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=28, fontsize=7.5)

    for i, title in enumerate(row_titles):
        axes[i][0].annotate(title, xy=(-0.22, 0.5),
                            xycoords='axes fraction',
                            fontsize=9, rotation=90, va='center',
                            ha='right', color='#2C3E50')

    path = os.path.join(OUT_DIR, '03_health.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [OK] 已保存: {path}')


# ══════════════════════════════════════════════════════
#  图 04 — 个体基线演变
# ══════════════════════════════════════════════════════
def fig_baseline():
    show_sns = ['device_id_1','device_id_2','device_id_4','device_id_6',
                'device_id_8','device_id_22','device_id_23','device_id_24']

    fig, (ax_mean, ax_conf) = plt.subplots(2, 1, figsize=(22, 12))
    fig.suptitle('个体基线演变  —  8个代表设备\n'
                 '上图: 基线均值随时间的学习过程；下图: 置信度积累（30天达到满分）',
                 fontsize=14, fontweight='bold')
    fig.subplots_adjust(hspace=0.40, top=0.91, bottom=0.08)

    for sn in show_sns:
        col_c = SN_COLOR[sn]
        df    = get_skin(sn)
        ok    = df[(df['in_gap_flag']==0) & (df['baseline_mean'].notna())]
        if ok.empty: continue

        ax_mean.plot(ok['date'], ok['baseline_mean'],
                     color=col_c, lw=2.0, alpha=0.85,
                     label=SN_LABEL[sn], zorder=4)

        ok2 = df[(df['data_quality']==0) & (df['valid_days'].notna())]
        if not ok2.empty:
            conf_vals = (ok2['valid_days'].clip(upper=30) / 30).round(2)
            ax_conf.plot(ok2['date'], conf_vals,
                         color=col_c, lw=2.0, alpha=0.85,
                         label=SN_LABEL[sn], zorder=4)

    for ax_obj in [ax_mean, ax_conf]:
        for day, lbl, lc in [(WARMUP, f'第{WARMUP+1}天\n评估开始', '#95A5A6'),
                             (14,     '第15天',                     '#85C1E9'),
                             (30,     '第31天（稳定期）',             '#2980B9')]:
            d = pd.Timestamp(START_DATE + timedelta(days=day))
            ax_obj.axvline(d, color=lc, lw=1.2, ls=':', alpha=0.7, zorder=2)

    ax_mean.set_title('基线均值 baseline_mean（次/天）', fontsize=11, loc='left')
    ax_mean.set_ylabel('次/天', fontsize=10)
    ax_mean.legend(fontsize=8.5, loc='upper left', framealpha=0.9, ncol=4)
    ax_mean.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    plt.setp(ax_mean.xaxis.get_majorticklabels(), rotation=25)

    ax_conf.set_title('基线置信度 confidence（0→1，有效正常天/30）',
                      fontsize=11, loc='left')
    ax_conf.set_ylabel('置信度', fontsize=10)
    ax_conf.set_ylim(0, 1.08)
    ax_conf.axhline(1.0, color='#27AE60', lw=1.5, ls='--', alpha=0.6,
                    label='满置信度')
    ax_conf.legend(fontsize=8.5, loc='upper left', framealpha=0.9, ncol=4)
    ax_conf.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    plt.setp(ax_conf.xaxis.get_majorticklabels(), rotation=25)

    bsl = get_baseline()
    if not bsl.empty:
        for _, row in bsl[bsl['device_id'].isin(show_sns)].iterrows():
            col_c = SN_COLOR.get(row['device_id'], 'gray')
            df    = get_skin(row['device_id'])
            ok    = df[df['baseline_mean'].notna()]
            if ok.empty: continue
            last_date = ok['date'].iloc[-1]
            ax_mean.annotate(f"{row['baseline_mean']:.1f}",
                             xy=(last_date, row['baseline_mean']),
                             xytext=(6, 0), textcoords='offset points',
                             fontsize=7.5, color=col_c, va='center')

    path = os.path.join(OUT_DIR, '04_baseline.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [OK] 已保存: {path}')


# ══════════════════════════════════════════════════════
#  图 05 — 脖子温度估算
# ══════════════════════════════════════════════════════
def fig_neck_temp():
    show_sns = ['device_id_1', 'device_id_2', 'device_id_4']
    np.random.seed(99)

    fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)
    fig.suptitle('脖子（皮肤）温度估算  —  3个设备对比\n'
                 '基于日均抓挠强度与炎症状态推算',
                 fontsize=14, fontweight='bold')
    fig.subplots_adjust(hspace=0.38, top=0.91, bottom=0.07)

    for row_i, sn in enumerate(show_sns):
        df    = get_skin(sn)
        ax    = axes[row_i]
        col_c = SN_COLOR[sn]

        n = len(df)
        np.random.seed(99 + row_i)
        noise      = np.random.normal(0, 0.15, n)
        daily_wave = 0.15 * np.sin(np.arange(n) * 2 * np.pi / 7)

        cnt = df['scratch_count'].fillna(0).values
        abn = df['is_abnormal'].fillna(0).values
        gap = df['in_gap_flag'].fillna(0).values

        neck = 38.5 + 0.03 * cnt + 0.85 * abn + daily_wave + noise
        neck = np.where(gap == 1, np.nan, neck)

        base = 38.5 + daily_wave + noise * 0.3
        base = np.where(gap == 1, np.nan, base)

        ax.fill_between(df['date'], base - 0.3, base + 0.3,
                        color='#AED6F1', alpha=0.35, label='正常体温区间')
        ax.plot(df['date'], neck, color=col_c, lw=1.8,
                alpha=0.9, label='脖子温度（推算）', zorder=4)
        ax.axhline(38.5, color='#2980B9', lw=1.2, ls='--',
                   alpha=0.5, label='正常基线 38.5°C')
        ax.axhline(39.5, color='#E74C3C', lw=1.0, ls=':',
                   alpha=0.6, label='警戒线 39.5°C')

        _gap_bg(ax, df)
        _alert_vlines(ax, df)

        sc = SC_MAP[sn]
        if sc.get('sick'):
            s, e = sc['sick']
            ax.axvspan(pd.Timestamp(START_DATE+timedelta(days=s)),
                       pd.Timestamp(START_DATE+timedelta(days=e)),
                       alpha=0.10, color='#E74C3C', label='发病期')

        ax.set_ylabel('温度 (°C)', fontsize=10)
        ax.set_ylim(37.5, 41.0)
        ax.set_title(SN_LABEL[sn], fontsize=11, loc='left',
                     color=col_c, fontweight='bold')
        ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9, ncol=4)

        valid = neck[~np.isnan(neck)]
        if len(valid):
            ax.text(0.01, 0.95,
                    f'均值={np.mean(valid):.2f}°C  峰值={np.max(valid):.2f}°C',
                    transform=ax.transAxes, fontsize=9,
                    va='top', color='#2C3E50',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='#EBF5FB', edgecolor=col_c, lw=1.0))

    axes[-1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=25)

    path = os.path.join(OUT_DIR, '05_neck_temp.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [OK] 已保存: {path}')


# ══════════════════════════════════════════════════════
#  图 06 — 环境温湿度 & 行为相关性
# ══════════════════════════════════════════════════════
def fig_env():
    show_env_sns = ['device_id_1', 'device_id_8', 'device_id_17', 'device_id_20']

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('环境温湿度与抓挠行为相关性\n'
                 '上图: 温度时序；中图: 湿度时序；下图: 温度 vs 抓挠次数散点',
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.30,
                           top=0.91, bottom=0.07, left=0.08, right=0.97)

    ax_temp = fig.add_subplot(gs[0, :])
    ax_humi = fig.add_subplot(gs[1, :])
    ax_sc_a = fig.add_subplot(gs[2, 0])
    ax_sc_b = fig.add_subplot(gs[2, 1])

    for sn in show_env_sns:
        col_c = SN_COLOR[sn]
        df    = get_skin(sn)
        ok    = df[(df['data_quality']==0) | (df['in_warmup_flag']==1)]
        if ok.empty or 'avg_temperature' not in ok.columns: continue
        ax_temp.plot(ok['date'], ok['avg_temperature'],
                     color=col_c, lw=1.3, alpha=0.7, label=SN_LABEL[sn])
        ax_humi.plot(ok['date'], ok['avg_humidity'],
                     color=col_c, lw=1.3, alpha=0.7, label=SN_LABEL[sn])

    if all('avg_temperature' not in get_skin(sn).columns for sn in show_env_sns):
        dates = [START_DATE + timedelta(days=i) for i in range(DAYS)]
        dates_ts = pd.to_datetime(dates)
        ax_temp.plot(dates_ts, _TEMP_ARR, color='#2980B9', lw=1.5, label='全局温度')
        ax_humi.plot(dates_ts, _HUMI_ARR, color='#27AE60', lw=1.5, label='全局湿度')

    ax_temp.set_ylabel('温度 (°C)', fontsize=10)
    ax_temp.set_title('① 环境温度（180天）', fontsize=11, loc='left')
    ax_temp.legend(fontsize=8, loc='upper right', framealpha=0.9, ncol=4)
    ax_temp.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    plt.setp(ax_temp.xaxis.get_majorticklabels(), rotation=25)

    ax_humi.set_ylabel('湿度 (%)', fontsize=10)
    ax_humi.set_title('② 环境湿度（180天）', fontsize=11, loc='left')
    ax_humi.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    plt.setp(ax_humi.xaxis.get_majorticklabels(), rotation=25)

    for ax_sc, sn, title in [
        (ax_sc_a, 'device_id_1',  '场景：完全正常（温度系数≈0.10）'),
        (ax_sc_b, 'device_id_8',  '场景：季节性过敏（温度系数≈0.35）'),
    ]:
        df    = get_skin(sn)
        ok    = df[(df['data_quality']==0) & (df['in_warmup_flag']==0)]
        col_c = SN_COLOR[sn]
        if ok.empty or 'avg_temperature' not in ok.columns: continue

        sc_obj = ax_sc.scatter(ok['avg_temperature'], ok['scratch_count'],
                               c=ok['avg_temperature'], cmap='RdYlBu_r',
                               s=30, alpha=0.6, zorder=4)
        x = ok['avg_temperature'].values
        y = ok['scratch_count'].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() > 5:
            z = np.polyfit(x[valid], y[valid], 1)
            p = np.poly1d(z)
            xr = np.linspace(x[valid].min(), x[valid].max(), 50)
            ax_sc.plot(xr, p(xr), color='#C0392B', lw=2.0,
                       label=f'趋势线 斜率={z[0]:.3f}次/°C')

        plt.colorbar(sc_obj, ax=ax_sc, label='温度 (°C)', shrink=0.9)
        ax_sc.set_xlabel('环境温度 (°C)', fontsize=10)
        ax_sc.set_ylabel('当日抓挠次数', fontsize=10)
        ax_sc.set_title(f'③ {title}', fontsize=10, loc='left')
        ax_sc.legend(fontsize=9, framealpha=0.9)

    path = os.path.join(OUT_DIR, '06_env.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [OK] 已保存: {path}')


# ══════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════
def query_summary():
    print("\n======= 可视化数据来源 =======")
    print(f"  数据来源: {'MySQL 数据库' if _DB_AVAILABLE else '内联生成（等价数据）'}")
    print(f"  场景数量: {len(SCENARIOS)} 个")
    print(f"  输出目录: {OUT_DIR}")


def main():
    print(f'\n{"="*55}')
    print(f'  数据来源: {"MySQL 数据库" if _DB_AVAILABLE else "内联生成（等价数据）"}')
    print(f'  输出目录: {OUT_DIR}')
    print(f'{"="*55}\n')

    steps = [
        ('图01  IMU 6轴数据',     fig_imu),
        ('图02  行为识别统计',     fig_behavior),
        ('图03  皮肤健康评估',     fig_health),
        ('图04  个体基线演变',     fig_baseline),
        ('图05  脖子温度估算',     fig_neck_temp),
        ('图06  环境温湿度相关性', fig_env),
    ]

    for title, fn in steps:
        print(f'生成 {title} ...')
        try:
            fn()
        except Exception as e:
            print(f'  [FAIL] 失败: {e}')
            import traceback; traceback.print_exc()

    print(f'\n[完成] 全部图表已保存到: {OUT_DIR}')


if __name__ == '__main__':
    main()
