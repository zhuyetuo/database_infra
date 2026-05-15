"""
宠物项圈皮肤健康监测 — 数据库可视化展示
==========================================
优先从 MySQL 数据库读取数据；数据库不可用时自动切换到内联数据生成
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
    ('DEV_A_NORMAL',   '场景A  完全正常',             '#2ECC71'),
    ('DEV_B_SICK',     '场景B  皮肤病后康复',           '#E74C3C'),
    ('DEV_C_SEASON',   '场景C  季节性升高',             '#F39C12'),
    ('DEV_D_ALLERGY',  '场景D  缓慢升高(过敏)',          '#9B59B6'),
    ('DEV_E1_UNWORN',  '场景E1 忘记佩戴',              '#1ABC9C'),
    ('DEV_E2_BATTERY', '场景E2 没电+缺口后皮肤病',      '#E67E22'),
    ('DEV_E3_SIGNAL',  '场景E3 信号断续',              '#3498DB'),
    ('DEV_E4_LOOSE',   '场景E4 松动无效',              '#7F8C8D'),
]
SN_COLOR  = {sn: c for sn, _, c in DEV_META}
SN_LABEL  = {sn: lb for sn, lb, _ in DEV_META}

KEY_SNS = ['DEV_A_NORMAL', 'DEV_B_SICK', 'DEV_D_ALLERGY', 'DEV_E2_BATTERY']

# ── 全局时间序列（复现 demo_all.py 及入库脚本的随机种子）──
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
    {'sn': 'DEV_A_NORMAL',   'phases': [(0,180,10,2)],                              'tc': 0.10, 'gaps': [],                                        'sick': None},
    {'sn': 'DEV_B_SICK',     'phases': [(0,60,10,2),(60,80,22,3),(80,180,10,2)],    'tc': 0.10, 'gaps': [],                                        'sick': (60,80)},
    {'sn': 'DEV_C_SEASON',   'phases': [(0,180,10,2)],                              'tc': 0.25, 'gaps': [],                                        'sick': None},
    {'sn': 'DEV_D_ALLERGY',  'phases': [(0,60,10,2),(60,120,13,2),(120,180,15,2)],  'tc': 0.10, 'gaps': [],                                        'sick': None},
    {'sn': 'DEV_E1_UNWORN',  'phases': [(0,180,10,2)],                              'tc': 0.10, 'gaps': [(35,38,'unworn')],                        'sick': None},
    {'sn': 'DEV_E2_BATTERY', 'phases': [(0,60,10,2),(60,73,22,3),(73,180,10,2)],   'tc': 0.10, 'gaps': [(40,45,'battery')],                       'sick': (60,73)},
    {'sn': 'DEV_E3_SIGNAL',  'phases': [(0,180,10,2)],                              'tc': 0.10, 'gaps': [(d,d+1,'signal') for d in sorted(_SIG_GAPS)], 'sick': None},
    {'sn': 'DEV_E4_LOOSE',   'phases': [(0,180,10,2)],                              'tc': 0.10, 'gaps': [(50,58,'loose')],                         'sick': None},
]
SC_MAP = {s['sn']: s for s in SCENARIOS}

# ══════════════════════════════════════════════════════
#  数据库连接（可选）
# ══════════════════════════════════════════════════════
IMU_DB  = 'pet_imu'
SKIN_DB = 'pet_skin_health'
_DB_AVAILABLE = False

try:
    import mysql.connector as _mc
    _test = _mc.connect(host='127.0.0.1', port=3306, user='root',
                        password='123456', connection_timeout=3)
    _test.close()
    _DB_AVAILABLE = True
    print('✅ 数据库连接成功，使用数据库数据')
except Exception:
    print('⚠  数据库不可用，使用内联生成数据（与数据库内容算法一致）')


def _db_read(sql: str, db: str) -> pd.DataFrame:
    conn = _mc.connect(host='127.0.0.1', port=3306, user='root',
                       password='123456', database=db)
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


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

DQ_MAP = {'unworn':1,'battery':2,'signal':3,'loose':4}

# ══════════════════════════════════════════════════════
#  内联数据生成：皮肤健康每日评估
# ══════════════════════════════════════════════════════
def gen_skin_daily(sc: dict, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    gap_map = _gap_map(sc['gaps'])
    sick    = sc.get('sick')
    mean=std=None; bc=[]; bt=[]; consec=0; vd=0
    gc=0; in_gap=False; resumed=False; zbuf=[]
    rows = []

    for i in range(DAYS):
        d    = START_DATE + timedelta(days=i)
        temp = round(float(_TEMP_ARR[i]), 1)
        humi = round(float(_HUMI_ARR[i]), 1)
        is_sick = bool(sick and sick[0] <= i < sick[1])

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
                         'threshold_avgz': None,
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

        cnt = _scratch_count(i, sc['phases'], temp, sc['tc'])
        avg_dur = max(500, int(np.random.normal(4500 if is_sick else 4000, 800)))
        night   = max(0, int(cnt * (np.random.uniform(0.2, 0.35) if is_sick
                                    else np.random.uniform(0.05, 0.15))))

        if i < WARMUP:
            bc.append(cnt); bt.append(temp)
            rows.append({'date': d, 'scratch_count': cnt,
                         'avg_temperature': temp, 'avg_humidity': humi,
                         'baseline_mean': None, 'baseline_std': None,
                         'temp_coef': None, 'temp_effect': None,
                         'zscore': None, 'avg_zscore': None,
                         'consec_abnormal': 0, 'eval_phase': 0,
                         'threshold_z': None, 'threshold_consec': None, 'threshold_avgz': None,
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
                         'threshold_z': tz, 'threshold_consec': tc, 'threshold_avgz': ta,
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
                     'threshold_z': tz, 'threshold_consec': tc, 'threshold_avgz': ta,
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
                    gy=np.random.normal(0,2), gz=np.random.normal(0,1.5),
                    az_rms=abs(np.random.normal(980,25))+np.random.uniform(5,20),
                    scratch_hz=None)
    elif btype == 1:  # move
        return dict(ax=np.random.normal(40,150), ay=np.random.normal(20,120),
                    az=np.random.normal(650,280), gx=np.random.normal(0,90),
                    gy=np.random.normal(0,70), gz=np.random.normal(0,55),
                    az_rms=abs(np.random.normal(650,280))+np.random.uniform(80,250),
                    scratch_hz=None)
    else:  # scratch
        return dict(ax=np.random.normal(180*si,80), ay=np.random.normal(40,60),
                    az=np.random.normal(800,180), gx=np.random.normal(0,130*si),
                    gy=np.random.normal(0,90), gz=np.random.normal(0,70),
                    az_rms=abs(np.random.normal(800,180))+np.random.uniform(150*si,350*si),
                    scratch_hz=np.random.uniform(2.0, 5.5))

def gen_imu_daily_agg(sc: dict, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    gap_map = _gap_map(sc['gaps'])
    sick    = sc.get('sick')
    records = []

    for i in range(DAYS):
        if i in gap_map: continue
        temp  = float(_TEMP_ARR[i])
        cnt   = _scratch_count(i, sc['phases'], temp, sc['tc'])
        si    = 1.8 if (sick and sick[0] <= i < sick[1]) else 1.0
        d_obj = START_DATE + timedelta(days=i)

        seg_scratch = [cnt//3, cnt - cnt//3]
        seg_btypes  = [[2,1], [2,1]]  # morning/afternoon background types

        for btype in [1, 2]:  # generate move + sleep events
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
                                'az_rms_mean': np.mean([e['az_rms'] for e in evs]),
                                'hz_mean': None, 'event_count': n_ev,
                                'total_minutes': n_ev * np.random.uniform(3, 30)})

        if cnt > 0:  # scratch events
            evs = [_imu_feat(3, si) for _ in range(cnt)]
            records.append({'date': d_obj, 'behavior': 3,
                            'ax_mean': np.mean([e['ax'] for e in evs]),
                            'ay_mean': np.mean([e['ay'] for e in evs]),
                            'az_mean': np.mean([e['az'] for e in evs]),
                            'gx_mean': np.mean([e['gx'] for e in evs]),
                            'gy_mean': np.mean([e['gy'] for e in evs]),
                            'gz_mean': np.mean([e['gz'] for e in evs]),
                            'az_rms_mean': np.mean([e['az_rms'] for e in evs]),
                            'hz_mean': np.mean([e['scratch_hz'] for e in evs]),
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
            t   = sn.lower()
            sql = f'''SELECT FROM_UNIXTIME(stat_date_ts/1000) AS date,
                        scratch_count, avg_temperature, avg_humidity,
                        baseline_mean, baseline_std, temp_coef, temp_effect,
                        zscore, avg_zscore, consec_abnormal, eval_phase,
                        threshold_z, threshold_consec, threshold_avgz,
                        valid_days, is_abnormal, alert_triggered, alert_reason,
                        data_quality, in_warmup_flag, in_gap_flag, just_resumed_flag
                      FROM `{t}` ORDER BY stat_date_ts'''
            df = _db_read(sql, SKIN_DB)
            df['date'] = pd.to_datetime(df['date'])
        else:
            sc = SC_MAP[sn]
            df = gen_skin_daily(sc, seed=42 + list(SC_MAP).index(sn))
        _skin_cache[sn] = df
    return _skin_cache[sn]

def get_imu(sn: str) -> pd.DataFrame:
    if sn not in _imu_cache:
        if _DB_AVAILABLE:
            t   = sn.lower()
            sql = f'''SELECT DATE(FROM_UNIXTIME(ts_start/1000)) AS date, behavior,
                        AVG(ax) ax_mean, AVG(ay) ay_mean, AVG(az) az_mean,
                        AVG(gx) gx_mean, AVG(gy) gy_mean, AVG(gz) gz_mean,
                        AVG(az_rms) az_rms_mean, AVG(scratch_hz) hz_mean,
                        COUNT(*) event_count,
                        SUM(ts_end-ts_start)/60000.0 total_minutes
                      FROM `{t}` GROUP BY date, behavior ORDER BY date, behavior'''
            df = _db_read(sql, IMU_DB)
            df['date'] = pd.to_datetime(df['date'])
        else:
            sc = SC_MAP[sn]
            df = gen_imu_daily_agg(sc, seed=42 + list(SC_MAP).index(sn))
        _imu_cache[sn] = df
    return _imu_cache[sn]

def get_baseline() -> pd.DataFrame:
    if _DB_AVAILABLE:
        sql = 'SELECT device_sn, baseline_mean, baseline_std, temp_coef, valid_days, eval_phase, confidence FROM skin_baseline'
        return _db_read(sql, SKIN_DB)
    rows = []
    for idx, sc in enumerate(SCENARIOS):
        df = get_skin(sc['sn'])
        ok = df[(df['data_quality']==0) & (df['in_warmup_flag']==0) & (df['in_gap_flag']==0)]
        last30 = ok.tail(30)
        if last30.empty: continue
        counts = last30['scratch_count'].values
        temps  = last30['avg_temperature'].values
        vd     = min(int(last30['valid_days'].max()), 30)
        rows.append({'device_sn': sc['sn'],
                     'baseline_mean': round(float(np.mean(counts)), 2),
                     'baseline_std':  round(max(float(np.std(counts)), MIN_STD), 2),
                     'temp_coef':     _temp_coef(list(counts), list(temps)),
                     'valid_days':    vd, 'eval_phase': _phase(vd),
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
    sn  = 'DEV_B_SICK'
    imu = get_imu(sn)
    if imu.empty:
        print('  [01] IMU 数据为空，跳过'); return

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle(f'IMU 6轴原始数据  —  {SN_LABEL[sn]}\n'
                 f'（每点=当日该行为类型的均值；抓挠事件在第60-80天明显增强）',
                 fontsize=14, fontweight='bold', y=0.98)

    axes_info = [
        ('az_rms_mean', '活动强度 az_rms (mg)', None),
        ('ax_mean',     '加速度 ax (mg)',         None),
        ('ay_mean',     '加速度 ay (mg)',         None),
        ('az_mean',     '加速度 az (mg)',         None),
        ('gx_mean',     '陀螺仪 gx (deg/s)',      None),
        ('gy_mean',     '陀螺仪 gy (deg/s)',      None),
        ('gz_mean',     '陀螺仪 gz (deg/s)',      None),
    ]

    gs = gridspec.GridSpec(7, 1, hspace=0.18, top=0.92, bottom=0.06,
                           left=0.09, right=0.97)

    for row_i, (col, ylabel, _) in enumerate(axes_info):
        ax = fig.add_subplot(gs[row_i])
        for btype in [2, 1, 3]:
            sub = imu[imu['behavior'] == btype]
            if sub.empty: continue
            y = sub[col].fillna(0)
            ax.scatter(sub['date'], y, c=BEH_COLORS[btype],
                       s=22, alpha=0.55, label=BEH_LABELS[btype], zorder=4)

        # 发病期高亮
        d0 = pd.Timestamp('2024-03-01')   # day 60
        d1 = pd.Timestamp('2024-03-21')   # day 80
        ax.axvspan(d0, d1, alpha=0.10, color='#E74C3C', zorder=0)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis='x', labelsize=8,
                       labelbottom=(row_i == len(axes_info)-1))
        if row_i == 0:
            handles = [mpatches.Patch(color=BEH_COLORS[b], label=BEH_LABELS[b])
                       for b in [2, 1, 3]]
            handles.append(mpatches.Patch(color='#E74C3C', alpha=0.3, label='发病期(第60-80天)'))
            ax.legend(handles=handles, fontsize=9, loc='upper left',
                      framealpha=0.9, ncol=4)
        if row_i == len(axes_info) - 1:
            ax.xaxis.set_major_formatter(
                plt.matplotlib.dates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=25)

    path = os.path.join(OUT_DIR, '01_imu_6axis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✅ 已保存: {path}')


# ══════════════════════════════════════════════════════
#  图 02 — 行为识别统计
# ══════════════════════════════════════════════════════
def fig_behavior():
    sns_show = KEY_SNS
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('行为识别分析  —  4个代表场景\n'
                 '上行: 行为时间占比（运动/睡眠/抓挠）；下行: 每日抓挠次数时序',
                 fontsize=14, fontweight='bold', y=0.99)
    fig.subplots_adjust(hspace=0.45, wspace=0.32, top=0.90, bottom=0.08)

    for col, sn in enumerate(sns_show):
        imu  = get_imu(sn)
        skin = get_skin(sn)
        col_c = SN_COLOR[sn]

        # 上行：饼图（行为时长占比）
        ax_pie = axes[0][col]
        if not imu.empty:
            agg = imu.groupby('behavior')['total_minutes'].sum()
            sizes  = [agg.get(b, 0) for b in [2, 1, 3]]
            labels = [f'{BEH_LABELS[b]}\n{agg.get(b,0):.0f}h' for b in [2,1,3]]
            colors = [BEH_COLORS[b] for b in [2, 1, 3]]
            wedges, texts, autotexts = ax_pie.pie(
                sizes, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90,
                pctdistance=0.75, labeldistance=1.15,
                textprops={'fontsize': 9})
            for at in autotexts:
                at.set_fontsize(8)
        ax_pie.set_title(SN_LABEL[sn], fontsize=10, pad=8, fontweight='bold',
                         color=col_c)

        # 下行：每日抓挠次数柱状图
        ax_bar = axes[1][col]
        ok = skin[(skin['data_quality']==0) | (skin['in_warmup_flag']==1)]
        gap_df = skin[skin['in_gap_flag']==1]

        bar_c = ['#E74C3C' if r['is_abnormal'] else '#BDC3C7' if r['in_warmup_flag']
                 else col_c
                 for _, r in ok.iterrows()]
        ax_bar.bar(ok['date'], ok['scratch_count'], color=bar_c,
                   alpha=0.75, width=1.0, zorder=3)

        _gap_bg(ax_bar, skin)

        # 发病期标注
        sc = SC_MAP[sn]
        if sc.get('sick'):
            s, e = sc['sick']
            d0 = pd.Timestamp(START_DATE + timedelta(days=s))
            d1 = pd.Timestamp(START_DATE + timedelta(days=e))
            ax_bar.axvspan(d0, d1, alpha=0.12, color='#E74C3C', zorder=0)

        _alert_vlines(ax_bar, skin)
        ax_bar.set_ylabel('抓挠次数/天', fontsize=9)
        ax_bar.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter('%m-%d'))
        plt.setp(ax_bar.xaxis.get_majorticklabels(), rotation=30, fontsize=8)
        ax_bar.set_title(f'日抓挠次数  (异常天=红色)', fontsize=9)

        total_abn = int(skin['is_abnormal'].sum())
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
    print(f'  ✅ 已保存: {path}')


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
        df   = get_skin(sn)
        col_c = SN_COLOR[sn]
        ok   = df[df['data_quality'] == 0]
        ev   = ok[ok['baseline_mean'].notna()]
        sc   = SC_MAP[sn]

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
    print(f'  ✅ 已保存: {path}')


# ══════════════════════════════════════════════════════
#  图 04 — 个体基线演变
# ══════════════════════════════════════════════════════
def fig_baseline():
    fig, (ax_mean, ax_conf) = plt.subplots(2, 1, figsize=(22, 12))
    fig.suptitle('个体基线演变  —  全部 8 个设备\n'
                 '上图: 基线均值随时间的学习过程；下图: 置信度积累（30天达到满分）',
                 fontsize=14, fontweight='bold')
    fig.subplots_adjust(hspace=0.40, top=0.91, bottom=0.08)

    bsl_all = get_baseline()
    phase_colors = {0:'#BDC3C7', 1:'#FADBD8', 2:'#FEF9E7', 3:'#D5F5E3'}

    for sn, _, col_c in DEV_META:
        df = get_skin(sn)
        ok = df[(df['in_gap_flag']==0) & (df['baseline_mean'].notna())]
        if ok.empty: continue

        ax_mean.plot(ok['date'], ok['baseline_mean'],
                     color=col_c, lw=2.0, alpha=0.85,
                     label=SN_LABEL[sn], zorder=4)

        # 置信度 = valid_days / 30，只取 data_quality==0 的天
        ok2 = df[(df['data_quality']==0) & (df['valid_days'].notna())]
        if not ok2.empty:
            conf_vals = (ok2['valid_days'].clip(upper=30) / 30).round(2)
            ax_conf.plot(ok2['date'], conf_vals,
                         color=col_c, lw=2.0, alpha=0.85,
                         label=SN_LABEL[sn], zorder=4)

    # 阶段分界线
    for ax_obj in [ax_mean, ax_conf]:
        for day, lbl, lc in [(WARMUP, f'第{WARMUP+1}天\n评估开始', '#95A5A6'),
                             (14,     '第15天',                     '#85C1E9'),
                             (30,     '第31天（稳定期）',             '#2980B9')]:
            d = pd.Timestamp(START_DATE + timedelta(days=day))
            ax_obj.axvline(d, color=lc, lw=1.2, ls=':', alpha=0.7, zorder=2)
            ax_obj.text(d, ax_obj.get_ylim()[1] if ax_obj.get_ylim()[1] != 0 else 1,
                        lbl, fontsize=7.5, ha='center', color=lc)

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

    # 右侧最终基线值标注
    bsl = get_baseline()
    if not bsl.empty:
        for _, row in bsl.iterrows():
            col_c = SN_COLOR.get(row['device_sn'], 'gray')
            df = get_skin(row['device_sn'])
            ok = df[df['baseline_mean'].notna()]
            if ok.empty: continue
            last_date = ok['date'].iloc[-1]
            ax_mean.annotate(f"{row['baseline_mean']:.1f}",
                             xy=(last_date, row['baseline_mean']),
                             xytext=(6, 0), textcoords='offset points',
                             fontsize=7.5, color=col_c, va='center')

    path = os.path.join(OUT_DIR, '04_baseline.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✅ 已保存: {path}')


# ══════════════════════════════════════════════════════
#  图 05 — 脖子温度估算
# ══════════════════════════════════════════════════════
def fig_neck_temp():
    show_sns = ['DEV_A_NORMAL', 'DEV_B_SICK', 'DEV_D_ALLERGY']
    np.random.seed(99)

    fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)
    fig.suptitle('脖子（皮肤）温度估算  —  3个设备对比\n'
                 '基于日均抓挠强度与炎症状态推算（真实系统中由项圈内置温度传感器提供）',
                 fontsize=14, fontweight='bold')
    fig.subplots_adjust(hspace=0.38, top=0.91, bottom=0.07)

    for row_i, sn in enumerate(show_sns):
        df    = get_skin(sn)
        ax    = axes[row_i]
        col_c = SN_COLOR[sn]

        n = len(df)
        np.random.seed(99 + row_i)
        noise = np.random.normal(0, 0.15, n)
        daily_wave = 0.15 * np.sin(np.arange(n) * 2 * np.pi / 7)  # 7天周期波动

        cnt  = df['scratch_count'].fillna(0).values
        abn  = df['is_abnormal'].fillna(0).values
        gap  = df['in_gap_flag'].fillna(0).values

        neck = 38.5 + 0.03 * cnt + 0.85 * abn + daily_wave + noise
        neck = np.where(gap == 1, np.nan, neck)

        # 基础温度线
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
    print(f'  ✅ 已保存: {path}')


# ══════════════════════════════════════════════════════
#  图 06 — 环境温湿度 & 行为相关性
# ══════════════════════════════════════════════════════
def fig_env():
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('环境温湿度与抓挠行为相关性\n'
                 '上图: 温度时序（8设备共享同一环境）；中图: 湿度时序；下图: 温度 vs 抓挠次数散点',
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.30,
                           top=0.91, bottom=0.07, left=0.08, right=0.97)

    ax_temp = fig.add_subplot(gs[0, :])
    ax_humi = fig.add_subplot(gs[1, :])
    ax_sc_a = fig.add_subplot(gs[2, 0])   # 散点（A: normal）
    ax_sc_b = fig.add_subplot(gs[2, 1])   # 散点（C: season）

    # 温度 & 湿度：所有设备从同一环境（取A的数据即可，但叠加所有展示一致性）
    for sn, _, col_c in DEV_META:
        df = get_skin(sn)
        ok = df[(df['data_quality']==0) | (df['in_warmup_flag']==1)]
        if ok.empty: continue
        ax_temp.plot(ok['date'], ok['avg_temperature'],
                     color=col_c, lw=1.3, alpha=0.55, label=SN_LABEL[sn])
        ax_humi.plot(ok['date'], ok['avg_humidity'],
                     color=col_c, lw=1.3, alpha=0.55, label=SN_LABEL[sn])

    ax_temp.set_ylabel('温度 (°C)', fontsize=10)
    ax_temp.set_title('① 环境温度（180天）', fontsize=11, loc='left')
    ax_temp.legend(fontsize=8, loc='upper right', framealpha=0.9,
                   ncol=4, bbox_to_anchor=(1.0, 1.0))
    ax_temp.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    plt.setp(ax_temp.xaxis.get_majorticklabels(), rotation=25)

    ax_humi.set_ylabel('湿度 (%)', fontsize=10)
    ax_humi.set_title('② 环境湿度（180天）', fontsize=11, loc='left')
    ax_humi.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    plt.setp(ax_humi.xaxis.get_majorticklabels(), rotation=25)

    # 散点：温度 vs 抓挠次数
    for ax_sc, sn, title in [
        (ax_sc_a, 'DEV_A_NORMAL',  '场景A（正常）  温度系数≈0.10'),
        (ax_sc_b, 'DEV_C_SEASON',  '场景C（季节性）温度系数≈0.25'),
    ]:
        df   = get_skin(sn)
        ok   = df[(df['data_quality']==0) & (df['in_warmup_flag']==0)]
        col_c = SN_COLOR[sn]
        if ok.empty: continue

        sc_obj = ax_sc.scatter(ok['avg_temperature'], ok['scratch_count'],
                               c=ok['avg_temperature'], cmap='RdYlBu_r',
                               s=30, alpha=0.6, zorder=4)
        # 趋势线
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
    print(f'  ✅ 已保存: {path}')


# ══════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
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
            print(f'  ❌ 失败: {e}')
            import traceback; traceback.print_exc()

    print(f'\n🎉 全部完成！图表已保存到: {OUT_DIR}')
