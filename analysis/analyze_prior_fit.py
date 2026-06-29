# Created: 2026-06-18 01:40 JST
"""
prior（距離減衰式）の Tobit（左打ち切り）最小二乗フィット

目的: Si-Midorikawa(1999) 形の係数を実データで再フィットし、
      「裾が広すぎる(震度1過大)・頂が低い」prior を急峻に直す。

汚染対策:
  - 「存在しない観測点を震度0」と誤カウントする問題を、各観測点の稼働期間
    [初観測, 最終観測] で近似し、窓内の地震のみ非有感(=censored)として扱う。
  - 有感点(>=0.5)は実測値（打ち切りなし）、稼働中の非有感点は「真値<0.5」
    という左打ち切り情報として尤度に入れる（Tobit）。

モデル:
  mu = mcoef*M + depcoef*depth - alpha*log10(D + R0) - dcoef*D + C
  R0 = 0.0028 * 10^(0.5*M)   (近傍飽和項, 固定)
  intensity ~ N(mu, sigma),  observed only if >= 0.5
"""
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
MAX_DIST   = 800.0
N_SAMPLE_EQ = 4000
CENS_THRESH = 0.5
rng = np.random.default_rng(42)

# 現行係数（比較用）
CUR = dict(mcoef=0.58, depcoef=0.0038, alpha=1.0, dcoef=0.002, C=0.211)

print('データ読み込み中...')
eq  = pd.read_parquet(ROOT/'earthquakes.parquet').dropna(
        subset=['hypo_lat','hypo_lon','depth','magnitude','datetime'])
obs = pd.read_parquet(ROOT/'observations.parquet').dropna(subset=['obs_lat','obs_lon'])

eq['t'] = pd.to_datetime(eq['datetime']).astype('int64')   # ns
eq_t = eq.set_index('event_id')['t'].to_dict()

# マスタ観測点 + 稼働窓 [初観測t, 最終観測t]
obs2 = obs.merge(eq[['event_id','t']], on='event_id', how='inner')
sta = obs2.groupby('station_id').agg(
    obs_lat=('obs_lat','first'), obs_lon=('obs_lon','first'),
    t_first=('t','min'), t_last=('t','max')).reset_index()
STA_LAT = np.radians(sta['obs_lat'].values)
STA_LON = np.radians(sta['obs_lon'].values)
T_FIRST = sta['t_first'].values
T_LAST  = sta['t_last'].values
N_STA = len(sta)
sid_to_idx = {s:i for i,s in enumerate(sta['station_id'].values)}
print(f'マスタ観測点数: {N_STA:,}')

obs['sidx'] = obs['station_id'].map(sid_to_idx)
felt = {eid:(g['sidx'].values, g['intensity'].values.astype('float32'))
        for eid,g in obs.groupby('event_id')}

eq = eq[eq['event_id'].isin(felt)].reset_index(drop=True)
if len(eq) > N_SAMPLE_EQ:
    eq = eq.sample(N_SAMPLE_EQ, random_state=42).reset_index(drop=True)
print(f'フィット対象地震数: {len(eq):,}')

def haversine_km(lat_deg, lon_deg):
    lat_r=math.radians(lat_deg); lon_r=math.radians(lon_deg)
    dlat=STA_LAT-lat_r; dlon=STA_LON-lon_r
    a=np.sin(dlat/2)**2+math.cos(lat_r)*np.cos(STA_LAT)*np.sin(dlon/2)**2
    return 6371.0*2.0*np.arcsin(np.sqrt(np.clip(a,0,1)))

# ── サンプル収集: (D, M, depth, y, is_cens) ──
D_list, M_list, dep_list, y_list, cens_list = [],[],[],[],[]
near_felt_n = 0; near_total = 0   # 近傍(<30km)有感率の検証用
for row in eq.itertuples():
    et = eq_t[row.event_id]
    active = (T_FIRST <= et) & (et <= T_LAST)       # 稼働中の観測点のみ
    d_epi = haversine_km(row.hypo_lat, row.hypo_lon)
    sel = active & (d_epi < MAX_DIST)
    idx = np.where(sel)[0]
    if len(idx)==0: continue
    d_in = d_epi[idx]
    D = np.sqrt(d_in**2 + max(row.depth,1.0)**2)

    is_felt = np.zeros(len(idx), dtype=bool)
    y = np.zeros(len(idx), dtype='float32')
    pos = {gi:j for j,gi in enumerate(idx)}
    f_idx, f_int = felt[row.event_id]
    for gi, iv in zip(f_idx, f_int):
        j = pos.get(gi)
        if j is not None:
            is_felt[j]=True; y[j]=iv

    near = d_in < 30
    near_total += int(near.sum()); near_felt_n += int((near & is_felt).sum())

    D_list.append(D.astype('float32'))
    M_list.append(np.full(len(idx), row.magnitude, dtype='float32'))
    dep_list.append(np.full(len(idx), row.depth, dtype='float32'))
    y_list.append(y)
    cens_list.append(~is_felt)   # 非有感 = 左打ち切り

D   = np.concatenate(D_list)
M   = np.concatenate(M_list)
dep = np.concatenate(dep_list)
Y   = np.concatenate(y_list)
CENS= np.concatenate(cens_list)
R0  = 0.0028 * 10.0**(0.5*M)
logD = np.log10(D + R0)
print(f'総サンプル数: {len(D):,}  (有感={np.sum(~CENS):,}, 非有感={np.sum(CENS):,})')
print(f'検証: 震源<30km 有感率 = {near_felt_n/max(near_total,1):.3f} '
      f'(汚染除去前は約0.5。0.8+なら稼働窓フィルタが効いている)')

felt_mask = ~CENS
yf = Y[felt_mask]

def mu_of(p):
    mcoef, depcoef, alpha, dcoef, C = p
    return mcoef*M + depcoef*dep - alpha*logD - dcoef*D + C

def negloglik(theta):
    mcoef, depcoef, alpha, dcoef, C, log_sigma = theta
    sigma = math.exp(log_sigma)
    mu = mcoef*M + depcoef*dep - alpha*logD - dcoef*D + C
    # 有感: 正規対数尤度
    z = (Y[felt_mask]-mu[felt_mask])/sigma
    ll_felt = -0.5*z*z - log_sigma - 0.5*math.log(2*math.pi)
    # 非有感: P(true < 0.5) = Phi((0.5-mu)/sigma)
    zc = (CENS_THRESH - mu[CENS])/sigma
    ll_cens = norm.logcdf(zc)
    return -(ll_felt.sum() + ll_cens.sum())

# 初期値 = 現行係数。境界張り付き回避のため上限を大きく緩める
x0 = [CUR['mcoef'], CUR['depcoef'], CUR['alpha'], CUR['dcoef'], CUR['C'], math.log(0.6)]
bounds = [(0.1,4.0),(-0.02,0.02),(0.3,6.0),(0.0,0.03),(-8,5),(math.log(0.2),math.log(2.0))]
print('\nTobit フィット中...')
res = minimize(negloglik, x0, method='L-BFGS-B', bounds=bounds,
               options={'maxiter':500,'ftol':1e-10})
mcoef, depcoef, alpha, dcoef, C, log_sigma = res.x
sigma = math.exp(log_sigma)

# 境界張り付きチェック
for nm, v, (lo,hi) in zip(['mcoef','depcoef','alpha','dcoef','C','log_sigma'],
                          res.x, bounds):
    if abs(v-lo) < 1e-4 or abs(v-hi) < 1e-4:
        print(f'  ⚠️ {nm} が境界に張り付き: {v:.4f} (範囲 {lo}..{hi})')

# felt値へのフィット精度（値を無視していないかの検証）
mu_f = (mu_of(res.x[:5]))[felt_mask]
rmse_f = float(np.sqrt(np.mean((yf - mu_f)**2)))
bias_f = float(np.mean(mu_f - yf))
print(f'  収束: {res.success}  negLL={res.fun:,.0f}  iter={res.nit}')
print(f'  有感点 fit: RMSE={rmse_f:.3f}  bias={bias_f:+.3f}  (mu vs 実測震度)')
print('\n── フィット結果（現行 → 新）──')
print(f'  mcoef  (M係数)      : {CUR["mcoef"]:.4f}  →  {mcoef:.4f}')
print(f'  depcoef(深さ係数)   : {CUR["depcoef"]:.4f}  →  {depcoef:.4f}')
print(f'  alpha  (幾何減衰)   : {CUR["alpha"]:.4f}  →  {alpha:.4f}')
print(f'  dcoef  (非弾性減衰) : {CUR["dcoef"]:.4f}  →  {dcoef:.4f}')
print(f'  C      (オフセット) : {CUR["C"]:.4f}  →  {C:.4f}')
print(f'  sigma               :     ----    →  {sigma:.4f}')

# ── 新prior の 0.5 等値線（震度1の縁）を現行と比較 ──
def contour_dist(p_mcoef,p_dep,p_alpha,p_dcoef,p_C, mag, depth=10.0, target=0.5):
    dd = np.arange(0, 1000, 1.0)
    DD = np.sqrt(dd**2 + max(depth,1.0)**2)
    rr0 = 0.0028*10.0**(0.5*mag)
    val = p_mcoef*mag + p_dep*depth - p_alpha*np.log10(DD+rr0) - p_dcoef*DD + p_C
    below = np.where(val < target)[0]
    return dd[below[0]] if len(below) else 999.0

print('\n── 震度1の縁（prior=0.5となる震央距離, depth=10km）──')
print(f'  {"M":>4} {"現行縁":>8} {"新縁":>8} {"半径比":>7} {"面積比":>7}')
for mag in [4.0,4.5,5.0,5.5,6.0,6.5,7.0]:
    c_cur = contour_dist(CUR['mcoef'],CUR['depcoef'],CUR['alpha'],CUR['dcoef'],CUR['C'], mag)
    c_new = contour_dist(mcoef,depcoef,alpha,dcoef,C, mag)
    rr = c_new/c_cur if c_cur>0 else float('nan')
    print(f'  {mag:>4.1f} {c_cur:>7.0f}km {c_new:>7.0f}km {rr:>7.2f} {rr*rr:>7.2f}')

print('\n=== 新係数（tools.py / analyze_*.py に反映する値）===')
print(f'  mu = {mcoef:.4f}*M + {depcoef:.5f}*depth - {alpha:.4f}*log10(D+R0) - {dcoef:.5f}*D + {C:.4f}')
