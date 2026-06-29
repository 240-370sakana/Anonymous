# Created: 2026-06-18 01:10 JST
"""
真の距離減衰の測定（選択バイアスなし）

analyze_prior_vs_obs.py は有感観測(>=0.5)のみで集計したため、遠方では
「たまたま揺れた点」しか入らず選択バイアスがあった。

ここでは各地震について、マスタ観測点リスト全点を母集団とし、
  - 有感(obsに存在)         → その実測震度
  - 非有感(obsに無い=censored) → 震度0 とみなす
として、距離×M帯ごとに「真の平均震度(0込み)」「有感率 P(>=0.5)」を集計する。
これと prior を同じ母集団で比較すれば、減衰カーブの形のズレが分かる。
"""
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
C_OFFSET = 0.211
MAX_DIST = 800.0          # 集計する最大震央距離(km)
N_SAMPLE_EQ = 8000        # サブサンプルする地震数(速度のため)

print('データ読み込み中...')
eq  = pd.read_parquet(ROOT/'earthquakes.parquet').dropna(
        subset=['hypo_lat','hypo_lon','depth','magnitude'])
obs = pd.read_parquet(ROOT/'observations.parquet').dropna(subset=['obs_lat','obs_lon'])

# マスタ観測点（データ内で一度でも観測した点 = モデルと同じ母集団）
sta = (obs.groupby('station_id')[['obs_lat','obs_lon']].first().reset_index())
STA_LAT = np.radians(sta['obs_lat'].values)
STA_LON = np.radians(sta['obs_lon'].values)
STA_LAT_DEG = sta['obs_lat'].values
N_STA = len(sta)
sid_to_idx = {s:i for i,s in enumerate(sta['station_id'].values)}
print(f'マスタ観測点数: {N_STA:,}')

# 地震ごとの {station_idx: intensity}
obs['sidx'] = obs['station_id'].map(sid_to_idx)
felt = obs.groupby('event_id').apply(
    lambda g: (g['sidx'].values, g['intensity'].values), include_groups=False
).to_dict()

eq = eq[eq['event_id'].isin(felt.keys())].reset_index(drop=True)
if len(eq) > N_SAMPLE_EQ:
    eq = eq.sample(N_SAMPLE_EQ, random_state=42).reset_index(drop=True)
print(f'集計対象地震数: {len(eq):,}')

def haversine_km(lat_deg, lon_deg):
    lat_r = math.radians(lat_deg); lon_r = math.radians(lon_deg)
    dlat = STA_LAT - lat_r; dlon = STA_LON - lon_r
    a = np.sin(dlat/2)**2 + math.cos(lat_r)*np.cos(STA_LAT)*np.sin(dlon/2)**2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a,0,1)))

def prior_fn(d_epi, depth, mag, c_off=C_OFFSET, mcoef=0.58, dcoef=0.002):
    D  = np.sqrt(d_epi**2 + max(depth,1.0)**2)
    R0 = 0.0028 * 10.0**(0.5*mag)
    return mcoef*mag + 0.0038*depth - np.log10(D + R0) - dcoef*D + c_off

# 距離ビン×M帯で集計
DIST_EDGES = np.arange(0, MAX_DIST+1e-6, 20.0)
NB = len(DIST_EDGES)-1
MBANDS = [(3,4,'M3-4'),(4,5,'M4-5'),(5,6,'M5-6'),(6,9,'M6+')]

# accum[mband] = dict of arrays per dist bin
acc = {lab: {'n':np.zeros(NB),'felt':np.zeros(NB),
             'sum_int':np.zeros(NB),'sum_prior':np.zeros(NB)} for *_,lab in MBANDS}

for row in eq.itertuples():
    mag = row.magnitude; depth = row.depth
    mlab = None
    for lo,hi,lab in MBANDS:
        if lo <= mag < hi: mlab = lab; break
    if mlab is None: continue

    d = haversine_km(row.hypo_lat, row.hypo_lon)   # 全観測点への震央距離
    within = d < MAX_DIST
    d_in = d[within]
    idx_in = np.where(within)[0]

    intensity = np.zeros(len(idx_in), dtype='float32')  # 既定=非有感=0
    f_idx, f_int = felt[row.event_id]
    # within内の有感点に実測震度を入れる
    pos = {gi:j for j,gi in enumerate(idx_in)}
    for gi, iv in zip(f_idx, f_int):
        j = pos.get(gi)
        if j is not None: intensity[j] = iv

    pr = prior_fn(d_in, depth, mag)

    b = np.clip(np.digitize(d_in, DIST_EDGES)-1, 0, NB-1)
    a = acc[mlab]
    np.add.at(a['n'],       b, 1)
    np.add.at(a['felt'],    b, (intensity>=0.5).astype(float))
    np.add.at(a['sum_int'], b, intensity)
    np.add.at(a['sum_prior'],b, pr)

centers = (DIST_EDGES[:-1]+DIST_EDGES[1:])/2

print('\n真の距離減衰（censored=0込み） vs prior')
for lo,hi,lab in MBANDS:
    a = acc[lab]
    print(f'\n── {lab} ──')
    print(f'  {"距離":>8} {"N":>9} {"有感率":>7} {"真平均震度":>10} {"prior平均":>9} {"prior-真":>9}')
    for k in range(NB):
        if a['n'][k] < 30: continue
        mean_int = a['sum_int'][k]/a['n'][k]
        mean_pr  = a['sum_prior'][k]/a['n'][k]
        feltrate = a['felt'][k]/a['n'][k]
        print(f'  {centers[k]:>7.0f} {int(a["n"][k]):>9,} {feltrate:>7.2f} '
              f'{mean_int:>10.3f} {mean_pr:>9.3f} {mean_pr-mean_int:>+9.3f}')

# ── プロット ──
fig, axes = plt.subplots(2,2,figsize=(14,10))
colors = {'M3-4':'green','M4-5':'blue','M5-6':'orange','M6+':'red'}

ax = axes[0,0]
for lo,hi,lab in MBANDS:
    a=acc[lab]; m=a['n']>=30
    ax.plot(centers[m], (a['sum_int'][m]/a['n'][m]), '-o', ms=3, c=colors[lab], label=f'{lab} obs')
    ax.plot(centers[m], (a['sum_prior'][m]/a['n'][m]), '--', c=colors[lab], alpha=0.7, label=f'{lab} prior')
ax.axhline(0.5, color='gray', ls=':', alpha=0.6)
ax.set_xlabel('Epicentral Distance (km)'); ax.set_ylabel('Mean Intensity (censored=0)')
ax.set_title('True decay (solid) vs Prior (dashed)'); ax.legend(fontsize=8); ax.set_ylim(-0.5,5)

ax = axes[0,1]
for lo,hi,lab in MBANDS:
    a=acc[lab]; m=a['n']>=30
    ax.plot(centers[m], a['felt'][m]/a['n'][m], '-o', ms=3, c=colors[lab], label=lab)
ax.axhline(0.5, color='gray', ls=':', alpha=0.6)
ax.set_xlabel('Epicentral Distance (km)'); ax.set_ylabel('P(felt, intensity>=0.5)')
ax.set_title('Felt probability vs Distance'); ax.legend()

ax = axes[1,0]
for lo,hi,lab in MBANDS:
    a=acc[lab]; m=a['n']>=30
    ax.plot(centers[m], (a['sum_prior'][m]-a['sum_int'][m])/a['n'][m], '-o', ms=3, c=colors[lab], label=lab)
ax.axhline(0, color='gray', ls=':', alpha=0.6)
ax.set_xlabel('Epicentral Distance (km)'); ax.set_ylabel('Prior - True (bias)')
ax.set_title('Prior bias by distance (>0 = prior too high)'); ax.legend()

ax = axes[1,1]
for lo,hi,lab in MBANDS:
    a=acc[lab]; m=a['n']>=30
    ax.semilogy(centers[m], np.maximum(a['sum_int'][m]/a['n'][m],1e-3), '-o', ms=3, c=colors[lab], label=f'{lab} obs')
    ax.semilogy(centers[m], np.maximum(a['sum_prior'][m]/a['n'][m],1e-3), '--', c=colors[lab], alpha=0.7)
ax.set_xlabel('Epicentral Distance (km)'); ax.set_ylabel('Mean Intensity (log)')
ax.set_title('Decay slope (log scale)'); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(ROOT/'true_decay_analysis.png', dpi=150)
print(f'\nプロット保存: true_decay_analysis.png')
