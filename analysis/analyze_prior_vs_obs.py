# Created: 2026-06-18 00:30 JST
"""
Si-Midorikawa(1999) prior vs 実測震度の距離プロット
prior の距離減衰カーブが実データとどこでズレているかを可視化する。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
C_OFFSET = 0.211

eq  = pd.read_parquet(DATA_DIR / 'earthquakes.parquet').dropna(
        subset=['hypo_lat','hypo_lon','depth','magnitude'])
obs = pd.read_parquet(DATA_DIR / 'observations.parquet').dropna(
        subset=['intensity'])

print(f'地震数: {len(eq):,}  観測レコード: {len(obs):,}')
print(f'観測震度>=0.5のレコード: {(obs["intensity"]>=0.5).sum():,}')

merged = obs.merge(eq[['event_id','hypo_lat','hypo_lon','depth','magnitude']],
                   on='event_id')

def haversine_km_vec(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2
         + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))
         *np.sin(dlon/2)**2)
    return R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

merged['d_epi_km'] = haversine_km_vec(
    merged['hypo_lat'].values, merged['hypo_lon'].values,
    merged['obs_lat'].values, merged['obs_lon'].values)
merged['d_hypo_km'] = np.sqrt(merged['d_epi_km']**2 + np.maximum(merged['depth'], 1.0)**2)

def si_midorikawa_prior(d_epi_km, depth_km, mag):
    D  = np.sqrt(d_epi_km**2 + np.maximum(depth_km, 1.0)**2)
    R0 = 0.0028 * 10.0**(0.5*mag)
    return 0.58*mag + 0.0038*depth_km - np.log10(D + R0) - 0.002*D + C_OFFSET

merged['prior'] = si_midorikawa_prior(
    merged['d_epi_km'].values, merged['depth'].values, merged['magnitude'].values)

obs_only = merged[merged['intensity'] >= 0.5].copy()
obs_only['residual'] = obs_only['intensity'] - obs_only['prior']

print(f'\n有感観測数: {len(obs_only):,}')
print(f'残差 (実測-prior) 統計:')
print(f'  mean = {obs_only["residual"].mean():.4f}')
print(f'  std  = {obs_only["residual"].std():.4f}')
print(f'  median = {obs_only["residual"].median():.4f}')

# --- 距離ビン別の統計 ---
dist_bins = [0, 20, 50, 100, 150, 200, 300, 500, 1000]
obs_only['dist_bin'] = pd.cut(obs_only['d_hypo_km'], bins=dist_bins)

print('\n── 距離ビン別: prior vs 実測 ──')
print(f'{"距離帯":<20s} {"N":>7s} {"prior平均":>10s} {"実測平均":>10s} {"残差平均":>10s} {"残差std":>10s}')
for name, grp in obs_only.groupby('dist_bin', observed=True):
    if len(grp) == 0: continue
    print(f'{str(name):<20s} {len(grp):>7,d} {grp["prior"].mean():>10.3f} '
          f'{grp["intensity"].mean():>10.3f} {grp["residual"].mean():>10.3f} '
          f'{grp["residual"].std():>10.3f}')

# --- マグニチュード帯別にも ---
mag_bins = [3.0, 4.0, 5.0, 6.0, 9.0]
obs_only['mag_bin'] = pd.cut(obs_only['magnitude'], bins=mag_bins)

print('\n── マグニチュード帯別: prior vs 実測 ──')
print(f'{"M帯":<15s} {"N":>7s} {"prior平均":>10s} {"実測平均":>10s} {"残差平均":>10s}')
for name, grp in obs_only.groupby('mag_bin', observed=True):
    if len(grp) == 0: continue
    print(f'{str(name):<15s} {len(grp):>7,d} {grp["prior"].mean():>10.3f} '
          f'{grp["intensity"].mean():>10.3f} {grp["residual"].mean():>10.3f}')

# --- 距離 x マグニチュード帯 ---
print('\n── 距離×M帯別 残差平均 (実測-prior) ──')
pivot = obs_only.pivot_table(values='residual', index='dist_bin',
                              columns='mag_bin', aggfunc='mean',
                              observed=True)
print(pivot.to_string(float_format='{:.3f}'.format))

# --- プロット ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 散布図: 距離 vs prior & 実測
ax = axes[0, 0]
sample = obs_only.sample(min(30000, len(obs_only)), random_state=42)
ax.scatter(sample['d_hypo_km'], sample['intensity'], alpha=0.1, s=1, c='red', label='observed')
ax.scatter(sample['d_hypo_km'], sample['prior'], alpha=0.1, s=1, c='blue', label='prior')
ax.set_xlabel('Hypocentral Distance (km)')
ax.set_ylabel('Intensity')
ax.set_title('Prior vs Observed (scatter)')
ax.legend()
ax.set_xlim(0, 600)
ax.set_ylim(-1, 8)

# 2. 距離ビン別の平均曲線
ax = axes[0, 1]
fine_bins = np.arange(0, 601, 20)
obs_only['fine_dist'] = pd.cut(obs_only['d_hypo_km'], bins=fine_bins)
means = obs_only.groupby('fine_dist', observed=True).agg(
    prior_mean=('prior', 'mean'),
    obs_mean=('intensity', 'mean'),
    count=('intensity', 'count')
).reset_index()
means['dist_center'] = [(b.left + b.right)/2 for b in means['fine_dist']]
means = means[means['count'] >= 50]
ax.plot(means['dist_center'], means['prior_mean'], 'b-o', ms=3, label='prior mean')
ax.plot(means['dist_center'], means['obs_mean'], 'r-o', ms=3, label='observed mean')
ax.axhline(y=0.5, color='gray', ls='--', alpha=0.5, label='intensity 1 threshold')
ax.set_xlabel('Hypocentral Distance (km)')
ax.set_ylabel('Mean Intensity')
ax.set_title('Mean Prior vs Observed by Distance')
ax.legend()
ax.set_xlim(0, 600)

# 3. 残差 (実測-prior) の距離依存性
ax = axes[1, 0]
ax.plot(means['dist_center'], means['obs_mean'] - means['prior_mean'],
        'k-o', ms=3)
ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Hypocentral Distance (km)')
ax.set_ylabel('Residual (Obs - Prior)')
ax.set_title('Residual by Distance')
ax.set_xlim(0, 600)

# 4. M帯別の残差カーブ
ax = axes[1, 1]
colors = ['green', 'blue', 'orange', 'red']
for (mname, mgrp), c in zip(obs_only.groupby('mag_bin', observed=True), colors):
    if len(mgrp) < 100: continue
    m_means = mgrp.groupby('fine_dist', observed=True).agg(
        prior_mean=('prior', 'mean'),
        obs_mean=('intensity', 'mean'),
        count=('intensity', 'count')
    ).reset_index()
    m_means['dist_center'] = [(b.left + b.right)/2 for b in m_means['fine_dist']]
    m_means = m_means[m_means['count'] >= 20]
    residual = m_means['obs_mean'] - m_means['prior_mean']
    ax.plot(m_means['dist_center'], residual, '-o', ms=2, c=c, label=str(mname))
ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Hypocentral Distance (km)')
ax.set_ylabel('Residual (Obs - Prior)')
ax.set_title('Residual by Distance × Magnitude')
ax.legend()
ax.set_xlim(0, 600)

plt.tight_layout()
plt.savefig(DATA_DIR / 'prior_vs_obs_analysis.png', dpi=150)
print(f'\nプロット保存: prior_vs_obs_analysis.png')
# plt.show()  # non-interactive
