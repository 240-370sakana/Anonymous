# Created: 2026-06-16 JST
"""
震度分布の「それっぽさ」指標
  指標1: 最大震度誤差 = |max(pred) - max(obs)|  （地震ごと）
  指標2: 面積比 = count(pred >= N) / count(obs >= N)  （震度N=1,2,3,4 ごと）
         > 1.0 なら過大（広すぎ）、< 1.0 なら過小（狭すぎ）、1.0 が理想
         ※ 面積比の分子は打ち切り点も含む（地図上での見え方を再現）
"""
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.stdout.reconfigure(encoding='utf-8')

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--ckpt', default='best_model.pt',
                 help='checkpoints/ 配下のパス (例: exp22/best_model.pt)')
_ap.add_argument('--prior', choices=['old','new'], default='new',
                 help='距離減衰式: old=Si-Midorikawa原式, new=exp29 Tobit再フィット')
_args = _ap.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EQ_PATH  = PROJECT_ROOT / 'earthquakes.parquet'
OBS_PATH = PROJECT_ROOT / 'observations.parquet'
CKPT     = PROJECT_ROOT / 'checkpoints' / _args.ckpt

HIDDEN = 256; N_HEADS = 4; BATCH_SIZE = 32
CENS_DIST_MIN_KM = 200.0; N_CENS_FAR = 500
C_OFFSET = 0.211
print(f'CKPT={_args.ckpt}  prior={_args.prior}')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── データ読み込み ──
print('データ読み込み中...')
eq  = pd.read_parquet(EQ_PATH).dropna(subset=['magnitude','hypo_lat','hypo_lon','depth'])
obs = pd.read_parquet(OBS_PATH)

rng = np.random.default_rng(seed=42)
all_ids = eq['event_id'].to_numpy(dtype=str).copy()
rng.shuffle(all_ids)
n = len(all_ids)
test_ids = set(all_ids[int(n * 0.925):])

obs_merged = obs.merge(
    eq[['event_id','hypo_lat','hypo_lon','depth','magnitude']],
    on='event_id', how='inner'
).dropna(subset=['obs_lat','obs_lon'])

all_stations   = sorted(obs_merged['station_id'].unique())
station_to_idx = {sid: i for i, sid in enumerate(all_stations)}
N_STATIONS     = len(station_to_idx)
obs_merged['station_idx'] = obs_merged['station_id'].map(station_to_idx)

sta_master_df = (obs_merged
    .groupby('station_id')[['obs_lat','obs_lon','station_idx']]
    .first().reset_index()
    .sort_values('station_idx').reset_index(drop=True))
STA_MASTER_POS    = sta_master_df[['obs_lat','obs_lon']].values.astype('float64')
STA_LAT_RAD       = np.radians(STA_MASTER_POS[:, 0])
STA_LON_RAD       = np.radians(STA_MASTER_POS[:, 1])
STA_MASTER_POS_F32 = STA_MASTER_POS.astype('float32')

eq_lookup = eq.set_index('event_id')

def haversine_km_vec(lat_deg, lon_deg):
    lat_r = math.radians(lat_deg); lon_r = math.radians(lon_deg)
    dlat = STA_LAT_RAD - lat_r; dlon = STA_LON_RAD - lon_r
    a = (np.sin(dlat/2)**2
         + math.cos(lat_r)*np.cos(STA_LAT_RAD)*np.sin(dlon/2)**2)
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

def attenuation_prior(d_epi_km, depth_km, mag):
    D  = np.sqrt(d_epi_km**2 + max(depth_km, 1.0)**2)
    R0 = 0.0028 * 10.0**(0.5*mag)
    if _args.prior == 'old':   # Si-Midorikawa(1999)原式（exp28以前）
        return (0.58*mag + 0.0038*depth_km
                - np.log10(D + R0) - 0.002*D + C_OFFSET).astype('float32')
    # new: exp29 Tobit再フィット係数（学習側 tools.py と一致させること）
    return (1.7022*mag + 0.00695*depth_km
            - 3.0161*np.log10(D + R0) - 0.00564*D - 1.2175).astype('float32')

# ── Dataset ──
class TestDataset(Dataset):
    def __init__(self, event_ids, obs_df, eq_df):
        eq_sub  = eq_df[eq_df['event_id'].isin(event_ids)].set_index('event_id')
        obs_sub = obs_df[obs_df['event_id'].isin(event_ids)]
        self.rng = np.random.default_rng(seed=0)
        self.samples = []
        for eid, grp in obs_sub.groupby('event_id'):
            if eid not in eq_sub.index: continue
            src = eq_sub.loc[eid]
            self.samples.append({
                'event_id': eid,
                'src': [float(src['hypo_lat']), float(src['hypo_lon']),
                        float(src['depth']), float(src['magnitude'])],
                'sta_idx_obs':   grp['station_idx'].values.astype('int64'),
                'intensity_obs': grp['intensity'].values.astype('float32'),
            })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        src = list(s['src'])
        dists_km = haversine_km_vec(src[0], src[1])
        non_obs  = np.ones(N_STATIONS, dtype=bool)
        non_obs[s['sta_idx_obs']] = False
        far_mask = non_obs & (dists_km >= CENS_DIST_MIN_KM)
        far_idx  = np.where(far_mask)[0]
        if len(far_idx) > N_CENS_FAR:
            cens_idx = self.rng.choice(far_idx, N_CENS_FAR, replace=False)
        else:
            cens_idx = far_idx

        n_obs = len(s['sta_idx_obs']); n_cens = len(cens_idx)
        n_total = n_obs + n_cens
        all_sta = np.empty(n_total, dtype=np.int64)
        all_sta[:n_obs] = s['sta_idx_obs']; all_sta[n_obs:] = cens_idx

        all_pos = STA_MASTER_POS_F32[all_sta]
        d_epi   = dists_km[all_sta]
        prior   = attenuation_prior(d_epi, src[2], src[3])
        hypo_dist = np.sqrt(d_epi**2 + max(src[2], 1.0)**2)
        log_dist  = np.log10(hypo_dist + 1.0).astype('float32')
        dlat = all_pos[:, 0] - src[0]; dlon = all_pos[:, 1] - src[1]
        az_rad = np.arctan2(dlon, dlat)
        sin_az = np.sin(az_rad).astype('float32')
        cos_az = np.cos(az_rad).astype('float32')
        all_pos_p = np.column_stack([all_pos, prior, log_dist, sin_az, cos_az])

        all_int = np.zeros(n_total, dtype='float32')
        all_int[:n_obs] = s['intensity_obs']
        is_cens = np.ones(n_total, dtype=bool)
        is_cens[:n_obs] = (s['intensity_obs'] < 0.5)

        return (torch.tensor(src, dtype=torch.float32),
                torch.tensor(all_sta, dtype=torch.int64),
                torch.tensor(all_pos_p, dtype=torch.float32),
                torch.tensor(all_int, dtype=torch.float32),
                torch.tensor(is_cens, dtype=torch.bool),
                n_obs)

def collate_fn(batch):
    srcs, sta_idxs, obs_poss, tgts, censs, n_obs_list = zip(*batch)
    B = len(srcs); maxN = max(t.size(0) for t in tgts)
    srcs_t    = torch.stack(srcs)
    obs_pos_t = torch.zeros(B, maxN, 6, dtype=torch.float32)
    sta_idx_t = torch.zeros(B, maxN, dtype=torch.int64)
    tgt_t     = torch.zeros(B, maxN, dtype=torch.float32)
    cens_t    = torch.ones( B, maxN, dtype=torch.bool)
    mask_t    = torch.ones( B, maxN, dtype=torch.bool)
    for i, (sp, si, op, tg, ce) in enumerate(zip(srcs, sta_idxs, obs_poss, tgts, censs)):
        N = tg.size(0)
        obs_pos_t[i,:N] = op; sta_idx_t[i,:N] = si
        tgt_t[i,:N] = tg; cens_t[i,:N] = ce; mask_t[i,:N] = False
    return srcs_t, obs_pos_t, sta_idx_t, tgt_t, cens_t, mask_t, n_obs_list

# ── モデル定義 ──
def make_mlp(in_dim, hidden, out_dim, n_layers):
    layers = [nn.Linear(in_dim, hidden), nn.GELU()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)

class SeismicModel(nn.Module):
    def __init__(self, hidden=256, n_heads=4, src_layers=3, sta_layers=2, n_stations=8000):
        super().__init__()
        self.src_enc   = make_mlp(4, hidden, hidden, src_layers)
        self.sta_enc   = make_mlp(7, hidden, hidden, sta_layers)
        self.sta_embed = nn.Embedding(n_stations, hidden)
        self.attn      = nn.MultiheadAttention(hidden, n_heads, batch_first=True, dropout=0.1)
        self.norm      = nn.LayerNorm(hidden)
        sa_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden*2,
            dropout=0.1, batch_first=True, activation='gelu')
        self.self_attn = nn.TransformerEncoder(sa_layer, num_layers=1)
        self.head      = nn.Linear(hidden, 1)

    def forward(self, src, obs_pos, sta_idx, mask):
        ctx      = self.src_enc(src).unsqueeze(1)
        pos2     = obs_pos[:,:,:2]
        prior    = obs_pos[:,:,2]
        extra    = obs_pos[:,:,3:6]
        hypo_pos = src[:,:2].unsqueeze(1).expand_as(pos2)
        delta    = pos2 - hypo_pos
        obs_feat = torch.cat([pos2, delta, extra], dim=-1)
        q        = self.sta_enc(obs_feat) + self.sta_embed(sta_idx)
        out, _   = self.attn(q, ctx, ctx, key_padding_mask=None)
        out      = self.norm(out + q)
        out      = self.self_attn(out, src_key_padding_mask=mask)
        return self.head(out).squeeze(-1) + prior

# ── テスト実行 ──
print('Dataset 構築中...')
test_ds     = TestDataset(test_ids, obs_merged, eq)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

model = SeismicModel(HIDDEN, N_HEADS, n_stations=N_STATIONS).to(DEVICE)
ckpt  = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()
print(f'チェックポイント: val_mae={ckpt.get("val_mae",float("nan")):.4f}')
print(f'テスト地震数: {len(test_ds):,}')

# ── 地震ごとの指標を収集 ──
print('\n地震ごとの指標を計算中...')

THRESHOLDS = [0.5, 1.5, 2.5, 3.5, 4.5]
THRESH_LABELS = ['>=0.5 (震度1+)', '>=1.5 (震度2+)', '>=2.5 (震度3+)',
                 '>=3.5 (震度4+)', '>=4.5 (震度5弱+)']

results = []
all_pred_obs, all_tgt_obs, all_depth_obs = [], [], []

with torch.no_grad():
    for src, obs_pos, sta_idx, tgt, is_cens, mask, n_obs_list in test_loader:
        src=src.to(DEVICE); obs_pos=obs_pos.to(DEVICE); sta_idx=sta_idx.to(DEVICE)
        tgt=tgt.to(DEVICE); is_cens=is_cens.to(DEVICE); mask=mask.to(DEVICE)
        pred = model(src, obs_pos, sta_idx, mask)

        B = src.shape[0]
        for i in range(B):
            valid   = ~mask[i]
            obs_m   = valid & ~is_cens[i]
            cens_m  = valid & is_cens[i]

            if obs_m.sum() == 0:
                continue

            p_obs  = pred[i, obs_m].cpu().numpy()
            t_obs  = tgt[i, obs_m].cpu().numpy()
            p_cens = pred[i, cens_m].cpu().numpy() if cens_m.any() else np.array([])
            mag    = src[i, 3].item()
            depth  = src[i, 2].item()

            all_pred_obs.append(p_obs)
            all_tgt_obs.append(t_obs)
            all_depth_obs.append(np.full(len(t_obs), depth, dtype='float32'))

            max_pred = p_obs.max()
            max_obs  = t_obs.max()
            max_err  = abs(max_pred - max_obs)
            max_bias = max_pred - max_obs

            area_ratios = {}
            for thr in THRESHOLDS:
                n_obs_above  = (t_obs >= thr).sum()
                n_pred_obs   = (p_obs >= thr).sum()
                n_pred_cens  = (p_cens >= thr).sum() if len(p_cens) > 0 else 0
                n_pred_total = n_pred_obs + n_pred_cens

                if n_obs_above > 0:
                    ratio_obs_only = n_pred_obs / n_obs_above
                    ratio_total    = n_pred_total / n_obs_above
                else:
                    ratio_obs_only = float('nan')
                    ratio_total    = float('nan')

                area_ratios[thr] = {
                    'n_obs': int(n_obs_above),
                    'n_pred_obs': int(n_pred_obs),
                    'n_pred_cens': int(n_pred_cens),
                    'ratio_obs': ratio_obs_only,
                    'ratio_total': ratio_total,
                }

            results.append({
                'mag': mag, 'depth': depth,
                'max_obs': max_obs, 'max_pred': max_pred,
                'max_err': max_err, 'max_bias': max_bias,
                'n_obs': int(obs_m.sum()),
                'area': area_ratios,
            })

print(f'  集計完了: {len(results):,} 地震')

# ── 基本指標: MAE と 平均偏差 (bias) ──
ap = np.concatenate(all_pred_obs)
at = np.concatenate(all_tgt_obs)
ad = np.concatenate(all_depth_obs)
residual = ap - at

print('\n' + '=' * 75)
print('基本指標: MAE（平均絶対誤差）と Bias（平均偏差 = mean(pred - obs)）')
print('  bias > 0: 過大予測傾向、bias < 0: 過小予測傾向')
print('=' * 75)
print(f'  全体: MAE={np.abs(residual).mean():.4f}  bias={residual.mean():+.4f}  n={len(residual):,}')

print(f'\n  {"震度帯":<24} {"MAE":>6} {"bias":>8} {"n":>10}')
print('-' * 55)
for lo, hi, label in [(0.5,1.5,'0.5-1.4 (震度1)'),(1.5,2.5,'1.5-2.4 (震度2)'),
                      (2.5,3.5,'2.5-3.4 (震度3)'),(3.5,4.5,'3.5-4.4 (震度4)'),
                      (4.5,9.0,'4.5+    (震度5弱+)')]:
    m = (at >= lo) & (at < hi)
    if m.sum() > 0:
        r = residual[m]
        print(f'  {label:<24} {np.abs(r).mean():>6.4f} {r.mean():>+8.4f} {m.sum():>10,}')

print(f'\n  {"深さ帯":<24} {"MAE":>6} {"bias":>8} {"n":>10}')
print('-' * 55)
for lo, hi, label in [(0,30,'h<30km'),(30,80,'30-80km'),(80,200,'80-200km'),
                      (200,9999,'h>=200km')]:
    m = (ad >= lo) & (ad < hi)
    if m.sum() > 0:
        r = residual[m]
        print(f'  {label:<24} {np.abs(r).mean():>6.4f} {r.mean():>+8.4f} {m.sum():>10,}')

# ── 緊急地震速報との比較用: 震度4以上のみ ──
print(f'\n  --- 参考: 震度4以上のみ（EEW警報と条件を揃えた比較） ---')
m4 = (at >= 3.5)
if m4.sum() > 0:
    r4 = residual[m4]
    print(f'  震度4以上: MAE={np.abs(r4).mean():.4f}  bias={r4.mean():+.4f}  n={m4.sum():,}')

# ── 指標1: 最大震度誤差 ──
print('\n' + '=' * 75)
print('指標1: 最大震度誤差 |max(pred) - max(obs)|')
print('=' * 75)

max_errs = np.array([r['max_err'] for r in results])
max_bias = np.array([r['max_bias'] for r in results])
print(f'  全体: MAE={max_errs.mean():.3f}  med={np.median(max_errs):.3f}'
      f'  90%={np.percentile(max_errs, 90):.3f}  bias={max_bias.mean():+.3f}  n={len(max_errs):,}')

print(f'\n  {"M帯":<12} {"MAE":>6} {"med":>6} {"90%":>6} {"bias":>7} {"n":>7}')
print('-' * 55)
for lo, hi, label in [(3,4,'M3-4'),(4,5,'M4-5'),(5,6,'M5-6'),(6,7,'M6-7'),(7,9,'M7+')]:
    m = np.array([lo <= r['mag'] < hi for r in results])
    if m.sum() > 0:
        e = max_errs[m]; b = max_bias[m]
        print(f'  {label:<12} {e.mean():>6.3f} {np.median(e):>6.3f}'
              f' {np.percentile(e,90):>6.3f} {b.mean():>+7.3f} {m.sum():>7,}')

print(f'\n  {"深さ帯":<16} {"MAE":>6} {"med":>6} {"90%":>6} {"bias":>7} {"n":>7}')
print('-' * 60)
for lo, hi, label in [(0,30,'h<30km'),(30,80,'30-80km'),(80,150,'80-150km'),
                       (150,300,'150-300km'),(300,9999,'h>=300km')]:
    m = np.array([lo <= r['depth'] < hi for r in results])
    if m.sum() > 0:
        e = max_errs[m]; b = max_bias[m]
        print(f'  {label:<16} {e.mean():>6.3f} {np.median(e):>6.3f}'
              f' {np.percentile(e,90):>6.3f} {b.mean():>+7.3f} {m.sum():>7,}')

# ── 指標2: 面積比 ──
print('\n' + '=' * 75)
print('指標2: 面積比 count(pred >= N) / count(obs >= N)')
print('       > 1.0 = 過大（広すぎ）、< 1.0 = 過小（狭すぎ）')
print('       total = 観測点+打ち切り点（地図上の見え方）')
print('       obs   = 観測点のみ（震度の精度）')
print('=' * 75)

for thr, label in zip(THRESHOLDS, THRESH_LABELS):
    valid = [(r['area'][thr]['ratio_total'], r['area'][thr]['ratio_obs'],
              r['area'][thr]['n_obs'], r['mag'])
             for r in results if not math.isnan(r['area'][thr]['ratio_total'])]
    if not valid:
        print(f'\n  {label}: データなし')
        continue

    totals = np.array([v[0] for v in valid])
    obs_only = np.array([v[1] for v in valid])
    n_quakes = len(valid)

    print(f'\n  {label}  (n={n_quakes:,} 地震)')
    print(f'    total面積比: mean={totals.mean():.2f}  med={np.median(totals):.2f}'
          f'  25%={np.percentile(totals,25):.2f}  75%={np.percentile(totals,75):.2f}')
    print(f'    obs面積比  : mean={obs_only.mean():.2f}  med={np.median(obs_only):.2f}'
          f'  25%={np.percentile(obs_only,25):.2f}  75%={np.percentile(obs_only,75):.2f}')

    # M帯別
    print(f'    {"M帯":<8} {"total_mean":>10} {"total_med":>10} {"obs_mean":>10} {"n":>6}')
    for mlo, mhi, mlabel in [(3,4,'M3-4'),(4,5,'M4-5'),(5,6,'M5-6'),(6,9,'M6+')]:
        sub = [(t, o) for t, o, _, mg in valid if mlo <= mg < mhi]
        if len(sub) >= 5:
            st = np.array([s[0] for s in sub])
            so = np.array([s[1] for s in sub])
            print(f'    {mlabel:<8} {st.mean():>10.2f} {np.median(st):>10.2f}'
                  f' {so.mean():>10.2f} {len(sub):>6,}')

# ── 指標2b: 階級別 面積比（累積ではなく各階級の帯ごと） ──
print('\n' + '=' * 75)
print('指標2b: 階級別 面積比  count(pred in 階級N) / count(obs in 階級N)')
print('        各階級を独立に評価（累積ではない）。total=観測点+打ち切り点。')
print('        > 1.0 = その階級の面積が広すぎ、< 1.0 = 狭すぎ')
print('=' * 75)

# 階級 k: [THRESHOLDS[k], THRESHOLDS[k+1]) 、最上位は [4.5, inf)
CLASS_DEFS = [(1,0.5,1.5),(2,1.5,2.5),(3,2.5,3.5),(4,3.5,4.5),(5,4.5,None)]
CLASS_LABEL = {1:'震度1',2:'震度2',3:'震度3',4:'震度4',5:'震度5弱+'}

def cum(area, thr):
    """area dict から閾値 thr の累積カウント (obs, pred_total) を返す"""
    a = area[thr]
    return a['n_obs'], a['n_pred_obs'] + a['n_pred_cens']

# 地震ごとの階級別比 + マクロ合算
class_ratios = {k: [] for k,_,_ in CLASS_DEFS}
macro_pred = {k: 0 for k,_,_ in CLASS_DEFS}
macro_obs  = {k: 0 for k,_,_ in CLASS_DEFS}

for r in results:
    area = r['area']
    for k, lo, hi in CLASS_DEFS:
        o_lo, p_lo = cum(area, lo)
        if hi is None:
            o_band, p_band = o_lo, p_lo
        else:
            o_hi, p_hi = cum(area, hi)
            o_band, p_band = o_lo - o_hi, p_lo - p_hi
        macro_pred[k] += p_band
        macro_obs[k]  += o_band
        if o_band > 0:
            class_ratios[k].append(p_band / o_band)

print(f'  {"階級":<8} {"地震毎mean":>10} {"med":>6} {"総pred":>10} {"総obs":>10} {"総比(macro)":>11} {"n地震":>7}')
print('-' * 70)
for k,_,_ in CLASS_DEFS:
    r = np.array(class_ratios[k])
    rm  = r.mean()     if len(r) else float('nan')
    rmd = np.median(r) if len(r) else float('nan')
    macro = (macro_pred[k]/macro_obs[k]) if macro_obs[k] > 0 else float('nan')
    print(f'  {CLASS_LABEL[k]:<8} {rm:>10.2f} {rmd:>6.2f} '
          f'{macro_pred[k]:>10,} {macro_obs[k]:>10,} {macro:>11.2f} {len(r):>7,}')
print('  (理想=1.00。震度1>1かつ震度2-4<1なら「裾広・頂低」= 勾配が緩い)')

# ── 面積比の分布（過大/適正/過小） ──
print('\n' + '=' * 75)
print('面積比の分布: 過大 (>1.5) / 適正 (0.67-1.5) / 過小 (<0.67)')
print('=' * 75)
print(f'  {"閾値":<20} {"過小%":>7} {"適正%":>7} {"過大%":>7} {"n":>7}')
print('-' * 60)
for thr, label in zip(THRESHOLDS, THRESH_LABELS):
    valid_t = [r['area'][thr]['ratio_total']
               for r in results if not math.isnan(r['area'][thr]['ratio_total'])]
    if len(valid_t) < 10: continue
    arr = np.array(valid_t)
    under = (arr < 0.67).mean() * 100
    ok    = ((arr >= 0.67) & (arr <= 1.5)).mean() * 100
    over  = (arr > 1.5).mean() * 100
    print(f'  {label:<20} {under:>6.1f}% {ok:>6.1f}% {over:>6.1f}% {len(arr):>7,}')

print('\n分析完了。')
