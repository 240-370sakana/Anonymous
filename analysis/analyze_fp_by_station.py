# Created: 2026-06-16 JST
"""
FP（空振り）の地域偏り分析
仮説: 学習データの少ない観測点ほど FP 率が高いのではないか

出力:
  1. 観測点ごとの学習データ件数 vs FP率の相関
  2. FP率の高い観測点の地理的分布
  3. 学習件数帯別の FP率
"""
import sys
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EQ_PATH  = PROJECT_ROOT / 'earthquakes.parquet'
OBS_PATH = PROJECT_ROOT / 'observations.parquet'
CKPT     = PROJECT_ROOT / 'checkpoints' / 'best_model.pt'

HIDDEN = 256; N_HEADS = 4; BATCH_SIZE = 32
CENS_DIST_MIN_KM = 200.0; N_CENS_FAR = 500
C_OFFSET = 0.211

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ── データ読み込み ──
print('データ読み込み中...')
eq  = pd.read_parquet(EQ_PATH).dropna(subset=['magnitude','hypo_lat','hypo_lon','depth'])
obs = pd.read_parquet(OBS_PATH)

rng = np.random.default_rng(seed=42)
all_ids = eq['event_id'].to_numpy(dtype=str).copy()
rng.shuffle(all_ids)
n = len(all_ids)
train_ids = set(all_ids[:int(n * 0.85)])
test_ids  = set(all_ids[int(n * 0.925):])

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

# ── 学習データでの観測点別出現回数 ──
train_obs = obs_merged[obs_merged['event_id'].isin(train_ids)]
sta_train_counts = train_obs.groupby('station_idx').size().to_dict()
print(f'観測点数: {N_STATIONS:,}')
print(f'学習データでの観測あり: {len(sta_train_counts):,} 点')
print(f'学習データなし: {N_STATIONS - len(sta_train_counts):,} 点')

# ── Haversine / Prior ──
def haversine_km_vec(lat_deg, lon_deg):
    lat_r = math.radians(lat_deg); lon_r = math.radians(lon_deg)
    dlat = STA_LAT_RAD - lat_r; dlon = STA_LON_RAD - lon_r
    a = (np.sin(dlat/2)**2
         + math.cos(lat_r)*np.cos(STA_LAT_RAD)*np.sin(dlon/2)**2)
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

def attenuation_prior(d_epi_km, depth_km, mag):
    D  = np.sqrt(d_epi_km**2 + max(depth_km, 1.0)**2)
    R0 = 0.0028 * 10.0**(0.5*mag)
    return (0.58*mag + 0.0038*depth_km
            - np.log10(D + R0) - 0.002*D + C_OFFSET).astype('float32')

# ── Dataset (テストセットのみ) ──
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
                torch.tensor(is_cens, dtype=torch.bool))

def collate_fn(batch):
    srcs, sta_idxs, obs_poss, tgts, censs = zip(*batch)
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
    return srcs_t, obs_pos_t, sta_idx_t, tgt_t, cens_t, mask_t

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
print(f'チェックポイント読み込み完了 (val_mae={ckpt.get("val_mae",float("nan")):.4f})')

# ── 観測点別 FP 集計 ──
print('\n観測点別FP集計中...')
sta_fp_count = defaultdict(int)   # sta_idx -> FP件数
sta_fp_total = defaultdict(int)   # sta_idx -> 打ち切り総件数

with torch.no_grad():
    for src, obs_pos, sta_idx, tgt, is_cens, mask in test_loader:
        src=src.to(DEVICE); obs_pos=obs_pos.to(DEVICE); sta_idx=sta_idx.to(DEVICE)
        tgt=tgt.to(DEVICE); is_cens=is_cens.to(DEVICE); mask=mask.to(DEVICE)
        pred = model(src, obs_pos, sta_idx, mask)

        cens_valid = ~mask & is_cens
        if not cens_valid.any():
            continue

        cens_sta  = sta_idx[cens_valid].cpu().numpy()
        cens_pred = pred[cens_valid].cpu().numpy()

        for si, p in zip(cens_sta, cens_pred):
            sta_fp_total[si] += 1
            if p >= 0.5:
                sta_fp_count[si] += 1

# ── 結果集計 ──
print('\n' + '=' * 75)
print('学習データ件数帯別の FP率')
print('=' * 75)
print(f'  {"学習件数帯":<20} {"観測点数":>8} {"FP件数":>10} {"打ち切り総数":>12} {"FP率":>8}')
print('-' * 75)

count_bins = [
    (0, 0, 'n_train=0 (未学習)'),
    (1, 10, '1-10'),
    (11, 50, '11-50'),
    (51, 100, '51-100'),
    (101, 300, '101-300'),
    (301, 1000, '301-1000'),
    (1001, 99999, '1001+'),
]

for lo, hi, label in count_bins:
    stas_in_bin = []
    fp_sum = 0
    total_sum = 0
    for si in range(N_STATIONS):
        cnt = sta_train_counts.get(si, 0)
        if lo <= cnt <= hi and si in sta_fp_total:
            stas_in_bin.append(si)
            fp_sum    += sta_fp_count.get(si, 0)
            total_sum += sta_fp_total[si]
    if total_sum > 0:
        print(f'  {label:<20} {len(stas_in_bin):>8,} {fp_sum:>10,} {total_sum:>12,} {fp_sum/total_sum:>8.4f}')
    else:
        print(f'  {label:<20} {len(stas_in_bin):>8,} {"---":>10} {"---":>12} {"---":>8}')

# ── FP率 Top20 の観測点 ──
print('\n' + '=' * 75)
print('FP率 Top20 の観測点（打ち切り件数 >= 50 の観測点のみ）')
print('=' * 75)
print(f'  {"idx":>5} {"lat":>7} {"lon":>8} {"n_train":>8} {"FP件数":>7} {"打ち切り":>8} {"FP率":>7}')
print('-' * 75)

sta_fp_rates = []
for si in range(N_STATIONS):
    if sta_fp_total.get(si, 0) >= 50:
        fp_r = sta_fp_count.get(si, 0) / sta_fp_total[si]
        sta_fp_rates.append((si, fp_r, sta_fp_count.get(si, 0), sta_fp_total[si]))

sta_fp_rates.sort(key=lambda x: -x[1])
for si, fp_r, fp_n, total_n in sta_fp_rates[:20]:
    lat = STA_MASTER_POS[si, 0]
    lon = STA_MASTER_POS[si, 1]
    n_train = sta_train_counts.get(si, 0)
    print(f'  {si:>5} {lat:>7.2f} {lon:>8.2f} {n_train:>8} {fp_n:>7} {total_n:>8} {fp_r:>7.3f}')

# ── 地域別 FP率 ──
print('\n' + '=' * 75)
print('地域別 FP率（緯度帯 x 日本海側/太平洋側）')
print('=' * 75)

regions = [
    ('北海道北部',      43.0, 46.0, 141.0, 146.0),
    ('北海道南部',      41.0, 43.0, 139.0, 146.0),
    ('東北太平洋側',    37.0, 41.0, 140.0, 143.0),
    ('東北日本海側',    37.0, 41.0, 138.0, 140.0),
    ('関東',           34.5, 37.0, 139.0, 141.0),
    ('中部太平洋側',    34.0, 37.0, 136.5, 139.0),
    ('中部日本海側',    35.5, 38.0, 135.0, 138.0),
    ('近畿',           33.5, 35.5, 134.0, 137.0),
    ('中国',           33.5, 35.5, 131.0, 134.0),
    ('四国',           32.5, 34.5, 132.0, 135.0),
    ('九州北部',        32.0, 34.0, 129.5, 132.0),
    ('九州南部',        30.0, 32.0, 129.5, 132.0),
    ('沖縄',           24.0, 28.0, 122.0, 130.0),
]

print(f'  {"地域":<16} {"観測点数":>8} {"平均学習件数":>12} {"FP件数":>8} {"FP総数":>10} {"FP率":>7}')
print('-' * 75)
for name, lat_lo, lat_hi, lon_lo, lon_hi in regions:
    stas = []
    fp_s = total_s = 0
    train_sum = 0
    for si in range(N_STATIONS):
        lat, lon = STA_MASTER_POS[si]
        if lat_lo <= lat < lat_hi and lon_lo <= lon < lon_hi:
            stas.append(si)
            fp_s    += sta_fp_count.get(si, 0)
            total_s += sta_fp_total.get(si, 0)
            train_sum += sta_train_counts.get(si, 0)
    if len(stas) > 0 and total_s > 0:
        avg_train = train_sum / len(stas)
        print(f'  {name:<16} {len(stas):>8,} {avg_train:>12.0f} {fp_s:>8,} {total_s:>10,} {fp_s/total_s:>7.4f}')

print('\n分析完了。')
