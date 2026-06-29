# Created: 2026-06-03 JST
"""
QuakeView コマンドラインツール集

使い方:
  python py/tools.py <command> [args]

コマンド一覧:
  parse-dat            JMA .dat ファイル → Parquet 変換
  stats-check          Parquet 統計チェック
  predict              震度分布予測AI 推論
  download-monitor     強震モニタ / 長周期GIF 一括取得
  analyze-station      観測点 train/val/test 分割診断
  compare-attenuation  距離減衰式の比較
  validate-hypothesis  震源類似地震の震度変動検証
  train                震度分布予測AI ローカル学習
  scrape-hinet         Hi-net JMA震源カタログ取得
  etas                 ETAS 時間モデル解析
"""

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
#  CMD: parse-dat — JMA .dat → Parquet 変換
# =============================================================================

def _parse_f(raw: bytes, scale: int):
    s = raw.decode("ascii", errors="replace").strip()
    if not s:
        return None
    try:
        return float(s) / scale
    except ValueError:
        return None


def _load_station_master(dat_path: Path) -> dict:
    if not dat_path.exists():
        print(f"[警告] {dat_path.name} が見つかりません。obs_lat/obs_lon は NaN になります。")
        return {}
    master = {}
    with open(dat_path, encoding="shift_jis", errors="replace") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 4:
                continue
            try:
                station_num = int(cols[0].strip())
                name = cols[1].strip()
                lat_raw = cols[2].strip()
                lat = int(lat_raw[0:2]) + int(lat_raw[2:4]) / 60.0
                lon_raw = cols[3].strip()
                lon = int(lon_raw[0:3]) + int(lon_raw[3:5]) / 60.0
                master[station_num] = {
                    "station_id": str(station_num),
                    "name": name,
                    "obs_lat": lat,
                    "obs_lon": lon,
                }
            except Exception:
                continue
    print(f"観測点マスタ読み込み完了: {len(master)} 件")
    return master


def _parse_hypocenter(line: bytes, event_id: int):
    import datetime
    if len(line) < 96:
        return None
    try:
        year   = int(line[1:5])
        month  = int(line[5:7])
        day    = int(line[7:9])
        hour   = int(line[9:11])
        minute = int(line[11:13])
        sec    = _parse_f(line[13:17], 100) or 0.0
        try:
            dt = datetime.datetime(year, month, day, hour, minute, int(sec),
                                   int((sec % 1) * 1_000_000))
        except ValueError:
            dt = None
        lat_deg = _parse_f(line[21:24], 1)
        lat_min = _parse_f(line[24:28], 100)
        hypo_lat = (lat_deg or 0) + (lat_min or 0) / 60.0 if lat_deg is not None else None
        lon_deg = _parse_f(line[32:36], 1)
        lon_min = _parse_f(line[36:40], 100)
        hypo_lon = (lon_deg or 0) + (lon_min or 0) / 60.0 if lon_deg is not None else None
        depth_raw = line[44:49]
        if depth_raw[3:5] == b"  ":
            depth = _parse_f(depth_raw[0:3], 1)
        else:
            depth = _parse_f(depth_raw, 100)
        magnitude = _parse_f(line[52:54], 10)
        return {
            "event_id":  f"{event_id:08d}",
            "datetime":  dt,
            "hypo_lat":  hypo_lat,
            "hypo_lon":  hypo_lon,
            "depth":     depth,
            "magnitude": magnitude,
        }
    except Exception:
        return None


def _parse_observation(line: bytes, event_id: int, station_master: dict):
    if len(line) < 22:
        return None
    try:
        station_num = int(line[0:7].decode("ascii"))
        raw = line[20:22].decode("ascii")
        if "//" in raw:
            return None
        s = raw.strip()
        if not s:
            return None
        intensity = int(s) / 10.0
        st = station_master.get(station_num)
        if st:
            return {
                "event_id":   f"{event_id:08d}",
                "station_id": st["station_id"],
                "obs_lat":    st["obs_lat"],
                "obs_lon":    st["obs_lon"],
                "intensity":  intensity,
                "is_censored": False,
            }
        else:
            return {
                "event_id":   f"{event_id:08d}",
                "station_id": str(station_num),
                "obs_lat":    None,
                "obs_lon":    None,
                "intensity":  intensity,
                "is_censored": False,
            }
    except Exception:
        return None


def main_parse_dat(args):
    import pandas as pd
    data_dir = Path(args.dir) if args.dir else PROJECT_ROOT
    code_p_path = data_dir / "code_p.dat"
    if not code_p_path.exists():
        code_p_path = data_dir / "code_p" / "code_p.dat"
    station_master = _load_station_master(code_p_path)

    dat_files = sorted(data_dir.glob("i????.dat"))
    print(f".dat ファイル数: {len(dat_files)}")

    earthquakes = []
    observations = []
    event_id = 0
    skip_lines = 0

    for dat_path in dat_files:
        print(f"  処理中: {dat_path.name}", end="", flush=True)
        eq_count = obs_count = 0
        with open(dat_path, "rb") as f:
            for raw_line in f:
                line = raw_line.rstrip(b"\r\n")
                if not line:
                    continue
                first = line[0:1]
                if first in (b"A", b"B", b"D"):
                    rec = _parse_hypocenter(line, event_id)
                    if rec:
                        earthquakes.append(rec)
                        eq_count += 1
                    event_id += 1
                else:
                    if event_id == 0:
                        skip_lines += 1
                        continue
                    rec = _parse_observation(line, event_id - 1, station_master)
                    if rec:
                        observations.append(rec)
                        obs_count += 1
        print(f"  →  地震 {eq_count:,} 件 / 観測 {obs_count:,} 件")

    if skip_lines:
        print(f"[情報] 震源レコード前の観測レコード {skip_lines} 行をスキップ")

    eq_df = pd.DataFrame(earthquakes).astype({
        "event_id":  "string",
        "hypo_lat":  "float32",
        "hypo_lon":  "float32",
        "depth":     "float32",
        "magnitude": "float32",
    })
    eq_df["datetime"] = pd.to_datetime(eq_df["datetime"], utc=False)

    obs_df = pd.DataFrame(observations).astype({
        "event_id":   "string",
        "station_id": "string",
        "obs_lat":    "float32",
        "obs_lon":    "float32",
        "intensity":  "float32",
        "is_censored": "bool",
    })

    eq_out  = data_dir / "earthquakes.parquet"
    obs_out = data_dir / "observations.parquet"
    eq_df.to_parquet(eq_out,  index=False, engine="pyarrow")
    obs_df.to_parquet(obs_out, index=False, engine="pyarrow")

    print(f"\n── 出力完了 ──")
    print(f"  {eq_out.name}  : {len(eq_df):,} 件")
    print(f"  {obs_out.name} : {len(obs_df):,} 件")


# =============================================================================
#  CMD: stats-check — Parquet 統計チェック
# =============================================================================

def main_stats_check(args):
    import numpy as np
    import pandas as pd

    data_dir = Path(args.dir) if args.dir else PROJECT_ROOT
    obs = pd.read_parquet(data_dir / "observations.parquet")
    eq  = pd.read_parquet(data_dir / "earthquakes.parquet")

    obs_per_eq = obs.groupby("event_id").size()

    print("=== 1地震あたり観測点数 ===")
    print(obs_per_eq.describe().round(1))
    print()
    print("パーセンタイル:")
    for p in [50, 75, 90, 95, 99, 99.9]:
        print(f"  {p:5.1f}%ile : {np.percentile(obs_per_eq, p):.0f} 点")
    print()

    print("=== 計測震度の分布 ===")
    print(obs["intensity"].describe().round(3))
    print()
    bins   = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7.0]
    labels = ["0.5-1.4","1.5-2.4","2.5-3.4","3.5-4.4","4.5-5.4","5.5+"]
    obs["band"] = pd.cut(obs["intensity"], bins=bins, labels=labels, right=False)
    print(obs["band"].value_counts().sort_index())
    print()

    print("=== 震源深さの分布 ===")
    dbins   = [0, 150, 300, 1000]
    dlabels = ["h<150km", "150<=h<300km", "h>=300km"]
    eq["depth_band"] = pd.cut(eq["depth"], bins=dbins, labels=dlabels, right=False)
    print(eq["depth_band"].value_counts().sort_index())
    print()

    print("=== magnitude 分布 ===")
    print(eq["magnitude"].describe().round(2))
    print(f"null件数: {eq['magnitude'].isna().sum()}")
    print()

    print("=== obs_lat/lon 範囲チェック ===")
    print(f"obs_lat: {obs.obs_lat.min():.2f} ~ {obs.obs_lat.max():.2f}  (期待: 20~50)")
    print(f"obs_lon: {obs.obs_lon.min():.2f} ~ {obs.obs_lon.max():.2f}  (期待: 120~160)")
    print()

    print("=== 観測点数が異常に多い地震 Top5 ===")
    print(obs_per_eq.sort_values(ascending=False).head())


# =============================================================================
#  CMD: predict — 震度分布予測AI 推論
# =============================================================================

def main_predict(args):
    import json
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from datetime import datetime, timezone, timedelta

    if args.json:
        sys.stdout = sys.stderr

    DATA_DIR      = PROJECT_ROOT
    CKPT_PATH     = DATA_DIR / 'checkpoints' / 'best_model.pt'
    EQ_PATH       = DATA_DIR / 'earthquakes.parquet'
    OBS_PATH      = DATA_DIR / 'observations.parquet'
    TEMPLATE_PATH = PROJECT_ROOT / 'web' / 'seismic_predict.html'
    DEFAULT_OUT   = PROJECT_ROOT / 'web' / 'seismic_predict_result.html'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def make_mlp(in_dim, hidden, out_dim, n_layers):
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, out_dim))
        return nn.Sequential(*layers)

    class SeismicModel(nn.Module):
        def __init__(self, hidden=256, n_heads=4, src_layers=3,
                     sta_layers=2, n_stations=8000):
            super().__init__()
            self.src_enc   = make_mlp(4, hidden, hidden, src_layers)
            self.sta_enc   = make_mlp(7, hidden, hidden, sta_layers)
            self.sta_embed = nn.Embedding(n_stations, hidden)
            self.attn      = nn.MultiheadAttention(hidden, n_heads,
                                 batch_first=True, dropout=0.1)
            self.norm      = nn.LayerNorm(hidden)
            sa_layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads, dim_feedforward=hidden*2,
                dropout=0.1, batch_first=True, activation='gelu')
            self.self_attn = nn.TransformerEncoder(sa_layer, num_layers=1)
            self.head      = nn.Linear(hidden, 1)
            self.log_temp_pos = nn.Parameter(torch.tensor(0.0))
            self.log_temp_neg = nn.Parameter(torch.tensor(0.0))

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
            residual = self.head(out).squeeze(-1)
            t_pos    = self.log_temp_pos.exp()
            t_neg    = self.log_temp_neg.exp()
            residual = torch.where(residual > 0, residual * t_pos, residual * t_neg)
            return residual + prior

    print('観測点データ読み込み中...')
    eq  = pd.read_parquet(EQ_PATH,
              columns=['event_id', 'hypo_lat', 'hypo_lon', 'depth', 'magnitude'])
    eq  = eq.dropna(subset=['magnitude', 'hypo_lat', 'hypo_lon', 'depth'])
    obs = pd.read_parquet(OBS_PATH)

    obs_merged = obs.merge(
        eq[['event_id', 'hypo_lat', 'hypo_lon', 'depth', 'magnitude']],
        on='event_id', how='inner'
    ).dropna(subset=['obs_lat', 'obs_lon'])

    all_stations   = sorted(obs_merged['station_id'].unique())
    station_to_idx = {sid: i for i, sid in enumerate(all_stations)}
    n_stations     = len(station_to_idx)

    coords = (obs_merged
              .groupby('station_id')[['obs_lat', 'obs_lon']]
              .agg(lambda x: x.mode().iloc[0])
              .reset_index())
    coords['station_idx'] = coords['station_id'].map(station_to_idx)
    coords = coords.sort_values('station_idx').reset_index(drop=True)
    print(f'観測点数: {n_stations:,}')

    model = SeismicModel(hidden=256, n_heads=4, n_stations=n_stations).to(DEVICE)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    print(f'モデル読み込み完了（val_mae={ckpt.get("val_mae", float("nan")):.4f}）')

    if all(v is not None for v in [args.lat, args.lon, args.depth, args.mag]):
        lat, lon, depth, mag = args.lat, args.lon, args.depth, args.mag
    else:
        print('\n── 震源情報を入力してください ──')
        lat   = float(input('  緯度  (例 38.10): '))
        lon   = float(input('  経度  (例 142.86): '))
        depth = float(input('  深さ km (例 24): '))
        mag   = float(input('  マグニチュード (例 9.0): '))

    out_path = Path(args.out) if args.out else DEFAULT_OUT

    print(f'\n予測中... 震源: ({lat:.2f}N, {lon:.2f}E) 深さ{depth:.0f}km M{mag:.1f}')

    n = len(coords)
    with torch.no_grad():
        src     = torch.tensor([[lat, lon, depth, mag]], dtype=torch.float32).to(DEVICE)
        sta_pos = coords[['obs_lat', 'obs_lon']].values.astype('float32')
        dlat_r = np.radians(sta_pos[:, 0] - lat)
        dlon_r = np.radians(sta_pos[:, 1] - lon)
        lat_r  = np.radians(lat)
        a = (np.sin(dlat_r/2)**2
             + np.cos(lat_r)*np.cos(np.radians(sta_pos[:, 0]))*np.sin(dlon_r/2)**2)
        d_epi = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        D = np.sqrt(d_epi**2 + max(depth, 1.0)**2)
        R0 = 0.0028 * 10.0**(0.5*mag)
        # exp29: Tobit(左打ち切り)回帰で再フィットした距離減衰係数。
        # 旧式(0.58M, -log10, -0.002D, +0.211)は減衰がなだらか過ぎ、震度1の縁が
        # 半径2-3倍(面積3-8倍)に広がっていた。alpha=3.02で幾何減衰を急峻化。
        prior = (1.7022*mag + 0.00695*depth
                 - 3.0161*np.log10(D + R0) - 0.00564*D - 1.2175).astype('float32')
        log_dist = np.log10(D + 1.0).astype('float32')
        az_rad = np.arctan2(sta_pos[:, 1] - lon, sta_pos[:, 0] - lat)
        sin_az = np.sin(az_rad).astype('float32')
        cos_az = np.cos(az_rad).astype('float32')
        obs_pos_np = np.column_stack([sta_pos, prior, log_dist, sin_az, cos_az])
        obs_pos = torch.tensor(obs_pos_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        sta_idx = torch.tensor(coords['station_idx'].values,
                               dtype=torch.long).unsqueeze(0).to(DEVICE)
        mask    = torch.zeros(1, n, dtype=torch.bool).to(DEVICE)
        pred_vals = model(src, obs_pos, sta_idx, mask).squeeze(0).cpu().numpy()

    result = coords.copy()
    result['predicted_intensity'] = pred_vals

    pred = result['predicted_intensity'].values
    print('\n── 震度帯別観測点数 ──')
    for lo, hi, label in [(0.5,1.5,'1'),(1.5,2.5,'2'),(2.5,3.5,'3'),
                           (3.5,4.5,'4'),(4.5,5.0,'5弱'),(5.0,5.5,'5強'),
                           (5.5,6.0,'6弱'),(6.0,6.5,'6強'),(6.5,9.9,'7')]:
        n_band = ((pred >= lo) & (pred < hi)).sum()
        if n_band > 0:
            print(f'  震度{label}: {n_band:4d} 観測点')
    print(f'  最大計測震度予測: {pred.max():.2f}')

    jst = datetime.now(tz=timezone(timedelta(hours=9)))
    pred_data = {
        'meta': {
            'lat': round(lat, 3), 'lon': round(lon, 3),
            'depth': depth, 'mag': mag,
            'generated_at': jst.strftime('%Y-%m-%d %H:%M JST'),
            'model_exp': ckpt.get('exp', ''),
            'model_epoch': ckpt.get('epoch', 0),
            'model_val_mae': round(ckpt.get('val_mae', float('nan')), 4),
        },
        'stations': [
            {
                'id':        str(row['station_id']),
                'lat':       round(float(row['obs_lat']), 4),
                'lon':       round(float(row['obs_lon']), 4),
                'intensity': round(max(0.0, float(row['predicted_intensity'])), 3),
            }
            for _, row in result.iterrows()
        ],
    }

    if args.json:
        sys.stdout = sys.__stdout__
        print(json.dumps(pred_data, ensure_ascii=False))
    else:
        template = TEMPLATE_PATH.read_text(encoding='utf-8')
        injected = template.replace(
            'const PRED_DATA = null; /* __PRED_DATA__ */',
            f'const PRED_DATA = {json.dumps(pred_data, ensure_ascii=False)};'
        )
        out_path.write_text(injected, encoding='utf-8')
        print(f'\nHTML を出力しました: {out_path}')


# =============================================================================
#  CMD: download-monitor — 強震モニタ / 長周期GIF 一括取得
# =============================================================================

_COLORMAP_JSON = '[{"Intensity":-3,"R":0,"G":0,"B":205},{"Intensity":-2.9,"R":0,"G":7,"B":209},{"Intensity":-2.8,"R":0,"G":14,"B":214},{"Intensity":-2.7,"R":0,"G":21,"B":218},{"Intensity":-2.6,"R":0,"G":28,"B":223},{"Intensity":-2.5,"R":0,"G":36,"B":227},{"Intensity":-2.4,"R":0,"G":43,"B":231},{"Intensity":-2.3,"R":0,"G":50,"B":236},{"Intensity":-2.2,"R":0,"G":57,"B":240},{"Intensity":-2.1,"R":0,"G":64,"B":245},{"Intensity":-2,"R":0,"G":72,"B":250},{"Intensity":-1.9,"R":0,"G":85,"B":238},{"Intensity":-1.8,"R":0,"G":99,"B":227},{"Intensity":-1.7,"R":0,"G":112,"B":216},{"Intensity":-1.6,"R":0,"G":126,"B":205},{"Intensity":-1.5,"R":0,"G":140,"B":194},{"Intensity":-1.4,"R":0,"G":153,"B":183},{"Intensity":-1.3,"R":0,"G":167,"B":172},{"Intensity":-1.2,"R":0,"G":180,"B":161},{"Intensity":-1.1,"R":0,"G":194,"B":150},{"Intensity":-1,"R":0,"G":208,"B":139},{"Intensity":-0.9,"R":6,"G":212,"B":130},{"Intensity":-0.8,"R":12,"G":216,"B":121},{"Intensity":-0.7,"R":18,"G":220,"B":113},{"Intensity":-0.6,"R":25,"G":224,"B":104},{"Intensity":-0.5,"R":31,"G":228,"B":96},{"Intensity":-0.4,"R":37,"G":233,"B":88},{"Intensity":-0.3,"R":44,"G":237,"B":79},{"Intensity":-0.2,"R":50,"G":241,"B":71},{"Intensity":-0.1,"R":56,"G":245,"B":62},{"Intensity":0,"R":63,"G":250,"B":54},{"Intensity":0.1,"R":75,"G":250,"B":49},{"Intensity":0.2,"R":88,"G":250,"B":45},{"Intensity":0.3,"R":100,"G":251,"B":41},{"Intensity":0.4,"R":113,"G":251,"B":37},{"Intensity":0.5,"R":125,"G":252,"B":33},{"Intensity":0.6,"R":138,"G":252,"B":28},{"Intensity":0.7,"R":151,"G":253,"B":24},{"Intensity":0.8,"R":163,"G":253,"B":20},{"Intensity":0.9,"R":176,"G":254,"B":16},{"Intensity":1,"R":189,"G":255,"B":12},{"Intensity":1.1,"R":195,"G":254,"B":10},{"Intensity":1.2,"R":202,"G":254,"B":9},{"Intensity":1.3,"R":208,"G":254,"B":8},{"Intensity":1.4,"R":215,"G":254,"B":7},{"Intensity":1.5,"R":222,"G":255,"B":5},{"Intensity":1.6,"R":228,"G":254,"B":4},{"Intensity":1.7,"R":235,"G":255,"B":3},{"Intensity":1.8,"R":241,"G":254,"B":2},{"Intensity":1.9,"R":248,"G":255,"B":1},{"Intensity":2,"R":255,"G":255,"B":0},{"Intensity":2.1,"R":254,"G":251,"B":0},{"Intensity":2.2,"R":254,"G":248,"B":0},{"Intensity":2.3,"R":254,"G":244,"B":0},{"Intensity":2.4,"R":254,"G":241,"B":0},{"Intensity":2.5,"R":255,"G":238,"B":0},{"Intensity":2.6,"R":254,"G":234,"B":0},{"Intensity":2.7,"R":255,"G":231,"B":0},{"Intensity":2.8,"R":254,"G":227,"B":0},{"Intensity":2.9,"R":255,"G":224,"B":0},{"Intensity":3,"R":255,"G":221,"B":0},{"Intensity":3.1,"R":254,"G":213,"B":0},{"Intensity":3.2,"R":254,"G":205,"B":0},{"Intensity":3.3,"R":254,"G":197,"B":0},{"Intensity":3.4,"R":254,"G":190,"B":0},{"Intensity":3.5,"R":255,"G":182,"B":0},{"Intensity":3.6,"R":254,"G":174,"B":0},{"Intensity":3.7,"R":255,"G":167,"B":0},{"Intensity":3.8,"R":254,"G":159,"B":0},{"Intensity":3.9,"R":255,"G":151,"B":0},{"Intensity":4,"R":255,"G":144,"B":0},{"Intensity":4.1,"R":254,"G":136,"B":0},{"Intensity":4.2,"R":254,"G":128,"B":0},{"Intensity":4.3,"R":254,"G":121,"B":0},{"Intensity":4.4,"R":254,"G":113,"B":0},{"Intensity":4.5,"R":255,"G":106,"B":0},{"Intensity":4.6,"R":254,"G":98,"B":0},{"Intensity":4.7,"R":255,"G":90,"B":0},{"Intensity":4.8,"R":254,"G":83,"B":0},{"Intensity":4.9,"R":255,"G":75,"B":0},{"Intensity":5,"R":255,"G":68,"B":0},{"Intensity":5.1,"R":254,"G":61,"B":0},{"Intensity":5.2,"R":253,"G":54,"B":0},{"Intensity":5.3,"R":252,"G":47,"B":0},{"Intensity":5.4,"R":251,"G":40,"B":0},{"Intensity":5.5,"R":250,"G":33,"B":0},{"Intensity":5.6,"R":249,"G":27,"B":0},{"Intensity":5.7,"R":248,"G":20,"B":0},{"Intensity":5.8,"R":247,"G":13,"B":0},{"Intensity":5.9,"R":246,"G":6,"B":0},{"Intensity":6,"R":245,"G":0,"B":0},{"Intensity":6.1,"R":238,"G":0,"B":0},{"Intensity":6.2,"R":230,"G":0,"B":0},{"Intensity":6.3,"R":223,"G":0,"B":0},{"Intensity":6.4,"R":215,"G":0,"B":0},{"Intensity":6.5,"R":208,"G":0,"B":0},{"Intensity":6.6,"R":200,"G":0,"B":0},{"Intensity":6.7,"R":192,"G":0,"B":0},{"Intensity":6.8,"R":185,"G":0,"B":0},{"Intensity":6.9,"R":177,"G":0,"B":0},{"Intensity":7.0,"R":170,"G":0,"B":0}]'

MONITORS = {
    "kyoshin": {
        "name": "強震モニタ",
        "img_type": "jma_s",
        "base_url": "https://smi.lmoniexp.bosai.go.jp",
        "url_template": (
            "https://smi.lmoniexp.bosai.go.jp/data/map_img/RealTimeImg"
            "/{img_type}/{date}/{datetime}.{img_type}.gif"
        ),
        "referer": "https://smi.lmoniexp.bosai.go.jp/",
    },
    "longperiod": {
        "name": "長周期地震動モニタ",
        "img_type": "abrspmx_s",
        "base_url": "https://www.lmoni.bosai.go.jp",
        "url_template": (
            "https://www.lmoni.bosai.go.jp/monitor/data/data/map_img/RealTimeImg"
            "/abrspmx_s/{date}/{datetime}.abrspmx_s.gif"
        ),
        "referer": "https://www.lmoni.bosai.go.jp/",
    },
}


def _build_palette_table(img):
    import json as _json
    import numpy as np
    cm_entries = _json.loads(_COLORMAP_JSON)
    cm_rgb = np.array([(e["R"], e["G"], e["B"]) for e in cm_entries], dtype=np.int32)
    cm_si  = np.array([e["Intensity"] for e in cm_entries], dtype=np.float32)
    pal          = img.getpalette()
    transparency = img.info.get("transparency")
    table        = np.full(256, float('nan'), dtype=np.float32)
    for i in range(256):
        if i == transparency:
            continue
        rgb    = np.array([[pal[i*3], pal[i*3+1], pal[i*3+2]]], dtype=np.int32)
        dist_sq = ((cm_rgb - rgb) ** 2).sum(axis=1)
        idx    = int(dist_sq.argmin())
        if dist_sq[idx] <= 800:
            table[i] = cm_si[idx]
    return table


def _extract_kyoshin(img, stations):
    import numpy as np
    table = _build_palette_table(img)
    imap  = table[np.array(img.convert("P"))]
    H, W  = imap.shape
    result = []
    for st in stations:
        px, py = st.get("pixel_x"), st.get("pixel_y")
        if px is None or py is None or not (0 <= py < H and 0 <= px < W):
            shindo = None
        else:
            v = imap[py, px]
            shindo = None if (v != v) else round(float(v), 1)  # nan check
        result.append({"code": st["code"], "name": st.get("name",""),
                        "lat": st.get("lat"), "lon": st.get("lon"), "shindo": shindo})
    return result


def _extract_longperiod(img, stations):
    import numpy as np
    rgba = img.convert("RGBA")
    arr  = np.array(rgba)
    H, W = arr.shape[:2]
    result = []
    for st in stations:
        px, py = st.get("pixel_x"), st.get("pixel_y")
        if px is None or py is None or not (0 <= py < H and 0 <= px < W):
            r = g = b = None
        else:
            pixel = arr[py, px]
            r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
        max_v = max(r, g, b) if r is not None else 0
        activity = 0.0
        if r is not None and max_v > 0:
            sat = (max_v - min(r, g, b)) / max_v
            lum = 0.2126*r + 0.7152*g + 0.0722*b
            activity = round(sat * lum, 2)
        result.append({"code": st["code"], "name": st.get("name",""),
                        "lat": st.get("lat"), "lon": st.get("lon"),
                        "r": r, "g": g, "b": b, "activity": activity})
    return result


def _load_stations_csv(csv_path: str) -> list:
    import csv
    def detect_enc(path):
        for enc in ("utf-8-sig", "utf-8", "cp932", "shift-jis"):
            try:
                open(path, "rb").read().decode(enc); return enc
            except Exception:
                continue
        return "utf-8"
    enc = detect_enc(csv_path)
    stations = []
    with open(csv_path, newline="", encoding=enc) as f:
        for row in csv.DictReader(f):
            try:
                stations.append({
                    "code":    row["code"].strip(),
                    "name":    row.get("name","").strip(),
                    "pixel_x": int(row["pixel_x"]),
                    "pixel_y": int(row["pixel_y"]),
                    "lat":     float(row["lat"])  if row.get("lat")  else None,
                    "lon":     float(row["lon"])  if row.get("lon")  else None,
                })
            except (ValueError, KeyError):
                continue
    return stations


def main_download_monitor(args):
    import io
    import json
    import time as _time
    import requests
    from datetime import datetime, timedelta
    from PIL import Image

    try:
        from zoneinfo import ZoneInfo
        JST = ZoneInfo("Asia/Tokyo")
    except ImportError:
        import pytz
        JST = pytz.timezone("Asia/Tokyo")

    def parse_dt(s):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                naive = datetime.strptime(s, fmt)
                try:
                    return naive.replace(tzinfo=JST)
                except Exception:
                    return JST.localize(naive)
            except ValueError:
                continue
        raise ValueError(f"日時フォーマット不正: {s!r}")

    monitor_key = args.monitor
    start = parse_dt(args.start)
    end   = parse_dt(args.end)
    if start > end:
        print("エラー: 開始日時が終了日時より後です"); return

    csv_path = args.csv or str(PROJECT_ROOT / "data" / "stations.csv")
    if not os.path.exists(csv_path):
        print(f"エラー: {csv_path} が見つかりません"); return

    stations  = _load_stations_csv(csv_path)
    cfg       = MONITORS[monitor_key]
    extract   = _extract_kyoshin if monitor_key == "kyoshin" else _extract_longperiod
    out_dir   = args.output or f"data_{monitor_key}_{start.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "QuakeView-Data-Downloader/1.0",
                             "Referer": cfg["referer"]})

    print(f"[{cfg['name']}] {start} 〜 {end}")
    saved = skipped = failed = 0
    current = start
    while current <= end:
        ts  = current.strftime("%Y%m%d%H%M%S")
        fp  = os.path.join(out_dir, ts + ".json")
        if os.path.exists(fp):
            skipped += 1; current += timedelta(seconds=1); continue
        url = cfg["url_template"].format(
            img_type=cfg["img_type"],
            date=current.strftime("%Y%m%d"),
            datetime=current.strftime("%Y%m%d%H%M%S"),
        )
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code == 200 and resp.content:
                img     = Image.open(io.BytesIO(resp.content))
                payload = {"time": current.strftime("%Y/%m/%d %H:%M:%S"),
                           "stations": extract(img, stations)}
                with open(fp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, separators=(",",":"))
                saved += 1
                print(f"  ✓ {ts}.json")
            else:
                failed += 1; print(f"  ✗ {ts} (取得失敗)")
        except Exception as e:
            failed += 1; print(f"  ✗ {ts} ({e})")
        current += timedelta(seconds=1)
        if args.delay > 0:
            _time.sleep(args.delay)

    print(f"完了: 保存 {saved} / スキップ {skipped} / 失敗 {failed}")


# =============================================================================
#  CMD: analyze-station — 観測点 train/val/test 分割診断
# =============================================================================

def main_analyze_station(args):
    import numpy as np
    import pandas as pd

    DATA_DIR = PROJECT_ROOT
    eq  = pd.read_parquet(DATA_DIR / 'earthquakes.parquet')
    obs = pd.read_parquet(DATA_DIR / 'observations.parquet')
    eq  = eq.dropna(subset=['magnitude', 'hypo_lat', 'hypo_lon', 'depth'])

    rng = np.random.default_rng(seed=42)
    all_ids = eq['event_id'].to_numpy(dtype=str).copy()
    rng.shuffle(all_ids)

    n = len(all_ids)
    train_ids = set(all_ids[:int(n * 0.85)])
    val_ids   = set(all_ids[int(n * 0.85):int(n * 0.925)])
    test_ids  = set(all_ids[int(n * 0.925):])

    print(f'train: {len(train_ids):,} / val: {len(val_ids):,} / test: {len(test_ids):,}')

    obs_merged = obs.merge(
        eq[['event_id', 'hypo_lat', 'hypo_lon', 'depth', 'magnitude']],
        on='event_id', how='inner'
    ).dropna(subset=['obs_lat', 'obs_lon'])

    train_obs = obs_merged[obs_merged['event_id'].isin(train_ids)]
    val_obs   = obs_merged[obs_merged['event_id'].isin(val_ids)]
    test_obs  = obs_merged[obs_merged['event_id'].isin(test_ids)]

    train_stations = set(train_obs['station_id'].unique())
    val_stations   = set(val_obs['station_id'].unique())
    test_stations  = set(test_obs['station_id'].unique())

    val_not_train  = val_stations  - train_stations
    test_not_train = test_stations - train_stations

    print(f'\ntrain に存在しない観測点:')
    print(f'  val  だけに出現: {len(val_not_train):,} 点')
    print(f'  test だけに出現: {len(test_not_train):,} 点')

    test_obs_not_train = test_obs[~test_obs['station_id'].isin(train_stations)]
    val_obs_not_train  = val_obs[~val_obs['station_id'].isin(train_stations)]

    test_unseen_ratio = len(test_obs_not_train) / len(test_obs) * 100
    val_unseen_ratio  = len(val_obs_not_train)  / len(val_obs)  * 100
    print(f'\ntest 未学習観測点寄与率: {test_unseen_ratio:.2f}%')
    print(f'val  未学習観測点寄与率: {val_unseen_ratio:.2f}%')


# =============================================================================
#  CMD: compare-attenuation — 距離減衰式の比較
# =============================================================================

def main_compare_attenuation(args):
    import sys as _sys
    import numpy as np
    import pandas as pd
    _sys.stdout.reconfigure(encoding='utf-8')

    DATA_DIR = PROJECT_ROOT
    eq  = pd.read_parquet(DATA_DIR / 'earthquakes.parquet')
    obs = pd.read_parquet(DATA_DIR / 'observations.parquet')

    df = obs.merge(
        eq[['event_id', 'hypo_lat', 'hypo_lon', 'depth', 'magnitude']],
        on='event_id', how='inner'
    ).dropna(subset=['obs_lat', 'obs_lon', 'intensity', 'depth', 'magnitude'])
    print(f'観測レコード数: {len(df):,}')

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2
             + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2)**2)
        return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    d_epi = haversine_km(df['hypo_lat'].values, df['hypo_lon'].values,
                          df['obs_lat'].values,  df['obs_lon'].values)
    depth = df['depth'].values.clip(1, None)
    D     = np.sqrt(d_epi**2 + depth**2)
    M     = df['magnitude'].values
    h     = depth
    I_obs = df['intensity'].values

    def R0(M): return 0.0028 * 10**(0.5*M)

    shape_A = 1.72*M - 1.58*np.log10(np.clip(D,1e-3,None)) - 0.002*D
    shape_B = 1.72*M - 1.58*np.log10(D + R0(M)) - 0.002*D
    shape_C = 1.72*(0.58*M + 0.0038*h - np.log10(D + R0(M)) - 0.002*D)
    Mw      = 0.92*M + 0.17
    shape_D = 1.72*(0.58*Mw + 0.0038*h - np.log10(D + R0(Mw)) - 0.002*D)

    def fit_offset(shape, target): return np.median(target - shape)

    c_A = fit_offset(shape_A, I_obs); I_A = np.clip(shape_A + c_A, -2, 8)
    c_B = fit_offset(shape_B, I_obs); I_B = np.clip(shape_B + c_B, -2, 8)
    c_C = fit_offset(shape_C, I_obs); I_C = np.clip(shape_C + c_C, -2, 8)
    c_D = fit_offset(shape_D, I_obs); I_D = np.clip(shape_D + c_D, -2, 8)

    print(f'フィット定数: A={c_A:.3f}  B={c_B:.3f}  C={c_C:.3f}  D={c_D:.3f}')

    def stats(pred, obs):
        err = pred - obs
        return np.abs(err).mean(), np.sqrt((err**2).mean()), err.mean(), err.std()

    print('\n全体統計（定数フィット後）')
    print(f'  {"式":<34}  {"MAE":>6}  {"RMSE":>6}  {"bias":>7}  {"std":>6}')
    for name, I_pred in [
        ('A: 簡略式（飽和なし）', I_A),
        ('B: 簡略式（飽和あり）', I_B),
        ('C: Si-Midorikawa型',   I_C),
        ('D: Si-Midorikawa正式', I_D),
    ]:
        mae, rmse, bias, std = stats(I_pred, I_obs)
        print(f'  {name:<34}  {mae:6.3f}  {rmse:6.3f}  {bias:+7.3f}  {std:6.3f}')
    print('処理完了')


# =============================================================================
#  CMD: validate-hypothesis — 震源類似地震の震度変動検証
# =============================================================================

def main_validate_hypothesis(args):
    import sys as _sys
    import math
    from itertools import combinations
    import numpy as np
    import pandas as pd
    _sys.stdout.reconfigure(encoding='utf-8')

    DATA_DIR  = PROJECT_ROOT
    MODEL_MAE = 0.3348
    DLAT=0.10; DLON=0.10; DDEPTH=5.0; DMAG=0.2; MIN_COMMON=3

    eq  = pd.read_parquet(DATA_DIR / 'earthquakes.parquet').dropna(
            subset=['hypo_lat','hypo_lon','depth','magnitude'])
    obs = pd.read_parquet(DATA_DIR / 'observations.parquet').dropna(
            subset=['intensity'])
    obs = obs[obs['intensity'] >= 0.5]

    print(f'地震数: {len(eq):,}  観測レコード数: {len(obs):,}')

    obs_by_event = obs.groupby('event_id').apply(
        lambda g: dict(zip(g['station_id'], g['intensity']))
    ).to_dict()

    sta_pos = (obs[['station_id','obs_lat','obs_lon']]
               .drop_duplicates('station_id').set_index('station_id'))
    eq_pos  = eq.set_index('event_id')[['hypo_lat','hypo_lon']]

    eq2 = eq.copy()
    eq2['g_lat']   = (eq2['hypo_lat']  / DLAT  ).round().astype(int)
    eq2['g_lon']   = (eq2['hypo_lon']  / DLON  ).round().astype(int)
    eq2['g_depth'] = (eq2['depth']     / DDEPTH).round().astype(int)
    eq2['g_mag']   = (eq2['magnitude'] / DMAG  ).round().astype(int)
    key_col = (eq2['g_lat'].astype(str) + '_' + eq2['g_lon'].astype(str) + '_' +
               eq2['g_depth'].astype(str) + '_' + eq2['g_mag'].astype(str))
    clusters = eq2.groupby(key_col)['event_id'].apply(list).to_dict()

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2)**2
             + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))
             *math.sin(dlon/2)**2)
        return R * 2 * math.asin(min(1.0, a**0.5))

    DIST_BINS = [(0,50,'d<50km'),(50,100,'50-100km'),(100,150,'100-150km'),
                 (150,200,'150-200km'),(200,300,'200-300km'),(300,999,'d>=300km')]

    all_diffs  = []
    dist_diffs = {lb: [] for _, _, lb in DIST_BINS}
    n_pairs    = 0

    for eids in clusters.values():
        valid = [e for e in eids if e in obs_by_event and e in eq_pos.index]
        if len(valid) < 2: continue
        for e1, e2 in combinations(valid, 2):
            d1, d2 = obs_by_event[e1], obs_by_event[e2]
            common = set(d1.keys()) & set(d2.keys())
            if len(common) < MIN_COMMON: continue
            n_pairs += 1
            hypo = eq_pos.loc[e1]
            for sid in common:
                diff = abs(d1[sid] - d2[sid])
                all_diffs.append(diff)
                if sid in sta_pos.index:
                    sp   = sta_pos.loc[sid]
                    dist = haversine_km(hypo['hypo_lat'], hypo['hypo_lon'],
                                        sp['obs_lat'],    sp['obs_lon'])
                    for lo, hi, lb in DIST_BINS:
                        if lo <= dist < hi:
                            dist_diffs[lb].append(diff); break

    print(f'有効ペア数: {n_pairs:,}  サンプル数: {len(all_diffs):,}')

    def row(label, arr):
        if not arr:
            print(f'  {label:<22}: データなし'); return
        a   = np.array(arr)
        mae = a.mean(); diff = mae - MODEL_MAE
        flag = '<<' if diff < -MODEL_MAE*0.15 else ('>>' if diff > MODEL_MAE*0.15 else '~=')
        print(f'  {label:<22}: MAE={mae:.3f} ({diff:+.3f}) {flag}  med={np.median(a):.3f}  n={len(a):,}')

    print(f'\n全体 (モデルMAE={MODEL_MAE})')
    row('全距離帯', all_diffs)
    print('\n距離帯別')
    for _, _, lb in DIST_BINS: row(lb, dist_diffs[lb])
    print('処理完了')


# =============================================================================
#  CMD: train — 震度分布予測AI ローカル学習
# =============================================================================

def main_train(args):
    import time
    import math
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    EXP_NAME    = args.exp or 'exp29'
    DATA_DIR    = PROJECT_ROOT
    CKPT_DIR    = DATA_DIR / 'checkpoints' / EXP_NAME
    GLOBAL_BEST = DATA_DIR / 'checkpoints' / 'best_model.pt'
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    EQ_PATH   = DATA_DIR / 'earthquakes.parquet'
    OBS_PATH  = DATA_DIR / 'observations.parquet'
    # exp29(修正): 変更点をpriorのみに絞る統制実験。exp22からwarm-startし
    # 価値あるembedding(盆地増幅)を保持。headバイアスが新priorの定数差を吸収する。
    PRETRAINED = DATA_DIR / 'checkpoints' / 'exp22' / 'best_model.pt'
    # MAE_CEIL=0.0: 純val_mae早期停止。複合基準(fp最小化)は未学習epochにロック
    # オンしてしまう問題があったため、prior効果のクリーン測定中は無効化する。
    MAE_CEIL   = 0.0

    HIDDEN      = 256; N_HEADS = 4; BATCH_SIZE = 32
    EPOCHS      = 150; LR = 1e-4; PATIENCE = 20
    HUBER_DELTA = 1.0; LAMBDA_CENS = 0.3
    JITTER_DEG  = 0.15; N_CENS_FAR = 500; CENS_DIST_MIN_KM = 200.0
    C_OFFSET    = 0.211

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'デバイス: {DEVICE}')
    if DEVICE == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    print('データ読み込み中...')
    eq  = pd.read_parquet(EQ_PATH)
    obs = pd.read_parquet(OBS_PATH)
    eq  = eq.dropna(subset=['magnitude', 'hypo_lat', 'hypo_lon', 'depth'])

    rng = np.random.default_rng(seed=42)
    all_ids = eq['event_id'].to_numpy(dtype=str).copy()
    rng.shuffle(all_ids)
    n = len(all_ids)
    train_ids = set(all_ids[:int(n * 0.85)])
    val_ids   = set(all_ids[int(n * 0.85):int(n * 0.925)])
    test_ids  = set(all_ids[int(n * 0.925):])
    print(f'train: {len(train_ids):,} / val: {len(val_ids):,} / test: {len(test_ids):,}')

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

    def haversine_km_vec(lat_deg, lon_deg):
        lat_r = math.radians(lat_deg); lon_r = math.radians(lon_deg)
        dlat = STA_LAT_RAD - lat_r; dlon = STA_LON_RAD - lon_r
        a = (np.sin(dlat/2)**2
             + math.cos(lat_r)*np.cos(STA_LAT_RAD)*np.sin(dlon/2)**2)
        return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    def attenuation_prior(d_epi_km, depth_km, mag):
        D  = np.sqrt(d_epi_km**2 + max(depth_km, 1.0)**2)
        R0 = 0.0028 * 10.0**(0.5*mag)
        # exp29: Tobit回帰で再フィット（旧式は減衰がなだらか過ぎ震度1が過大）。
        # 加法ベースラインなので定数項はheadバイアスが吸収、効くのは形(alpha,dcoef)。
        return (1.7022*mag + 0.00695*depth_km
                - 3.0161*np.log10(D + R0) - 0.00564*D - 1.2175).astype('float32')

    class SeismicDataset(Dataset):
        def __init__(self, event_ids, obs_df, eq_df, is_train=False):
            eq_sub  = eq_df[eq_df['event_id'].isin(event_ids)].set_index('event_id')
            obs_sub = obs_df[obs_df['event_id'].isin(event_ids)]
            self.is_train = is_train
            self.rng = np.random.default_rng(seed=None if is_train else 0)
            self.samples = []
            for eid, grp in obs_sub.groupby('event_id'):
                if eid not in eq_sub.index: continue
                src = eq_sub.loc[eid]
                self.samples.append({
                    'src': [float(src['hypo_lat']), float(src['hypo_lon']),
                            float(src['depth']), float(src['magnitude'])],
                    'sta_idx_obs':   grp['station_idx'].values.astype('int64'),
                    'intensity_obs': grp['intensity'].values.astype('float32'),
                    'depth':    float(src['depth']),
                    'magnitude': float(src['magnitude']),
                })

        def __len__(self): return len(self.samples)

        def __getitem__(self, idx):
            s   = self.samples[idx]
            src = list(s['src'])
            if self.is_train and JITTER_DEG > 0 and src[2] < 150.0:
                src[0] += self.rng.uniform(-JITTER_DEG, JITTER_DEG)
                src[1] += self.rng.uniform(-JITTER_DEG, JITTER_DEG)

            dists_km    = haversine_km_vec(src[0], src[1])
            non_obs     = np.ones(N_STATIONS, dtype=bool)
            non_obs[s['sta_idx_obs']] = False
            far_mask    = non_obs & (dists_km >= CENS_DIST_MIN_KM)
            far_idx_all = np.where(far_mask)[0]
            if len(far_idx_all) > N_CENS_FAR:
                cens_idx = self.rng.choice(far_idx_all, N_CENS_FAR, replace=False)
            else:
                cens_idx = far_idx_all

            n_obs = len(s['sta_idx_obs']); n_cens = len(cens_idx)
            n_total = n_obs + n_cens
            all_sta = np.empty(n_total, dtype=np.int64)
            all_sta[:n_obs] = s['sta_idx_obs']; all_sta[n_obs:] = cens_idx

            all_pos  = STA_MASTER_POS_F32[all_sta]
            d_epi    = dists_km[all_sta]
            prior    = attenuation_prior(d_epi, src[2], src[3])
            hypo_dist = np.sqrt(d_epi**2 + max(src[2], 1.0)**2)
            log_dist  = np.log10(hypo_dist + 1.0).astype('float32')
            dlat = all_pos[:, 0] - src[0]
            dlon = all_pos[:, 1] - src[1]
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
            tgt_t[i,:N] = tg;     cens_t[i,:N]   = ce; mask_t[i,:N] = False
        return srcs_t, obs_pos_t, sta_idx_t, tgt_t, cens_t, mask_t

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

    print('Dataset 構築中...')
    NW = 0
    train_ds = SeismicDataset(train_ids, obs_merged, eq, is_train=True)
    val_ds   = SeismicDataset(val_ids,   obs_merged, eq, is_train=False)
    test_ds  = SeismicDataset(test_ids,  obs_merged, eq, is_train=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,   collate_fn=collate_fn, num_workers=NW)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,  collate_fn=collate_fn, num_workers=NW)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,  collate_fn=collate_fn, num_workers=NW)

    model = SeismicModel(HIDDEN, N_HEADS, n_stations=N_STATIONS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'パラメータ数: {n_params:,}')
    if PRETRAINED.exists():
        ckpt_pre  = torch.load(PRETRAINED, map_location=DEVICE)
        pre_state = ckpt_pre['model']
        old_w = pre_state.get('sta_enc.0.weight')
        if old_w is not None and old_w.shape[1] < 7:
            new_w = torch.zeros(old_w.shape[0], 7, dtype=old_w.dtype, device=old_w.device)
            new_w[:, :old_w.shape[1]] = old_w
            pre_state['sta_enc.0.weight'] = new_w
            print(f'  sta_enc.0.weight: ({old_w.shape[1]}) -> (7) にゼロ拡張')
        model.load_state_dict(pre_state, strict=False)
        print(f'Warm-start: val_mae={ckpt_pre.get("val_mae", float("nan")):.4f}')
    else:
        print('ランダム初期化で学習します。')

    def intensity_weight(intensity):
        base = torch.relu(intensity - 1.0)
        return 1.0 + 0.5 * base ** 1.5

    LAMBDA_VAR = 0.3   # exp27で0.3が最良(0.5は悪化)。新priorでもまず0.3から。

    def loss_fn(pred, target, is_cens, mask):
        valid = ~mask
        p, t, c = pred[valid], target[valid], is_cens[valid]
        obs_loss = cens_loss = torch.tensor(0.0, device=pred.device)
        if (~c).any():
            po, to = p[~c], t[~c]
            w = intensity_weight(to)
            obs_loss = (nn.functional.huber_loss(po, to, reduction='none',
                                                   delta=HUBER_DELTA) * w).mean()
        if c.any():
            cens_loss = (torch.relu(p[c] - 0.5)**2).mean()

        obs_mask = ~mask & ~is_cens
        obs_count = obs_mask.sum(dim=1)
        large = obs_count >= 3
        var_loss = torch.tensor(0.0, device=pred.device)
        if large.any():
            obs_f = obs_mask[large].float()
            n = obs_count[large].float().unsqueeze(1)
            p_masked = pred[large] * obs_f
            t_masked = target[large] * obs_f
            p_mean = p_masked.sum(dim=1, keepdim=True) / n
            t_mean = t_masked.sum(dim=1, keepdim=True) / n
            p_var = ((p_masked - p_mean * obs_f) ** 2).sum(dim=1) / (n.squeeze(1) - 1)
            t_var = ((t_masked - t_mean * obs_f) ** 2).sum(dim=1) / (n.squeeze(1) - 1)
            p_std = p_var.clamp(min=1e-8).sqrt()
            t_std = t_var.clamp(min=1e-8).sqrt()
            var_loss = ((p_std - t_std) ** 2).mean()

        return obs_loss + LAMBDA_CENS * cens_loss + LAMBDA_VAR * var_loss

    embed_params = list(model.sta_embed.parameters())
    other_params = [p for n, p in model.named_parameters() if 'sta_embed' not in n]
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'weight_decay': 1e-4},
        {'params': embed_params, 'weight_decay': 1e-4},
    ], lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        total_mae = total_n = fp_count = fp_total = op_count = op_total = 0
        for src, obs_pos, sta_idx, tgt, is_cens, mask in loader:
            src=src.to(DEVICE); obs_pos=obs_pos.to(DEVICE); sta_idx=sta_idx.to(DEVICE)
            tgt=tgt.to(DEVICE); is_cens=is_cens.to(DEVICE); mask=mask.to(DEVICE)
            pred = model(src, obs_pos, sta_idx, mask)
            obs_valid  = ~mask & ~is_cens
            cens_valid = ~mask & is_cens
            n = obs_valid.sum().item()
            if n == 0: continue
            total_mae += (pred[obs_valid] - tgt[obs_valid]).abs().mean().item() * n
            total_n   += n
            nc = cens_valid.sum().item()
            if nc > 0:
                fp_count += (pred[cens_valid] >= 0.5).sum().item()
                fp_total += nc
            low_obs = obs_valid & (tgt >= 0.5) & (tgt < 2.5)
            no = low_obs.sum().item()
            if no > 0:
                op_count += (pred[low_obs] > tgt[low_obs] + 0.5).sum().item()
                op_total += no
        return {
            'mae':     total_mae / total_n if total_n > 0 else float('nan'),
            'fp_rate': fp_count / fp_total if fp_total > 0 else float('nan'),
            'op_rate': op_count / op_total if op_total > 0 else float('nan'),
        }

    best_val_mae = float('inf'); best_fp = float('inf'); patience_cnt = 0
    history = {'train_loss': [], 'val_mae': []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time(); total_loss = n_batches = 0
        for src, obs_pos, sta_idx, tgt, is_cens, mask in train_loader:
            src=src.to(DEVICE); obs_pos=obs_pos.to(DEVICE); sta_idx=sta_idx.to(DEVICE)
            tgt=tgt.to(DEVICE); is_cens=is_cens.to(DEVICE); mask=mask.to(DEVICE)
            optimizer.zero_grad()
            pred = model(src, obs_pos, sta_idx, mask)
            loss = loss_fn(pred, tgt, is_cens, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1
        scheduler.step()
        train_loss  = total_loss / n_batches if n_batches > 0 else float('nan')
        val_metrics = evaluate(val_loader)
        val_mae     = val_metrics['mae']
        history['train_loss'].append(train_loss)
        history['val_mae'].append(val_mae)
        fp_s = f'{val_metrics["fp_rate"]:.3f}' if not math.isnan(val_metrics["fp_rate"]) else '---'
        op_s = f'{val_metrics["op_rate"]:.3f}' if not math.isnan(val_metrics["op_rate"]) else '---'
        print(f'Epoch {epoch:3d}/{EPOCHS} | train_loss={train_loss:.4f} | val_mae={val_mae:.4f}'
              f' | fp={fp_s} op={op_s} | {time.time()-t0:.0f}s')

        ckpt = {'epoch': epoch, 'exp': EXP_NAME, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'val_mae': val_mae, 'history': history}
        torch.save(ckpt, CKPT_DIR / f'epoch_{epoch:03d}.pt')

        fp_rate = val_metrics['fp_rate']
        is_better = False
        if not math.isnan(val_mae) and val_mae < MAE_CEIL:
            if not math.isnan(fp_rate) and fp_rate < best_fp:
                is_better = True
                best_fp = fp_rate
                best_val_mae = val_mae
        elif not math.isnan(val_mae) and val_mae < best_val_mae:
            is_better = True
            best_val_mae = val_mae

        if is_better:
            ckpt['fp_rate'] = fp_rate
            torch.save(ckpt, CKPT_DIR / 'best_model.pt')
            g_mae = float('inf'); g_fp = float('inf')
            if GLOBAL_BEST.exists():
                try:
                    g_ckpt = torch.load(GLOBAL_BEST, map_location='cpu')
                    g_mae = g_ckpt.get('val_mae', float('inf'))
                    g_fp  = g_ckpt.get('fp_rate', float('inf'))
                except Exception: pass
            update_global = False
            if val_mae < MAE_CEIL and g_mae < MAE_CEIL:
                update_global = fp_rate < g_fp
            else:
                update_global = val_mae < g_mae
            if update_global:
                torch.save(ckpt, GLOBAL_BEST)
                print(f'  → best model saved (mae={best_val_mae:.4f} fp={best_fp:.4f}) [グローバル更新]')
            else:
                print(f'  → best model saved (mae={best_val_mae:.4f} fp={best_fp:.4f})')
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f'Early stopping at epoch {epoch}'); break

    print(f'学習完了。best val_mae = {best_val_mae:.4f}')

    # ── テストセット評価 ──
    print('\nテストセット評価中...')
    best_ckpt = torch.load(CKPT_DIR / 'best_model.pt', map_location=DEVICE)
    model.load_state_dict(best_ckpt['model'])
    model.eval()
    all_pred, all_tgt, all_depth, all_fp_pred = [], [], [], []
    with torch.no_grad():
        for src, obs_pos, sta_idx, tgt, is_cens, mask in test_loader:
            src=src.to(DEVICE); obs_pos=obs_pos.to(DEVICE); sta_idx=sta_idx.to(DEVICE)
            tgt=tgt.to(DEVICE); is_cens=is_cens.to(DEVICE); mask=mask.to(DEVICE)
            pred = model(src, obs_pos, sta_idx, mask)
            obs_valid  = ~mask & ~is_cens
            cens_valid = ~mask & is_cens
            depth_exp  = src[:, 2].unsqueeze(1).expand_as(tgt)
            all_pred.append(pred[obs_valid].cpu())
            all_tgt.append(tgt[obs_valid].cpu())
            all_depth.append(depth_exp[obs_valid].cpu())
            if cens_valid.any():
                all_fp_pred.append(pred[cens_valid].cpu())
    all_pred  = torch.cat(all_pred).numpy()
    all_tgt   = torch.cat(all_tgt).numpy()
    all_depth = torch.cat(all_depth).numpy()
    all_fp_pred = torch.cat(all_fp_pred).numpy() if all_fp_pred else np.array([])

    print(f'\nTest MAE (全体): {np.abs(all_pred - all_tgt).mean():.4f}')
    print('\n-- 震度帯別 MAE --')
    for lo, hi, label in [(0.5,1.5,'0.5-1.4 (震度1)'),(1.5,2.5,'1.5-2.4 (震度2)'),
                          (2.5,3.5,'2.5-3.4 (震度3)'),(3.5,4.5,'3.5-4.4 (震度4)'),
                          (4.5,9.0,'4.5+    (震度5弱+)')]:
        m = (all_tgt >= lo) & (all_tgt < hi)
        if m.sum() > 0:
            print(f'  {label:24s}: MAE={np.abs(all_pred[m]-all_tgt[m]).mean():.4f}  (n={m.sum():,})')
    print('\n-- 深さ帯別 MAE --')
    for lo, hi, label in [(0,10,'h<10km'),(10,20,'10-20km'),(20,30,'20-30km'),
                          (30,40,'30-40km'),(40,50,'40-50km'),(50,80,'50-80km'),
                          (80,100,'80-100km'),(100,200,'100-200km (中発)'),
                          (200,300,'200-300km (スラブ内)'),(300,9999,'h>=300km (異常震域)')]:
        m = (all_depth >= lo) & (all_depth < hi)
        if m.sum() > 0:
            print(f'  {label:26s}: MAE={np.abs(all_pred[m]-all_tgt[m]).mean():.4f}  (n={m.sum():,})')
    print('\n-- FP評価 --')
    if len(all_fp_pred) > 0:
        print(f'  fp_rate : {(all_fp_pred>=0.5).mean():.4f}  ({(all_fp_pred>=0.5).sum():,}/{len(all_fp_pred):,})')
        for thr in [0.5, 1.0, 1.5, 2.0]:
            print(f'  pred>={thr}: {(all_fp_pred>=thr).mean():.4f}  ({(all_fp_pred>=thr).sum():,})')
    low_obs = (all_tgt >= 0.5) & (all_tgt < 2.5)
    if low_obs.sum() > 0:
        op = (all_pred[low_obs] > all_tgt[low_obs] + 0.5).mean()
        print(f'  over_pred_rate: {op:.4f}  (n={low_obs.sum():,})')

    print('\n-- それっぽさ指標 --')
    # 階級境界: 震度1=[0.5,1.5) … 震度4=[3.5,4.5), 震度5弱+=[4.5,inf)
    CLASS_BINS = [(1,0.5,1.5),(2,1.5,2.5),(3,2.5,3.5),(4,3.5,4.5),(5,4.5,99.0)]
    max_errs = []
    # 階級ごとに「地震単位の面積比 pred/obs」を集める（censored含む全観測点で数える）
    area_ratios = {k: [] for k,_,_ in CLASS_BINS}
    # 全地震合算の総セル数（マクロ集計用）
    sum_pred = {k: 0 for k,_,_ in CLASS_BINS}
    sum_obs  = {k: 0 for k,_,_ in CLASS_BINS}
    with torch.no_grad():
        for src_b, obs_pos_b, sta_idx_b, tgt_b, is_cens_b, mask_b in test_loader:
            src_b=src_b.to(DEVICE); obs_pos_b=obs_pos_b.to(DEVICE)
            sta_idx_b=sta_idx_b.to(DEVICE); tgt_b=tgt_b.to(DEVICE)
            is_cens_b=is_cens_b.to(DEVICE); mask_b=mask_b.to(DEVICE)
            pred_b = model(src_b, obs_pos_b, sta_idx_b, mask_b)
            B = src_b.shape[0]
            for i in range(B):
                v = ~mask_b[i]; obs_m = v & ~is_cens_b[i]; cens_m = v & is_cens_b[i]
                if obs_m.sum() == 0: continue
                p_o = pred_b[i, obs_m].cpu().numpy(); t_o = tgt_b[i, obs_m].cpu().numpy()
                p_c = pred_b[i, cens_m].cpu().numpy() if cens_m.any() else np.array([])
                p_all = np.concatenate([p_o, p_c]) if len(p_c) else p_o
                max_errs.append(abs(p_o.max() - t_o.max()))
                for k, lo, hi in CLASS_BINS:
                    n_pred = int(((p_all >= lo) & (p_all < hi)).sum())
                    n_obs  = int(((t_o   >= lo) & (t_o   < hi)).sum())  # censored=震度0なので寄与なし
                    sum_pred[k] += n_pred; sum_obs[k] += n_obs
                    if n_obs > 0:
                        area_ratios[k].append(n_pred / n_obs)
    max_errs = np.array(max_errs)
    print(f'  最大震度誤差  : MAE={max_errs.mean():.3f}  med={np.median(max_errs):.3f}')
    print(f'  {"階級":<8s} {"面積比(地震毎mean)":>18s} {"med":>6s} {"総pred":>9s} {"総obs":>9s} {"総比":>7s}')
    label_map = {1:'震度1',2:'震度2',3:'震度3',4:'震度4',5:'震度5弱+'}
    for k,_,_ in CLASS_BINS:
        r = np.array(area_ratios[k])
        rm  = r.mean()       if len(r) else float('nan')
        rmd = np.median(r)   if len(r) else float('nan')
        macro = (sum_pred[k]/sum_obs[k]) if sum_obs[k] > 0 else float('nan')
        print(f'  {label_map[k]:<8s} {rm:>18.2f} {rmd:>6.2f} '
              f'{sum_pred[k]:>9,d} {sum_obs[k]:>9,d} {macro:>7.2f}')
    print('  (理想=1.00。<1=狭すぎ/過小、>1=広すぎ/過大)')
    print('テスト評価完了。')


# =============================================================================
#  CMD: scrape-hinet — Hi-net JMA震源カタログ取得
# =============================================================================

import re as _re
import sqlite3 as _sqlite3

_LINE_PATTERN_FULL = _re.compile(
    r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)"
    r"\s+([\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+([\d.]+)"
    r"\s+([\d.]+)(?:\s+([\d.]+))?\s+(-?[\d.]+[A-Za-z]?)(?:\s+(-?[\d.]+[A-Za-z]?))?"
    r"\s{2,}(.+?)\s+([A-Za-z])\s*$", _re.MULTILINE,
)
_LINE_PATTERN_NOMAG = _re.compile(
    r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)"
    r"\s+([\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+([\d.]+)"
    r"\s+([\d.]+)(?:\s+([\d.]+))?\s{2,}(.+?)\s+([A-Za-z])\s*$", _re.MULTILINE,
)
_LINE_PATTERN_NOERR = _re.compile(
    r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)"
    r"\s{5,}(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)"
    r"(?:\s+(-?[\d.]+[A-Za-z]?))?(?:\s+(-?[\d.]+[A-Za-z]?))?"
    r"\s{2,}(.+?)\s+([A-Za-z])\s*$", _re.MULTILINE,
)
_DATA_LINE_RE = _re.compile(r"^\d{4}-\d{2}-\d{2}\s")


def _hinet_parse_line(line: str):
    m = _LINE_PATTERN_FULL.match(line)
    if m:
        return {"origin_time": m.group(1).strip(), "ot_err": m.group(2),
                "latitude": m.group(3), "lat_err": m.group(4),
                "longitude": m.group(5), "lon_err": m.group(6),
                "depth_km": m.group(7), "depth_err": m.group(8) or "",
                "magnitude1": m.group(9), "magnitude2": m.group(10) or "",
                "region": m.group(11).strip(), "flag": m.group(12)}
    m = _LINE_PATTERN_NOMAG.match(line)
    if m:
        return {"origin_time": m.group(1).strip(), "ot_err": m.group(2),
                "latitude": m.group(3), "lat_err": m.group(4),
                "longitude": m.group(5), "lon_err": m.group(6),
                "depth_km": m.group(7), "depth_err": m.group(8) or "",
                "magnitude1": "", "magnitude2": "",
                "region": m.group(9).strip(), "flag": m.group(10)}
    m = _LINE_PATTERN_NOERR.match(line)
    if m:
        return {"origin_time": m.group(1).strip(), "ot_err": "",
                "latitude": m.group(2), "lat_err": "",
                "longitude": m.group(3), "lon_err": "",
                "depth_km": m.group(4), "depth_err": "",
                "magnitude1": m.group(5) or "", "magnitude2": m.group(6) or "",
                "region": m.group(7).strip(), "flag": m.group(8)}
    return None


def main_scrape_hinet(args):
    import time as _time
    import requests
    import pandas as pd
    from bs4 import BeautifulSoup
    from datetime import datetime, timedelta
    from urllib.parse import urlparse, parse_qs

    USERNAME = args.user or "240370fksm"
    PASSWORD = args.password or "smtwtfs6"
    SLEEP_SEC = args.sleep or 2.0
    DEBUG     = args.debug

    BASE_URL = "https://hinetwww11.bosai.go.jp/auth/"
    LIST_URL = "https://hinetwww11.bosai.go.jp/auth/JMA/jmalist.php"
    DB_PATH  = PROJECT_ROOT / "data" / "hypolist.db"

    if args.auto:
        target = datetime.now() - timedelta(days=2)
        start_date = end_date = target.replace(hour=0, minute=0, second=0, microsecond=0)
        print(f"【自動更新モード】 取得対象日: {start_date.date()}")
    else:
        start_date = args.start_date
        end_date   = args.end_date

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; research-script)"})

    print("ログインページを取得中...")
    login_url = BASE_URL + "?LANG=ja"
    session.get(login_url, timeout=30)
    session.post(login_url, data={"auth_un": USERNAME, "auth_pw": PASSWORD}, timeout=30)
    test = session.get(LIST_URL + "?LANG=ja", timeout=30)
    test_html = test.content.decode("euc-jp", errors="replace")
    if 'name="auth_un"' in test_html:
        print("[NG] ログイン失敗"); return
    print("[OK] ログイン成功")

    def fetch_day(date):
        base_payload = {"LANG": "ja", "list_year": date.strftime("%Y"),
                        "list_month": date.strftime("%m"), "list_day": date.strftime("%d"),
                        "list_span": "1"}
        all_records = []; payload = base_payload; page = 1
        while True:
            resp = session.post(LIST_URL, data=payload, timeout=30)
            html = resp.content.decode("euc-jp", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            pre  = soup.find("pre")
            if pre is None: break
            records = [_hinet_parse_line(l) for l in pre.get_text().splitlines()
                       if _DATA_LINE_RE.match(l)]
            records = [r for r in records if r]
            print(f"  [{date.date()}] {len(records)} 件" + (f" (p{page})" if page > 1 else ""))
            all_records.extend(records)
            next_payload = None
            for a in soup.find_all("a", href=True):
                if a.get_text(strip=True) in {"次へ",">>","Next","next"}:
                    qs = parse_qs(urlparse(a["href"]).query)
                    if qs:
                        next_payload = {**payload, **{k: v[0] for k, v in qs.items()}}
                        break
            if not next_payload: break
            _time.sleep(SLEEP_SEC); payload = next_payload; page += 1
        return all_records

    all_records = []
    current = start_date
    while current <= end_date:
        try:
            all_records.extend(fetch_day(current))
        except Exception as e:
            print(f"  [ERROR] {current.date()}: {e}")
        _time.sleep(SLEEP_SEC)
        current += timedelta(days=1)

    if not all_records:
        print("データが取得できませんでした。"); return

    df = pd.DataFrame(all_records)
    for col in ["magnitude1", "magnitude2"]:
        qual = col + "_quality"
        df[qual] = df[col].str.extract(r"([A-Za-z]+)$")
        df[col]  = pd.to_numeric(df[col].str.replace(r"[A-Za-z]","",regex=True), errors="coerce")
    for col in ["ot_err","latitude","lat_err","longitude","lon_err","depth_km","depth_err"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(columns={"magnitude1_quality":"mag1_quality","magnitude2_quality":"mag2_quality"})

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS jmalist (
            origin_time TEXT, ot_err REAL, latitude REAL, lat_err REAL,
            longitude REAL, lon_err REAL, depth_km REAL, depth_err REAL,
            magnitude1 REAL, mag1_quality TEXT, magnitude2 REAL, mag2_quality TEXT,
            region TEXT, flag TEXT, UNIQUE (origin_time, latitude, longitude))""")
    conn.commit()

    inserted = skipped = 0
    for _, row in df.iterrows():
        if str(row["origin_time"]) in ("NaT","nan",""): skipped += 1; continue
        try:
            vals = [None if (isinstance(v, float) and pd.isna(v)) else v for v in [
                row["origin_time"], row["ot_err"], row["latitude"], row["lat_err"],
                row["longitude"], row["lon_err"], row["depth_km"], row["depth_err"],
                row["magnitude1"], row.get("mag1_quality"), row["magnitude2"], row.get("mag2_quality"),
                row["region"], row["flag"]]]
            cur.execute(
                "INSERT OR IGNORE INTO jmalist "
                "(origin_time,ot_err,latitude,lat_err,longitude,lon_err,depth_km,depth_err,"
                "magnitude1,mag1_quality,magnitude2,mag2_quality,region,flag) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                vals)
            inserted += cur.rowcount
        except Exception as e:
            skipped += 1; print(f"  [WARN] {row.get('origin_time')}: {e}")

    conn.commit(); cur.close(); conn.close()
    print(f"完了: {len(df)} 件処理 → 挿入 {inserted} / スキップ {skipped}  (DB: {DB_PATH})")


# =============================================================================
#  CMD: etas — ETAS 時間モデル解析
# =============================================================================

def main_etas(args):
    import sys as _sys
    import json as _json
    import warnings
    import math
    import time as _time
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
    from scipy.stats import kstest
    from datetime import datetime as _dt

    warnings.filterwarnings('ignore')

    try:
        from numba import njit, prange
        NUMBA_OK = True
    except ImportError:
        NUMBA_OK = False
        print("  警告: numba が見つかりません。pip install numba で大幅高速化できます。")
        def njit(*a, **k):
            return (lambda f: f)(a[0]) if a and callable(a[0]) else (lambda f: f)
        def prange(n): return range(n)

    @njit(cache=True, parallel=True)
    def _compute_lambda(times, mags, mu, K, alpha, mc, c, p, t_window):
        n = len(times); lam = np.empty(n)
        for i in prange(n):
            val = mu
            for j in range(i-1, -1, -1):
                dt = times[i] - times[j]
                if dt > t_window: break
                val += K * math.exp(alpha*(mags[j]-mc)) / (dt+c)**p
            lam[i] = val
        return lam

    @njit(cache=True)
    def _compute_integral(times, mags, mc, mu, K, alpha, c, p, t_window):
        T = times[-1]; integral = mu * T
        for j in range(len(times)):
            A = K * math.exp(alpha*(mags[j]-mc))
            T_eff = min(T, times[j]+t_window)
            if abs(p-1.0) < 1e-8:
                integral += A * math.log((T_eff-times[j]+c)/c)
            else:
                integral += A/(p-1.0)*(c**(1.0-p)-(T_eff-times[j]+c)**(1.0-p))
        return integral

    @njit(cache=True, parallel=True)
    def _negloglik(params, times, mags, mc, t_window):
        mu,K,alpha,c,p = params[0],params[1],params[2],params[3],params[4]
        if mu<=0 or K<=0 or alpha<=0 or c<=0 or p<=0.5: return 1e15
        lam = _compute_lambda(times,mags,mu,K,alpha,mc,c,p,t_window)
        s = 0.0
        for i in prange(len(times)):
            s += math.log(lam[i]) if lam[i]>0 else -1e15
        if s <= -1e14: return 1e15
        return -(s - _compute_integral(times,mags,mc,mu,K,alpha,c,p,t_window))

    @njit(cache=True)
    def _em_step(times, mags, mc, mu, K, alpha, c, p, t_window):
        T = times[-1]; lam = _compute_lambda(times,mags,mu,K,alpha,mc,c,p,t_window)
        sum_rho = sum_phi = 0.0
        for i in range(len(times)):
            if lam[i]>0: sum_rho += mu/lam[i]; sum_phi += (lam[i]-mu)/lam[i]
        mu_new = sum_rho/T
        integral_trigger = 0.0
        for j in range(len(times)):
            A = math.exp(alpha*(mags[j]-mc)); T_eff = min(T, times[j]+t_window)
            if abs(p-1.0)<1e-8: integral_trigger += A*math.log((T_eff-times[j]+c)/c)
            else: integral_trigger += A/(p-1.0)*(c**(1-p)-(T_eff-times[j]+c)**(1-p))
        K_new = sum_phi/integral_trigger if integral_trigger>0 else K
        return mu_new, K_new

    @njit(cache=True)
    def _lambda_series(t_grid, times, mags, mc, mu, K, alpha, c, p, t_window):
        lam = np.full(len(t_grid), mu)
        for j in range(len(times)):
            for k in range(len(t_grid)):
                dt = t_grid[k]-times[j]
                if dt<=0 or dt>t_window: continue
                lam[k] += K*math.exp(alpha*(mags[j]-mc))/(dt+c)**p
        return lam

    def load_catalog(filepath, col_dt, col_mag, col_lat=None, col_lon=None):
        df   = pd.read_csv(filepath, encoding='utf-8-sig', on_bad_lines='skip')
        cols = list(df.columns)
        def find(cands):
            for c in cands:
                for dc in cols:
                    if c.lower() in dc.lower(): return dc
            return None
        if not col_dt:  col_dt  = find(['time','date','日時','発生','origin'])
        if not col_mag: col_mag = find(['magnitude','mag','規模','マグ'])
        if not col_lat: col_lat = find(['lat','緯度'])
        if not col_lon: col_lon = find(['long','lon','lng','経度'])
        df['_dt']  = pd.to_datetime(df[col_dt].astype(str).str.replace(r'[年月]','-',regex=True).str.replace(r'日',' ',regex=True), errors='coerce')
        df['_mag'] = pd.to_numeric(df[col_mag], errors='coerce')
        df = df.dropna(subset=['_dt','_mag']).sort_values('_dt').reset_index(drop=True)
        if len(df)==0: print("エラー: 有効データ0件"); _sys.exit(1)
        t0 = df['_dt'].iloc[0]
        df['_t'] = (df['_dt']-t0).dt.total_seconds()/86400.0
        result = {'times': df['_t'].values.astype(np.float64),
                  'mags':  df['_mag'].values.astype(np.float64),
                  't0_str': str(t0), 'n_total': len(df)}
        if col_lat and col_lat in cols and col_lon and col_lon in cols:
            result['lats'] = pd.to_numeric(df[col_lat], errors='coerce').values.astype(np.float64)
            result['lons'] = pd.to_numeric(df[col_lon], errors='coerce').values.astype(np.float64)
        return result

    def estimate_mc(mags, bin_width=0.1):
        bins = np.arange(round(min(mags)-0.05,1), max(mags)+bin_width, bin_width)
        counts, edges = np.histogram(mags, bins=bins)
        return round(float((edges[:-1]+bin_width/2)[np.argmax(counts)]), 1)

    def estimate_b(mags, mc):
        m_use = mags[mags>=mc]
        if len(m_use)<10: return 1.0
        me = np.mean(m_use)-mc
        return float(np.log10(np.e)/me) if me>0 else 1.0

    def auto_twindow(times_mc):
        N=len(times_mc); T=float(times_mc[-1]-times_mc[0])
        if T<=0 or N<2: return 90.0
        tw = min(T, max(30.0, 2000.0/(N/T)))
        print(f"    [auto_twindow] twindow={round(tw,1)}日")
        return round(tw,1)

    def estimate_mle(times, mags, mc, t_window):
        mask=mags>=mc; t_use=times[mask].copy().astype(np.float64); m_use=mags[mask].copy().astype(np.float64)
        def nll(p): return _negloglik(np.array(p,dtype=np.float64),t_use,m_use,mc,t_window)
        candidates = [[0.01,0.1,1.0,0.01,1.1],[0.05,0.5,1.5,0.001,1.2],[0.1,0.3,2.0,0.1,1.05],[0.01,0.05,0.5,0.05,1.3]]
        best_x0 = min(candidates, key=nll)
        t0 = _time.time()
        res = minimize(nll, best_x0, method='Nelder-Mead', options={'maxiter':30000,'xatol':1e-7,'fatol':1e-7,'adaptive':True})
        print(f"    MLE完了: {_time.time()-t0:.1f}秒  logL={-res.fun:.2f}")
        return res.x, -res.fun

    def estimate_em(times, mags, mc, t_window):
        mask=mags>=mc; t_use=times[mask].copy().astype(np.float64); m_use=mags[mask].copy().astype(np.float64)
        N,T = len(t_use), t_use[-1]
        mu,K,alpha,c,p = N/(2*T),0.1,1.0,0.01,1.1
        lam0 = _compute_lambda(t_use,m_use,mu,K,alpha,mc,c,p,t_window)
        lam_mu = float(np.mean(lam0))
        n_hat = max(0.0, min(0.98, 1.0-mu/lam_mu)) if lam_mu>0 else 0.5
        tol = 1e-4
        max_iter = max(30, min(300, int(math.log(tol)/math.log(max(n_hat,0.01))*1.2)+1))
        print(f"    EM (N={N}, twindow={t_window}日, n̂={n_hat:.3f}, max_iter={max_iter})")
        t0 = _time.time(); logL_prev = -np.inf
        for it in range(max_iter):
            mu_new,K_new = _em_step(t_use,m_use,mc,mu,K,alpha,c,p,t_window)
            def neg_cond(acp):
                a,cc,pp = acp
                if a<=0 or cc<=0 or pp<=0.5: return 1e15
                return float(_negloglik(np.array([mu_new,K_new,a,cc,pp],dtype=np.float64),t_use,m_use,mc,t_window))
            res = minimize(neg_cond,[alpha,c,p],method='L-BFGS-B',
                           bounds=[(1e-6,None),(1e-6,None),(0.501,5.0)],
                           options={'maxfun':300,'ftol':1e-9,'gtol':1e-6})
            alpha_new,c_new,p_new = res.x
            params_new = np.array([mu_new,K_new,alpha_new,c_new,p_new])
            logL_new = float(-_negloglik(params_new,t_use,m_use,mc,t_window))
            delta = abs(logL_new-logL_prev)
            if (it+1)%10==0:
                print(f"    iter {it+1:3d}: logL={logL_new:.2f}  Δ={delta:.2e}  [{_time.time()-t0:.0f}s]")
            mu,K,alpha,c,p = mu_new,K_new,alpha_new,c_new,p_new; logL_prev = logL_new
            if delta<tol and it>5: print(f"    収束: iter={it+1}"); break
        else:
            print(f"    警告: max_iter={max_iter} に達しました")
        print(f"    EM完了: {_time.time()-t0:.1f}秒  logL={logL_prev:.2f}")
        return np.array([mu,K,alpha,c,p]), logL_prev

    def compute_ci(params_opt, times, mags, mc, t_window):
        mask=mags>=mc; t_use=times[mask].copy().astype(np.float64); m_use=mags[mask].copy().astype(np.float64)
        def nll(p): return float(_negloglik(np.array(p,dtype=np.float64),t_use,m_use,mc,t_window))
        try:
            eps_vec = np.maximum(np.abs(params_opt)*1e-4, 1e-6)
            H = np.zeros((5,5))
            for i in range(5):
                ei=np.zeros(5); ei[i]=eps_vec[i]
                H[i,i] = (-nll(params_opt+2*ei)+16*nll(params_opt+ei)-30*nll(params_opt)+16*nll(params_opt-ei)-nll(params_opt-2*ei))/(12*eps_vec[i]**2)
            for i in range(5):
                for j in range(i+1,5):
                    ei=np.zeros(5); ei[i]=eps_vec[i]; ej=np.zeros(5); ej[j]=eps_vec[j]
                    H[i,j] = (nll(params_opt+ei+ej)-nll(params_opt+ei-ej)-nll(params_opt-ei+ej)+nll(params_opt-ei-ej))/(4*eps_vec[i]*eps_vec[j])
                    H[j,i] = H[i,j]
            cov = np.linalg.inv(H); se = np.sqrt(np.abs(np.diag(cov)))
        except Exception:
            se = np.full(5, np.nan); cov = None
        def _safe(v): return None if not math.isfinite(v) else float(v)
        names = ['mu','K','alpha','c','p']
        params_dict = {name: {'estimate': float(params_opt[i]), 'se': _safe(se[i]),
                               'ci_lower': _safe(params_opt[i]-1.96*se[i]),
                               'ci_upper': _safe(params_opt[i]+1.96*se[i])}
                       for i, name in enumerate(names)}
        cov_list = [[_safe(cov[i,j]) for j in range(5)] for i in range(5)] if cov is not None else None
        return params_dict, cov_list

    def residual_analysis(times, mags, mc, params_opt, t_window):
        mu,K,alpha,c,p = params_opt
        mask=mags>=mc; t_use=times[mask].copy(); m_use=mags[mask].copy(); n=len(t_use)
        tau = np.zeros(n)
        for i in range(n):
            ti=t_use[i]; tau[i]=mu*ti
            for j in range(i-1,-1,-1):
                dt=ti-t_use[j]
                if dt>t_window: break
                A=K*np.exp(alpha*(m_use[j]-mc))
                tau[i] += A/((p-1)*(c**(1-p)-(dt+c)**(1-p))) if abs(p-1)>1e-8 else A*math.log((dt+c)/c)
        tau_norm = tau/n
        ks_stat, ks_pval = kstest(tau_norm, 'uniform')
        dtau = np.diff(tau); dtau_pos = dtau[dtau>0]
        ks_dtau_stat = ks_dtau_pval = float('nan')
        if len(dtau_pos)>=10:
            ks_dtau_stat, ks_dtau_pval = kstest(dtau_pos,'expon',args=(0,1))
        def safe(v): return None if not math.isfinite(float(v)) else float(v)
        return {'tau': tau.tolist(), 'tau_norm': tau_norm.tolist(), 'dtau': dtau.tolist(),
                'ks_stat': safe(ks_stat), 'ks_pvalue': safe(ks_pval),
                'ks_dtau_stat': safe(ks_dtau_stat), 'ks_dtau_pvalue': safe(ks_dtau_pval),
                'n_events': n}

    def anomaly_analysis(times, mags, mc, params_opt, t_window, lookback=20, thresh=0.001):
        mu,K,alpha,c,p = params_opt
        mask=mags>=mc; t_mc=times[mask].astype(np.float64); m_mc=mags[mask].astype(np.float64); N=len(t_mc)
        def G(s):
            if s<=0: return 0.0
            return math.log((s+c)/c) if abs(p-1)<1e-8 else (c**(1-p)-(s+c)**(1-p))/(p-1)
        def poisson_cdf(k, lam):
            if lam<=0: return 1.0
            if k<0: return 0.0
            log_term=-lam; total=math.exp(log_term)
            for n in range(1,k+1):
                log_term+=math.log(lam)-math.log(n); total+=math.exp(log_term)
                if total>=1.0: return 1.0
            return min(1.0,total)
        A = K*np.exp(alpha*(m_mc-mc)); scores=np.ones(N)
        for i in range(1,N):
            j_start=max(0,i-lookback); min_p=1.0
            for j in range(j_start,i):
                dt_ij=t_mc[i]-t_mc[j]
                if dt_ij<=0: continue
                lam=mu*dt_ij
                for k in range(i):
                    dt_ki=t_mc[i]-t_mc[k]
                    if dt_ki>t_window: continue
                    dt_kj=t_mc[j]-t_mc[k]
                    lam+=A[k]*(G(dt_ki)-(G(dt_kj) if dt_kj>0 else 0.0))
                lam=max(1e-12,lam); prob=1.0-poisson_cdf(i-j,lam)
                if prob<min_p: min_p=prob
            scores[i]=min_p
        log_scores=-np.log10(np.maximum(scores,1e-10))
        def _safe(v): return None if not math.isfinite(float(v)) else float(v)
        return {'times_mc': t_mc.tolist(), 'mags_mc': m_mc.tolist(),
                'scores': [_safe(s) for s in scores], 'log_scores': [_safe(s) for s in log_scores],
                'anomalous_idx': np.where(scores<thresh)[0].tolist(),
                'n_anomalous': int((scores<thresh).sum()),
                'thresh': thresh, 'lookback': lookback, 'Mc': mc}

    if args.columns:
        df = pd.read_csv(args.csv, encoding='utf-8-sig', nrows=3, on_bad_lines='skip')
        print(f"\nCSV列一覧: {args.csv}")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:>3}  {col:<30}  {str(df[col].iloc[0]) if len(df)>0 else ''}")
        return

    print(f"[1/7] CSV読み込み: {args.csv}")
    data  = load_catalog(args.csv, args.datetime, args.mag, args.lat, args.lon)
    times = data['times']; mags = data['mags']
    if args.tstart > 0:
        idx=times>=args.tstart; times=times[idx]; mags=mags[idx]
    print(f"    → {len(times)} イベント, 期間: {times[-1]:.2f} 日")

    mc = args.mc if args.mc is not None else estimate_mc(mags)
    n_above = int(np.sum(mags>=mc)); b_value = estimate_b(mags, mc)
    print(f"[2/7] Mc={mc:.1f} ({n_above}件)  b値={b_value:.3f}")

    print("[3/7] Numba JITコンパイル中...")
    _d_t = np.array([0.0,1.0,2.0],dtype=np.float64); _d_m=np.array([2.0,2.5,3.0],dtype=np.float64)
    _negloglik(np.array([0.01,0.1,1.0,0.01,1.1],dtype=np.float64),_d_t,_d_m,2.0,365.0)
    _em_step(_d_t,_d_m,2.0,0.01,0.1,1.0,0.01,1.1,365.0)
    print("    完了")

    t_window = args.twindow if args.twindow else auto_twindow(times[mags>=mc])
    method   = args.method or ('mle' if n_above<5000 else 'em')
    print(f"[4/7] パラメータ推定 ({method.upper()}, twindow={t_window:.1f}日)...")
    if method=='mle': params_opt,logL = estimate_mle(times,mags,mc,t_window)
    else:             params_opt,logL = estimate_em(times,mags,mc,t_window)

    print("    信頼区間計算中...")
    params_dict, cov_matrix = compute_ci(params_opt,times,mags,mc,t_window)
    for name,v in params_dict.items():
        lo,hi=v['ci_lower'],v['ci_upper']
        ci = f"[{lo:.4f},{hi:.4f}]" if lo is not None else "(計算不可)"
        print(f"      {name:5s}={v['estimate']:.5f}  95%CI:{ci}")

    print("[5/7] λ(t) 計算中...")
    T = times[-1]; t_grid = np.linspace(0,T,500).astype(np.float64)
    mask_mc = mags>=mc; t_use=times[mask_mc].astype(np.float64); m_use=mags[mask_mc].astype(np.float64)
    mu,K,alpha,c,p = params_opt
    lam_vals = _lambda_series(t_grid,t_use,m_use,mc,mu,K,alpha,c,p,t_window)

    print("[6/7] 残差解析...")
    residuals = residual_analysis(times,mags,mc,params_opt,t_window)
    v1 = "適合良好" if (residuals['ks_pvalue'] or 0)>0.05 else "適合要確認"
    print(f"    KS: D={residuals['ks_stat']:.4f}, p={residuals['ks_pvalue']:.4f} → {v1}")

    n_ev = int(np.sum(mags>=mc)); T_val=times[-1]
    mu_hat = n_ev/T_val; logL_pois = n_ev*math.log(mu_hat)-mu_hat*T_val
    aic = {'etas': {'logL':logL,'n_params':5,'AIC':-2*logL+10},
           'poisson': {'logL':logL_pois,'n_params':1,'AIC':-2*logL_pois+2},
           'delta_aic': (-2*logL_pois+2)-(-2*logL+10), 'n_events': n_ev}
    print(f"    AIC: ETAS={aic['etas']['AIC']:.1f}, Poisson={aic['poisson']['AIC']:.1f}")

    print("[7/7] 異常性解析...")
    anomaly = anomaly_analysis(times,mags,mc,params_opt,t_window)
    print(f"    異常検出: {anomaly['n_anomalous']} 件")

    output = {
        'meta': {'csv': os.path.basename(args.csv), 't0': data['t0_str'],
                 'n_total': data['n_total'], 'n_used': n_above,
                 'Mc': mc, 'b_value': b_value, 'T': float(times[-1]),
                 'method': method.upper(), 'twindow': t_window,
                 'generated': _dt.now().isoformat()},
        'params': params_dict, 'logL': logL, 'aic': aic,
        'lambda': {'t': t_grid.tolist(), 'vals': lam_vals.tolist()},
        'event_times': times[mags>=mc].tolist(), 'event_mags': mags[mags>=mc].tolist(),
        'residuals': residuals, 'anomaly': anomaly, 'spatial': None
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        _json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 完了: {args.output}")


# =============================================================================
#  Main Dispatcher
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='QuakeView コマンドラインツール集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='cmd', metavar='<command>')

    # parse-dat
    p1 = sub.add_parser('parse-dat', help='JMA .dat → Parquet 変換')
    p1.add_argument('--dir', default=None, help='データディレクトリ (省略時: プロジェクトルート)')

    # stats-check
    p2 = sub.add_parser('stats-check', help='Parquet 統計チェック')
    p2.add_argument('--dir', default=None, help='データディレクトリ (省略時: プロジェクトルート)')

    # predict
    p3 = sub.add_parser('predict', help='震度分布予測AI 推論')
    p3.add_argument('--lat',   type=float, help='震源緯度')
    p3.add_argument('--lon',   type=float, help='震源経度')
    p3.add_argument('--depth', type=float, help='震源深さ km')
    p3.add_argument('--mag',   type=float, help='マグニチュード')
    p3.add_argument('--out',   type=str,   default=None, help='出力HTMLパス')
    p3.add_argument('--json',  action='store_true', help='JSON を stdout に出力')

    # download-monitor
    p4 = sub.add_parser('download-monitor', help='強震モニタ / 長周期GIF 一括取得')
    p4.add_argument('monitor', choices=list(MONITORS.keys()), help='kyoshin / longperiod')
    p4.add_argument('start',   help='開始日時 JST (例: "2025-01-01 12:00:00")')
    p4.add_argument('end',     help='終了日時 JST')
    p4.add_argument('-o', '--output', default=None, help='保存先ディレクトリ')
    p4.add_argument('--csv',   default=None, help='stations.csv のパス')
    p4.add_argument('--delay', type=float, default=0.2, help='リクエスト間隔(秒)')

    # analyze-station
    p5 = sub.add_parser('analyze-station', help='観測点 train/val/test 分割診断')

    # compare-attenuation
    p6 = sub.add_parser('compare-attenuation', help='距離減衰式の比較')

    # validate-hypothesis
    p7 = sub.add_parser('validate-hypothesis', help='震源類似地震の震度変動検証')

    # train
    p8 = sub.add_parser('train', help='震度分布予測AI ローカル学習')
    p8.add_argument('--exp', default=None, help='実験名 (デフォルト: exp21)')

    # scrape-hinet
    p9 = sub.add_parser('scrape-hinet', help='Hi-net JMA震源カタログ取得')
    p9.add_argument('--auto',      action='store_true', help='2日前のデータを自動取得')
    p9.add_argument('--user',      default=None, help='Hi-net ユーザー名')
    p9.add_argument('--password',  default=None, help='Hi-net パスワード')
    p9.add_argument('--start-date', dest='start_date', type=lambda s: __import__('datetime').datetime.strptime(s,'%Y-%m-%d'), default=None)
    p9.add_argument('--end-date',   dest='end_date',   type=lambda s: __import__('datetime').datetime.strptime(s,'%Y-%m-%d'), default=None)
    p9.add_argument('--sleep',     type=float, default=2.0)
    p9.add_argument('--debug',     action='store_true')

    # etas
    p10 = sub.add_parser('etas', help='ETAS 時間モデル解析')
    p10.add_argument('csv')
    p10.add_argument('--columns',  action='store_true')
    p10.add_argument('--datetime', default=None, metavar='列名')
    p10.add_argument('--mag',      default=None, metavar='列名')
    p10.add_argument('--lat',      default=None, metavar='列名')
    p10.add_argument('--lon',      default=None, metavar='列名')
    p10.add_argument('--mc',       type=float, default=None)
    p10.add_argument('--tstart',   type=float, default=0.0)
    p10.add_argument('--twindow',  type=float, default=None)
    p10.add_argument('--method',   choices=['mle','em'], default=None)
    p10.add_argument('--output',   default='etas_output.json')

    args = parser.parse_args()

    dispatch = {
        'parse-dat':           main_parse_dat,
        'stats-check':         main_stats_check,
        'predict':             main_predict,
        'download-monitor':    main_download_monitor,
        'analyze-station':     main_analyze_station,
        'compare-attenuation': main_compare_attenuation,
        'validate-hypothesis': main_validate_hypothesis,
        'train':               main_train,
        'scrape-hinet':        main_scrape_hinet,
        'etas':                main_etas,
    }

    if args.cmd in dispatch:
        dispatch[args.cmd](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
