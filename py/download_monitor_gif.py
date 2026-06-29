# Created: 2026-05-21 JST
r"""
強震モニタ / 長周期地震動モニタ 数値データ一括取得スクリプト

GIF画像を1秒ごとに取得し、観測点ピクセル座標から数値を抽出してJSONで保存する。

使用例:
  python "C:\Users\齋藤 十\QuakeView\py\download_monitor_gif.py" kyoshin    "2026-05-01 12:00:00" "2026-05-01 12:01:00"
  python "C:\Users\齋藤 十\QuakeView\py\download_monitor_gif.py" longperiod "2026-05-01 12:00:00" "2026-05-01 12:00:30" -o ./output
  python "C:\Users\齋藤 十\QuakeView\py\download_monitor_gif.py" kyoshin    20260501120000 20260501120100 --delay 0.5

monitor 引数:
  kyoshin    = 強震モニタ     → 観測点ごとの計測震度(float)を保存
  longperiod = 長周期地震動モニタ → 観測点ごとのRGB + 活性度スコアを保存

出力JSON形式 (1秒=1ファイル):
  kyoshin    : {"time": "2025/01/01 12:00:00", "stations": [{"code":..., "shindo": 1.2}, ...]}
  longperiod : {"time": "2025/01/01 12:00:00", "stations": [{"code":..., "r":63,"g":250,"b":54,"activity":12.4}, ...]}

依存ライブラリ:
  pip install requests pillow numpy
"""

import argparse
import csv
import io
import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import requests
from PIL import Image

try:
    from zoneinfo import ZoneInfo
    JST = ZoneInfo("Asia/Tokyo")
except ImportError:
    import pytz
    JST = pytz.timezone("Asia/Tokyo")


# ── 強震モニタ カラーマップ（quakeview_proxy.py と同一定義）────────────────────
_COLORMAP_JSON = '[{"Intensity":-3,"R":0,"G":0,"B":205},{"Intensity":-2.9,"R":0,"G":7,"B":209},{"Intensity":-2.8,"R":0,"G":14,"B":214},{"Intensity":-2.7,"R":0,"G":21,"B":218},{"Intensity":-2.6,"R":0,"G":28,"B":223},{"Intensity":-2.5,"R":0,"G":36,"B":227},{"Intensity":-2.4,"R":0,"G":43,"B":231},{"Intensity":-2.3,"R":0,"G":50,"B":236},{"Intensity":-2.2,"R":0,"G":57,"B":240},{"Intensity":-2.1,"R":0,"G":64,"B":245},{"Intensity":-2,"R":0,"G":72,"B":250},{"Intensity":-1.9,"R":0,"G":85,"B":238},{"Intensity":-1.8,"R":0,"G":99,"B":227},{"Intensity":-1.7,"R":0,"G":112,"B":216},{"Intensity":-1.6,"R":0,"G":126,"B":205},{"Intensity":-1.5,"R":0,"G":140,"B":194},{"Intensity":-1.4,"R":0,"G":153,"B":183},{"Intensity":-1.3,"R":0,"G":167,"B":172},{"Intensity":-1.2,"R":0,"G":180,"B":161},{"Intensity":-1.1,"R":0,"G":194,"B":150},{"Intensity":-1,"R":0,"G":208,"B":139},{"Intensity":-0.9,"R":6,"G":212,"B":130},{"Intensity":-0.8,"R":12,"G":216,"B":121},{"Intensity":-0.7,"R":18,"G":220,"B":113},{"Intensity":-0.6,"R":25,"G":224,"B":104},{"Intensity":-0.5,"R":31,"G":228,"B":96},{"Intensity":-0.4,"R":37,"G":233,"B":88},{"Intensity":-0.3,"R":44,"G":237,"B":79},{"Intensity":-0.2,"R":50,"G":241,"B":71},{"Intensity":-0.1,"R":56,"G":245,"B":62},{"Intensity":0,"R":63,"G":250,"B":54},{"Intensity":0.1,"R":75,"G":250,"B":49},{"Intensity":0.2,"R":88,"G":250,"B":45},{"Intensity":0.3,"R":100,"G":251,"B":41},{"Intensity":0.4,"R":113,"G":251,"B":37},{"Intensity":0.5,"R":125,"G":252,"B":33},{"Intensity":0.6,"R":138,"G":252,"B":28},{"Intensity":0.7,"R":151,"G":253,"B":24},{"Intensity":0.8,"R":163,"G":253,"B":20},{"Intensity":0.9,"R":176,"G":254,"B":16},{"Intensity":1,"R":189,"G":255,"B":12},{"Intensity":1.1,"R":195,"G":254,"B":10},{"Intensity":1.2,"R":202,"G":254,"B":9},{"Intensity":1.3,"R":208,"G":254,"B":8},{"Intensity":1.4,"R":215,"G":254,"B":7},{"Intensity":1.5,"R":222,"G":255,"B":5},{"Intensity":1.6,"R":228,"G":254,"B":4},{"Intensity":1.7,"R":235,"G":255,"B":3},{"Intensity":1.8,"R":241,"G":254,"B":2},{"Intensity":1.9,"R":248,"G":255,"B":1},{"Intensity":2,"R":255,"G":255,"B":0},{"Intensity":2.1,"R":254,"G":251,"B":0},{"Intensity":2.2,"R":254,"G":248,"B":0},{"Intensity":2.3,"R":254,"G":244,"B":0},{"Intensity":2.4,"R":254,"G":241,"B":0},{"Intensity":2.5,"R":255,"G":238,"B":0},{"Intensity":2.6,"R":254,"G":234,"B":0},{"Intensity":2.7,"R":255,"G":231,"B":0},{"Intensity":2.8,"R":254,"G":227,"B":0},{"Intensity":2.9,"R":255,"G":224,"B":0},{"Intensity":3,"R":255,"G":221,"B":0},{"Intensity":3.1,"R":254,"G":213,"B":0},{"Intensity":3.2,"R":254,"G":205,"B":0},{"Intensity":3.3,"R":254,"G":197,"B":0},{"Intensity":3.4,"R":254,"G":190,"B":0},{"Intensity":3.5,"R":255,"G":182,"B":0},{"Intensity":3.6,"R":254,"G":174,"B":0},{"Intensity":3.7,"R":255,"G":167,"B":0},{"Intensity":3.8,"R":254,"G":159,"B":0},{"Intensity":3.9,"R":255,"G":151,"B":0},{"Intensity":4,"R":255,"G":144,"B":0},{"Intensity":4.1,"R":254,"G":136,"B":0},{"Intensity":4.2,"R":254,"G":128,"B":0},{"Intensity":4.3,"R":254,"G":121,"B":0},{"Intensity":4.4,"R":254,"G":113,"B":0},{"Intensity":4.5,"R":255,"G":106,"B":0},{"Intensity":4.6,"R":254,"G":98,"B":0},{"Intensity":4.7,"R":255,"G":90,"B":0},{"Intensity":4.8,"R":254,"G":83,"B":0},{"Intensity":4.9,"R":255,"G":75,"B":0},{"Intensity":5,"R":255,"G":68,"B":0},{"Intensity":5.1,"R":254,"G":61,"B":0},{"Intensity":5.2,"R":253,"G":54,"B":0},{"Intensity":5.3,"R":252,"G":47,"B":0},{"Intensity":5.4,"R":251,"G":40,"B":0},{"Intensity":5.5,"R":250,"G":33,"B":0},{"Intensity":5.6,"R":249,"G":27,"B":0},{"Intensity":5.7,"R":248,"G":20,"B":0},{"Intensity":5.8,"R":247,"G":13,"B":0},{"Intensity":5.9,"R":246,"G":6,"B":0},{"Intensity":6,"R":245,"G":0,"B":0},{"Intensity":6.1,"R":238,"G":0,"B":0},{"Intensity":6.2,"R":230,"G":0,"B":0},{"Intensity":6.3,"R":223,"G":0,"B":0},{"Intensity":6.4,"R":215,"G":0,"B":0},{"Intensity":6.5,"R":208,"G":0,"B":0},{"Intensity":6.6,"R":200,"G":0,"B":0},{"Intensity":6.7,"R":192,"G":0,"B":0},{"Intensity":6.8,"R":185,"G":0,"B":0},{"Intensity":6.9,"R":177,"G":0,"B":0},{"Intensity":7.0,"R":170,"G":0,"B":0}]'

_CM_ENTRIES  = json.loads(_COLORMAP_JSON)
_CM_RGB      = np.array([(e["R"], e["G"], e["B"]) for e in _CM_ENTRIES], dtype=np.int32)
_CM_SI       = np.array([e["Intensity"] for e in _CM_ENTRIES], dtype=np.float32)
_PAL_DIST_THRESHOLD = 800


# ── モニタ定義 ────────────────────────────────────────────────────────────────
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

# stations.csv の既定パス（スクリプトから相対）
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV   = os.path.join(os.path.dirname(_SCRIPT_DIR), "data", "stations.csv")


# ── 強震モニタ用 GIF 解析 ────────────────────────────────────────────────────

def _build_palette_table(img: Image.Image) -> np.ndarray:
    """GIF パレットインデックス → 震度値 のルックアップテーブルを構築する"""
    pal         = img.getpalette()
    transparency = img.info.get("transparency")
    table       = np.full(256, np.nan, dtype=np.float32)
    for i in range(256):
        if i == transparency:
            continue
        rgb    = np.array([[pal[i * 3], pal[i * 3 + 1], pal[i * 3 + 2]]], dtype=np.int32)
        dist_sq = ((_CM_RGB - rgb) ** 2).sum(axis=1)
        idx    = int(dist_sq.argmin())
        if dist_sq[idx] <= _PAL_DIST_THRESHOLD:
            table[i] = _CM_SI[idx]
    return table


def extract_kyoshin(img: Image.Image, stations: list) -> list:
    """GIF → 観測点ごとの震度 float（マッチしない＝背景は None）"""
    table = _build_palette_table(img)
    imap  = table[np.array(img.convert("P"))]
    H, W  = imap.shape
    result = []
    for st in stations:
        px, py = st.get("pixel_x"), st.get("pixel_y")
        if px is None or py is None or not (0 <= py < H and 0 <= px < W):
            shindo = None
        else:
            v      = imap[py, px]
            shindo = None if np.isnan(v) else round(float(v), 1)
        result.append({
            "code":   st["code"],
            "name":   st.get("name", ""),
            "lat":    st.get("lat"),
            "lon":    st.get("lon"),
            "shindo": shindo,
        })
    return result


# ── 長周期地震動モニタ用 GIF 解析 ────────────────────────────────────────────

def _calc_activity(r: int, g: int, b: int) -> float:
    """彩度 × 輝度（lmoni_map.html の calcActivityScore と同等）"""
    if r is None:
        return -1.0
    max_v = max(r, g, b)
    if max_v == 0:
        return 0.0
    saturation = (max_v - min(r, g, b)) / max_v
    luminance  = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return round(saturation * luminance, 2)


def extract_longperiod(img: Image.Image, stations: list) -> list:
    """GIF → 観測点ごとの RGB + 活性度スコア"""
    rgba  = img.convert("RGBA")
    arr   = np.array(rgba)
    H, W  = arr.shape[:2]
    result = []
    for st in stations:
        px, py = st.get("pixel_x"), st.get("pixel_y")
        if px is None or py is None or not (0 <= py < H and 0 <= px < W):
            r = g = b = None
        else:
            pixel  = arr[py, px]
            r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
        result.append({
            "code":     st["code"],
            "name":     st.get("name", ""),
            "lat":      st.get("lat"),
            "lon":      st.get("lon"),
            "r":        r,
            "g":        g,
            "b":        b,
            "activity": _calc_activity(r, g, b),
        })
    return result


# ── 共通ユーティリティ ─────────────────────────────────────────────────────────

def _detect_encoding(path: str) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift-jis"):
        try:
            with open(path, "rb") as f:
                f.read().decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "utf-8"


def load_stations(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"stations.csv が見つかりません: {csv_path}")
    enc      = _detect_encoding(csv_path)
    stations = []
    with open(csv_path, newline="", encoding=enc) as f:
        for row in csv.DictReader(f):
            try:
                stations.append({
                    "code":    row["code"].strip(),
                    "name":    row.get("name", "").strip(),
                    "pixel_x": int(row["pixel_x"]),
                    "pixel_y": int(row["pixel_y"]),
                    "lat":     float(row["lat"])  if row.get("lat")  else None,
                    "lon":     float(row["lon"])  if row.get("lon")  else None,
                })
            except (ValueError, KeyError):
                continue
    return stations


def build_url(monitor_key: str, dt: datetime) -> str:
    cfg = MONITORS[monitor_key]
    return cfg["url_template"].format(
        img_type=cfg["img_type"],
        date=dt.strftime("%Y%m%d"),
        datetime=dt.strftime("%Y%m%d%H%M%S"),
    )


def fetch_gif_bytes(url: str, session: requests.Session) -> bytes | None:
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200 and resp.content:
            return resp.content
        return None
    except requests.RequestException as e:
        print(f"    [NET ERROR] {e}")
        return None


def parse_dt(s: str) -> datetime:
    """YYYY-MM-DD HH:MM:SS / YYYYMMDDHHmmss → JST datetime"""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            naive = datetime.strptime(s, fmt)
            try:
                return naive.replace(tzinfo=JST)
            except Exception:
                return JST.localize(naive)  # type: ignore[attr-defined]
        except ValueError:
            continue
    raise ValueError(
        f"日時フォーマット不正: {s!r}\n"
        "  例: '2025-01-01 12:00:00'  または  '20250101120000'"
    )


# ── メイン処理 ─────────────────────────────────────────────────────────────────

def download_range(
    monitor_key: str,
    start: datetime,
    end: datetime,
    output_dir: str,
    stations: list,
    delay: float = 0.2,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "QuakeView-Data-Downloader/1.0",
        "Referer":    MONITORS[monitor_key]["referer"],
    })

    extract_fn   = extract_kyoshin if monitor_key == "kyoshin" else extract_longperiod
    monitor_name = MONITORS[monitor_key]["name"]
    total        = int((end - start).total_seconds()) + 1

    print(f"\n[{monitor_name}]")
    print(f"  期間: {start.strftime('%Y-%m-%d %H:%M:%S')} 〜 {end.strftime('%Y-%m-%d %H:%M:%S')} JST")
    print(f"  フレーム数: {total} 件 / 観測点数: {len(stations)}")
    print(f"  保存先: {os.path.abspath(output_dir)}")
    print()

    current = start
    saved = skipped = failed = 0

    while current <= end:
        ts       = current.strftime("%Y%m%d%H%M%S")
        filepath = os.path.join(output_dir, ts + ".json")

        if os.path.exists(filepath):
            skipped += 1
            current += timedelta(seconds=1)
            continue

        url  = build_url(monitor_key, current)
        data = fetch_gif_bytes(url, session)

        if data:
            try:
                img      = Image.open(io.BytesIO(data))
                stations_data = extract_fn(img, stations)
                payload  = {
                    "time":     current.strftime("%Y/%m/%d %H:%M:%S"),
                    "stations": stations_data,
                }
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
                saved += 1
                print(f"  ✓ {ts}.json  ({len(stations_data)} 観測点)")
            except Exception as e:
                failed += 1
                print(f"  ✗ {ts}  (解析失敗: {e})")
        else:
            failed += 1
            print(f"  ✗ {ts}  (取得失敗)")

        current += timedelta(seconds=1)
        if delay > 0:
            time.sleep(delay)

    print(f"\n完了: 保存 {saved} 件 / スキップ(既存) {skipped} 件 / 失敗 {failed} 件")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="強震モニタ / 長周期地震動モニタ 数値データ一括取得",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "monitor",
        choices=list(MONITORS.keys()),
        metavar="monitor",
        help="kyoshin=強震モニタ  longperiod=長周期地震動モニタ",
    )
    parser.add_argument("start", help="開始日時 (JST)  例: '2025-01-01 12:00:00'")
    parser.add_argument("end",   help="終了日時 (JST)  例: '2025-01-01 12:01:00'")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="保存先ディレクトリ (省略時: ./data_<monitor>_<開始日時>)",
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=f"stations.csv のパス (省略時: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="リクエスト間隔(秒)  デフォルト: 0.2",
    )
    args = parser.parse_args()

    try:
        start = parse_dt(args.start)
        end   = parse_dt(args.end)
    except ValueError as e:
        parser.error(str(e))

    if start > end:
        parser.error("開始日時が終了日時より後になっています")

    try:
        stations = load_stations(args.csv)
    except FileNotFoundError as e:
        parser.error(str(e))

    print(f"観測点ロード完了: {len(stations)} 件  ({args.csv})")

    output_dir = args.output or f"data_{args.monitor}_{start.strftime('%Y%m%d_%H%M%S')}"
    download_range(args.monitor, start, end, output_dir, stations, delay=args.delay)


if __name__ == "__main__":
    main()
