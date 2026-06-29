#!/usr/bin/env python3
"""
NII GeoShape の PBF ベクタータイルから津波予報区 GeoJSON を生成する。
zoom=5 の全タイルを取得し、AreaTsunami レイヤーのポリゴンを code 別にマージ。
出力: web/data/tsunami_area.geojson
"""
import urllib.request, ssl, json, sys, os, struct, io, gzip

sys.stdout.reconfigure(encoding='utf-8')

try:
    import mapbox_vector_tile as mvt
    HAS_MVT = True
except ImportError:
    HAS_MVT = False

# --- PBF 取得 ---
CTX = ssl.create_default_context()
HDRS = {'User-Agent': 'Mozilla/5.0 QuakeView/1.0'}
TILE_URL = 'https://geoshape.ex.nii.ac.jp/jma/vector/tile/AreaTsunami/{z}/{x}/{y}.pbf'

def fetch_pbf(z, x, y):
    url = TILE_URL.format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers=HDRS)
    try:
        with urllib.request.urlopen(req, timeout=10, context=CTX) as r:
            return r.read()
    except Exception:
        return None

def tile_to_lnglat(z, x, y, ex, ey, extent=4096):
    """タイル座標 + ピクセルを緯度経度に変換"""
    import math
    n = 2.0 ** z
    lon = (x + ex / extent) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + ey / extent) / n)))
    lat = math.degrees(lat_rad)
    return [round(lon, 2), round(lat, 2)]

def decode_geometry(geom, z, x, y, extent=4096):
    """MVT geometry を LngLat 座標のラインに変換"""
    gtype = geom['type']
    coords = geom['coordinates']
    if gtype in ('LineString',):
        return [[tile_to_lnglat(z, x, y, px, py, extent) for px, py in coords]]
    elif gtype in ('MultiLineString', 'Polygon'):
        return [[tile_to_lnglat(z, x, y, px, py, extent) for px, py in line] for line in coords]
    elif gtype == 'MultiPolygon':
        result = []
        for poly in coords:
            for ring in poly:
                result.append([tile_to_lnglat(z, x, y, px, py, extent) for px, py in ring])
        return result
    return []

def main():
    if not HAS_MVT:
        print("ERROR: mapbox_vector_tile が必要です。")
        print("  pip install mapbox-vector-tile")
        sys.exit(1)

    # zoom=7 で日本全域をカバーするタイル範囲
    # 与那国(x=107)～小笠原(x=114), 稚内(y=45)～沖ノ鳥島(y=58)
    z = 7
    features_by_code = {}
    total = 0

    for x in range(106, 118):
        for y in range(44, 66):
            data = fetch_pbf(z, x, y)
            if not data:
                continue
            try:
                decoded = mvt.decode(data, default_options={"y_coord_down": True})
            except Exception:
                continue
            layer = decoded.get('area')
            if not layer:
                continue
            extent = layer.get('extent', 4096)
            for feat in layer.get('features', []):
                props = feat.get('properties', {})
                code = str(props.get('code', ''))
                name = props.get('name', '')
                if not code:
                    continue
                rings = decode_geometry(feat['geometry'], z, x, y, extent)
                if code not in features_by_code:
                    features_by_code[code] = {'name': name, 'rings': []}
                features_by_code[code]['rings'].extend(rings)
                total += 1

    print("Decoded {} features from {} areas".format(total, len(features_by_code)))

    def simplify_line(pts, tolerance=0.01):
        """Douglas-Peucker 簡略化"""
        if len(pts) <= 2:
            return pts
        dmax = 0
        idx = 0
        p1, p2 = pts[0], pts[-1]
        for i in range(1, len(pts) - 1):
            d = abs((p2[1]-p1[1])*pts[i][0] - (p2[0]-p1[0])*pts[i][1] + p2[0]*p1[1] - p2[1]*p1[0])
            denom = ((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2) ** 0.5
            if denom > 0:
                d /= denom
            if d > dmax:
                dmax = d
                idx = i
        if dmax > tolerance:
            left = simplify_line(pts[:idx+1], tolerance)
            right = simplify_line(pts[idx:], tolerance)
            return left[:-1] + right
        return [pts[0], pts[-1]]

    # GeoJSON に変換（重複・ゼロ長線分を除去 + 簡略化）
    geojson_features = []
    for code, info in sorted(features_by_code.items()):
        seen = set()
        deduped = []
        for line in info['rings']:
            if len(line) < 2:
                continue
            # ゼロ長の線分を除去（全点が同一座標）
            unique_pts = set(tuple(p) for p in line)
            if len(unique_pts) < 2:
                continue
            # 短い線分は simplify しない（離島の小さな海岸線が消えないように）
            if len(line) > 10:
                simplified = simplify_line(line, 0.005)
            else:
                simplified = line
            if len(simplified) < 2:
                continue
            key = tuple(tuple(p) for p in simplified)
            if key not in seen:
                seen.add(key)
                deduped.append(simplified)
        if not deduped:
            continue
        geojson_features.append({
            'type': 'Feature',
            'properties': {'code': code, 'name': info['name']},
            'geometry': {
                'type': 'MultiLineString',
                'coordinates': deduped,
            }
        })

    geojson = {'type': 'FeatureCollection', 'features': geojson_features}

    # 出力
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'web', 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'tsunami_area.geojson')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, separators=(',', ':'))
    print("Written: {} ({} bytes, {} features)".format(out_path, os.path.getsize(out_path), len(geojson_features)))

if __name__ == '__main__':
    main()
