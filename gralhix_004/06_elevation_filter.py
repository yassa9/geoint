#!/usr/bin/env python3
import argparse
import json
import math
import time
import warnings
from pathlib import Path

import rasterio
from rasterio.errors import RasterioIOError

warnings.filterwarnings("ignore")

ARC_WIDTH_DEG = 100.0
RADII_KM = [2, 5, 8, 11, 14, 17, 20]
BEARING_STEP_DEG = 10
ARC_MIN_ELEV_M, ARC_MAX_ELEV_M = 100.0, 500.0
P0_MAX_ELEV_M = 50.0

TILE_URL = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
            "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
            "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")

HERE = Path(__file__).parent
DEFAULT_IN = HERE / "candidates_ndvi.json"
DEFAULT_OUT = HERE / "candidates_elevation.json"

_tile_cache = {}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default=str(DEFAULT_IN), help="input candidates json")
    p.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT), help="output candidates json")
    return p.parse_args()


def bearing(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def front_bearing(lat0, lon0, lat1, lon1, lat2, lon2):
    b1 = bearing(lat0, lon0, lat1, lon1)
    b2 = bearing(lat0, lon0, lat2, lon2)
    d = ((b1 - b2 + 180) % 360) - 180
    return (b2 + d / 2.0) % 360


def arc_points(lat0, lon0, front, half_width):
    pts = []
    lo, hi = front - half_width, front + half_width
    b = lo
    while b <= hi + 1e-6:
        for r_km in RADII_KM:
            rad = math.radians(b)
            dlat = (r_km / 111.0) * math.cos(rad)
            dlon = (r_km / (111.0 * math.cos(math.radians(lat0)))) * math.sin(rad)
            pts.append((lat0 + dlat, lon0 + dlon))
        b += BEARING_STEP_DEG
    return pts


def tile_name(lat, lon):
    lat_floor = math.floor(lat)
    lon_floor = math.floor(lon)
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return TILE_URL.format(ns=ns, lat=abs(lat_floor), ew=ew, lon=abs(lon_floor))


def open_tile(url):
    if url in _tile_cache:
        return _tile_cache[url]
    try:
        ds = rasterio.open(url)
    except RasterioIOError:
        ds = None
    _tile_cache[url] = ds
    return ds


def main():
    t0 = time.time()
    args = parse_args()
    IN_PATH = Path(args.in_path)
    OUT_PATH = Path(args.out_path)
    candidates = json.loads(IN_PATH.read_text())
    n = len(candidates)
    print(f"loaded {n} candidates", flush=True)

    arc_pts, p0_pt = [], []
    for c in candidates:
        lon0, lat0 = c["P0"]
        lon1, lat1 = c["P1"]
        lon2, lat2 = c["P2"]
        front = front_bearing(lat0, lon0, lat1, lon1, lat2, lon2)
        arc_pts.append(arc_points(lat0, lon0, front, ARC_WIDTH_DEG / 2.0))
        p0_pt.append((lat0, lon0))

    total_points = sum(len(p) for p in arc_pts) + n
    print(f"{total_points} elevation points to sample ({total_points/n:.0f}/candidate)", flush=True)

    tiles_needed = {}
    for ci, pts in enumerate(arc_pts):
        for lat, lon in pts:
            tiles_needed.setdefault(tile_name(lat, lon), []).append((ci, lat, lon, "arc"))
    for ci, (lat, lon) in enumerate(p0_pt):
        tiles_needed.setdefault(tile_name(lat, lon), []).append((ci, lat, lon, "p0"))
    print(f"{len(tiles_needed)} distinct dem tiles needed", flush=True)

    max_arc_elev = [0.0] * n
    p0_elev = [0.0] * n
    missing = 0
    done_tiles = 0
    for url, entries in tiles_needed.items():
        ds = open_tile(url)
        done_tiles += 1
        if ds is None:
            missing += 1
            print(f"  [{done_tiles}/{len(tiles_needed)}] tile unavailable, {missing} missing so far", flush=True)
            continue
        coords = [(lon, lat) for _, lat, lon, _ in entries]
        try:
            samples = list(ds.sample(coords))
        except Exception:
            missing += 1
            print(f"  [{done_tiles}/{len(tiles_needed)}] tile read error, {missing} missing so far", flush=True)
            continue
        for (ci, lat, lon, kind), val in zip(entries, samples):
            e = float(val[0])
            if e <= -1000:
                continue
            if kind == "arc":
                max_arc_elev[ci] = max(max_arc_elev[ci], e)
            else:
                p0_elev[ci] = e
        print(f"  [{done_tiles}/{len(tiles_needed)}] ok, {time.time()-t0:.1f}s elapsed", flush=True)

    for ds in _tile_cache.values():
        if ds is not None:
            ds.close()

    print(f"\nfetched in {time.time()-t0:.1f}s ({missing}/{len(tiles_needed)} tiles missing, "
          f"expected for open-ocean tiles)", flush=True)

    survivors = []
    for ci, c in enumerate(candidates):
        c["p0_elev_m"] = round(p0_elev[ci], 1)
        c["max_elev_in_arc_m"] = round(max_arc_elev[ci], 1)
        if p0_elev[ci] <= P0_MAX_ELEV_M and ARC_MIN_ELEV_M <= max_arc_elev[ci] <= ARC_MAX_ELEV_M:
            survivors.append(c)

    print(f"{len(survivors)}/{n} survive (p0<={P0_MAX_ELEV_M}m, "
          f"arc {ARC_MIN_ELEV_M}-{ARC_MAX_ELEV_M}m)", flush=True)

    OUT_PATH.write_text(json.dumps(survivors, indent=2))
    print(f"saved -> {OUT_PATH}")
    print(f"done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
