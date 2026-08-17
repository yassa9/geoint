#!/usr/bin/env python3
import argparse
import itertools
import json
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon

warnings.filterwarnings("ignore")

LAT_MIN, LAT_MAX = -30.0, 30.0
CLUSTER_RADIUS_KM = 20.0
LOCAL_DENSITY_RADIUS_KM = 5.0
LOCAL_DENSITY_CAP = 10
MAX_PER_CLUSTER = 60
P0_DIAM_MIN_M, P0_DIAM_MAX_M = 100.0, 1000.0
SEP_MIN, SEP_MAX = 5.0, 30.0
MIN_SIDE_KM, MAX_SIDE_KM = 0.1, 20.0
SELF_EXCLUDE_KM = 1.0

HERE = Path(__file__).parent
SHP_PATH = str(HERE / "data" / "land_polygons.shp")
FP_GLOB = "triangle_*.json"
DEFAULT_OUT = HERE / "candidates_search.json"

KERNEL_DIR = HERE / "gpu_kernel"
KERNEL_SRC = KERNEL_DIR / "match_kernel.cu"
KERNEL_BIN = KERNEL_DIR / "match_kernel"


def latest_fingerprint():
    files = sorted(HERE.glob(FP_GLOB))
    return files[-1] if files else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fingerprint", default=None,
                    help="triangle fingerprint json, defaults to newest triangle_*.json next to this script")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output candidates json")
    return p.parse_args()


def stratified_sample(idx_arr, area_arr, cap):
    if len(idx_arr) <= cap:
        return idx_arr
    order = np.argsort(area_arr[idx_arr])
    n_small = cap // 3
    n_large = cap // 3
    n_mid = cap - n_small - n_large
    mid_start = max(0, (len(idx_arr) - n_large - n_mid) // 2)
    keep = np.unique(np.concatenate([
        order[:n_small], order[-n_large:], order[mid_start:mid_start + n_mid],
    ]))
    return idx_arr[keep]


def gen_cluster_triples(idx_arr):
    if len(idx_arr) < 3:
        return None
    local = np.array(list(itertools.combinations(range(len(idx_arr)), 3)), dtype=np.int64)
    return idx_arr[local]


def point_diameters_m(gdf, sindex, lon, lat):
    out = np.zeros(len(lon))
    for i in range(len(lon)):
        pt = Point(lon[i], lat[i])
        bounds = (lon[i] - 0.02, lat[i] - 0.02, lon[i] + 0.02, lat[i] + 0.02)
        idx = list(sindex.intersection(bounds))
        if not idx:
            continue
        nearby = gdf.iloc[idx]
        row = gdf.geometry.loc[nearby.geometry.distance(pt).idxmin()]
        mrr = row.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        s1 = np.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]) * 111320.0
        s2 = np.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1]) * 111320.0
        out[i] = max(s1, s2)
    return out


def rectangle_empty(lon0, lat0, lon1, lat1, lon2, lat2, land_gdf, land_sindex):
    m_lat = 111320.0
    m_lon = 111320.0 * np.cos(np.radians(lat0))
    x1, y1 = (lon1 - lon0) * m_lon, (lat1 - lat0) * m_lat
    x2, y2 = (lon2 - lon0) * m_lon, (lat2 - lat0) * m_lat

    width = np.hypot(x1, y1)
    if width < 1e-6:
        return False
    u = np.array([x1, y1]) / width
    v = np.array([-u[1], u[0]])
    # p2 sits on the +v side by construction, so the check goes on -v
    length = 2 * width
    corners_local = [
        (0, 0), (x1, y1),
        (x1 - v[0] * length, y1 - v[1] * length),
        (-v[0] * length, -v[1] * length),
    ]
    corners_lonlat = [(lon0 + cx / m_lon, lat0 + cy / m_lat) for cx, cy in corners_local]
    rect = Polygon(corners_lonlat)

    cand_idx = list(land_sindex.intersection(rect.bounds))
    if not cand_idx:
        return True
    nearby = land_gdf.iloc[cand_idx]
    hits = nearby[nearby.geometry.intersects(rect)]
    if not len(hits):
        return True

    exclude_deg = SELF_EXCLUDE_KM / 111.0
    self_pts = [Point(lon0, lat0), Point(lon1, lat1), Point(lon2, lat2)]
    not_self = hits.geometry.apply(lambda g: all(g.distance(p) > exclude_deg for p in self_pts))
    return not_self.sum() == 0


def build_kernel():
    if KERNEL_BIN.exists() and KERNEL_BIN.stat().st_mtime > KERNEL_SRC.stat().st_mtime:
        return
    print(f"building {KERNEL_SRC.name} with nvcc...")
    subprocess.run(["nvcc", "-O3", "-arch=native", "-o", str(KERNEL_BIN), str(KERNEL_SRC)], check=True)


def run_kernel(f_lon, f_lat, f_area, f_diam, triples, ang_lo, ang_hi, ratio_lo, ratio_hi):
    build_kernel()
    with tempfile.TemporaryDirectory() as td:
        kernel_in = Path(td) / "match_input.bin"
        kernel_out = Path(td) / "match_output.bin"

        with open(kernel_in, "wb") as f:
            np.array([len(f_lon), len(triples)], dtype=np.int64).tofile(f)
            np.array([ang_lo, ang_hi, ratio_lo, ratio_hi], dtype=np.float64).tofile(f)
            f_lon.astype(np.float64).tofile(f)
            f_lat.astype(np.float64).tofile(f)
            f_area.astype(np.float64).tofile(f)
            f_diam.astype(np.float64).tofile(f)
            triples.astype(np.int64).tofile(f)

        subprocess.run([str(KERNEL_BIN), str(kernel_in), str(kernel_out)], check=True)

        with open(kernel_out, "rb") as f:
            n_hits = np.fromfile(f, dtype=np.int64, count=1)[0]
            p0 = np.fromfile(f, dtype=np.int64, count=n_hits)
            p1 = np.fromfile(f, dtype=np.int64, count=n_hits)
            p2 = np.fromfile(f, dtype=np.int64, count=n_hits)
            angle = np.fromfile(f, dtype=np.float64, count=n_hits)
            ratio = np.fromfile(f, dtype=np.float64, count=n_hits)
    return p0, p1, p2, angle, ratio


def main():
    t0 = time.time()
    args = parse_args()
    OUT_PATH = Path(args.out)
    fp_path = Path(args.fingerprint) if args.fingerprint else latest_fingerprint()
    if fp_path is None:
        print("no fingerprint file found")
        return
    fp = json.loads(fp_path.read_text())
    ang_lo, ang_hi = fp["angle_at_P0_bounds_deg"]
    ratio_lo, ratio_hi = fp["dist_ratio_bounds"]
    print(f"fingerprint: {fp_path.name}")
    print(f"target: angle_at_P0 in [{ang_lo}, {ang_hi}] deg, dist_ratio in [{ratio_lo}, {ratio_hi}]")

    print("\n[1] loading land polygons...")
    land = gpd.read_file(SHP_PATH, bbox=(-180, LAT_MIN, 180, LAT_MAX))
    land["area_km2"] = land.geometry.area * 12300
    land_sindex = land.sindex
    print(f"    {len(land)} polygons ({time.time()-t0:.1f}s)")

    cen = land.geometry.centroid
    lon = cen.x.values.astype(np.float64)
    lat = cen.y.values.astype(np.float64)
    area = land["area_km2"].values.astype(np.float64)

    print("\n[2] local density filter...")
    coords = np.column_stack((lon, lat))
    tree = cKDTree(coords)
    local_n = np.array([len(x) - 1 for x in tree.query_ball_point(coords, LOCAL_DENSITY_RADIUS_KM / 111.0)])
    keep = local_n <= LOCAL_DENSITY_CAP
    print(f"    {keep.sum()}/{len(keep)} survive")

    f_lon, f_lat, f_area = lon[keep], lat[keep], area[keep]
    f_pool = land.iloc[keep].reset_index(drop=True)
    f_pool_sindex = f_pool.sindex

    print("\n[3] point diameters...")
    f_diam = point_diameters_m(f_pool, f_pool_sindex, f_lon, f_lat)
    print(f"    done ({time.time()-t0:.1f}s elapsed)")

    print(f"\n[4] clustering (radius={CLUSTER_RADIUS_KM}km)...")
    f_coords = np.column_stack((f_lon, f_lat))
    f_tree = cKDTree(f_coords)
    neigh = f_tree.query_ball_point(f_coords, CLUSTER_RADIUS_KM / 111.0)
    clusters = set(tuple(sorted(n)) for n in neigh if len(n) >= 3)
    print(f"    {len(clusters)} clusters")

    print("\n[5] generating triples...")
    triple_arrays = []
    for cl in clusters:
        idx_arr = np.array(cl, dtype=np.int64)
        idx_arr = stratified_sample(idx_arr, f_area, MAX_PER_CLUSTER)
        t = gen_cluster_triples(idx_arr)
        if t is not None:
            triple_arrays.append(t)
    triples = np.concatenate(triple_arrays, axis=0)
    print(f"    {len(triples):,} triples ({time.time()-t0:.1f}s elapsed)")

    print("\n[6] matching on cuda c++ kernel...")
    p0_all, p1_all, p2_all, ang_all, ratio_all = run_kernel(
        f_lon, f_lat, f_area, f_diam, triples, ang_lo, ang_hi, ratio_lo, ratio_hi)
    print(f"    {len(p0_all)} raw matches ({time.time()-t0:.1f}s elapsed)")

    if not len(p0_all):
        print("no matches found")
        return

    seen = set()
    uniq = []
    for i in range(len(p0_all)):
        key = (p0_all[i], p1_all[i], p2_all[i])
        if key not in seen:
            seen.add(key)
            uniq.append(i)
    print(f"    {len(uniq)} unique after dedup")

    print("\n[7] empty-rectangle check...")
    final = []
    for i in uniq:
        i0, i1, i2 = p0_all[i], p1_all[i], p2_all[i]
        lon0, lat0 = f_lon[i0], f_lat[i0]
        lon1, lat1 = f_lon[i1], f_lat[i1]
        lon2, lat2 = f_lon[i2], f_lat[i2]
        if rectangle_empty(lon0, lat0, lon1, lat1, lon2, lat2, land, land_sindex):
            final.append({
                "P0": [round(lon0, 6), round(lat0, 6)],
                "P1": [round(lon1, 6), round(lat1, 6)],
                "P2": [round(lon2, 6), round(lat2, 6)],
                "area_km2": [round(f_area[i0], 4), round(f_area[i1], 4), round(f_area[i2], 4)],
                "p0_diameter_m": round(float(f_diam[i0]), 1),
                "angle_at_P0_deg": round(float(ang_all[i]), 2),
                "dist_ratio": round(float(ratio_all[i]), 4),
            })

    print(f"    {len(final)}/{len(uniq)} survive")

    final.sort(key=lambda c: abs(c["angle_at_P0_deg"] - (ang_lo + ang_hi) / 2))
    OUT_PATH.write_text(json.dumps(final, indent=2))

    print(f"\n{'='*60}")
    print(f"done in {time.time()-t0:.1f}s, {len(final)} candidates -> {OUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
