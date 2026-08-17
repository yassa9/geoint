#!/usr/bin/env python3
import argparse
import json
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

warnings.filterwarnings("ignore")

COMPACTNESS_MIN = 0.5
MICRO_KM2 = 0.05
HALO_KM = 1.5
MICRO_CAY_MIN = 1

LAT_MIN, LAT_MAX = -30.0, 30.0

HERE = Path(__file__).parent
SHP_PATH = str(HERE / "data" / "land_polygons.shp")
DEFAULT_IN = HERE / "candidates_search.json"
DEFAULT_OUT = HERE / "candidates_coral.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default=str(DEFAULT_IN), help="input candidates json")
    p.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT), help="output candidates json")
    return p.parse_args()


def own_polygon(gdf, sindex, lon, lat):
    pt = Point(lon, lat)
    bounds = (lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02)
    idx = list(sindex.intersection(bounds))
    if not idx:
        return None
    nearby = gdf.iloc[idx]
    row_idx = nearby.geometry.distance(pt).idxmin()
    return gdf.loc[row_idx]


def compactness(row):
    return (4 * np.pi * row.area_km2) / (row.perim_km ** 2 + 1e-12)


def micro_cay_count(gdf, sindex, lon, lat):
    pt = Point(lon, lat)
    bounds = (lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02)
    idx = list(sindex.intersection(bounds))
    if not idx:
        return 0
    nearby = gdf.iloc[idx]
    dists_km = nearby.geometry.distance(pt) * 111.0
    mask = (dists_km > 0) & (dists_km <= HALO_KM) & (nearby["area_km2"].values < MICRO_KM2)
    return int(mask.sum())


def main():
    t0 = time.time()
    args = parse_args()
    IN_PATH = Path(args.in_path)
    OUT_PATH = Path(args.out_path)
    candidates = json.loads(IN_PATH.read_text())
    print(f"loaded {len(candidates)} candidates")

    land = gpd.read_file(SHP_PATH, bbox=(-180, LAT_MIN, 180, LAT_MAX))
    land["area_km2"] = land.geometry.area * 12300
    land["perim_km"] = land.geometry.length * 111.0
    sindex = land.sindex
    print(f"land polygons ready ({time.time()-t0:.1f}s)")

    survivors = []
    for c in candidates:
        lon, lat = c["P0"]
        row = own_polygon(land, sindex, lon, lat)
        if row is None:
            continue

        pp = compactness(row)
        if pp < COMPACTNESS_MIN:
            continue

        cays = micro_cay_count(land, sindex, lon, lat)
        if cays < MICRO_CAY_MIN:
            continue

        c["p0_compactness"] = round(float(pp), 3)
        c["p0_micro_cays"] = cays
        survivors.append(c)

    print(f"{len(survivors)}/{len(candidates)} survive "
          f"(compactness>={COMPACTNESS_MIN}, micro_cays>={MICRO_CAY_MIN} within {HALO_KM}km)")

    survivors.sort(key=lambda c: -c["p0_compactness"])
    OUT_PATH.write_text(json.dumps(survivors, indent=2))
    print(f"saved -> {OUT_PATH}")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
