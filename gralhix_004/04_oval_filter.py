#!/usr/bin/env python3
import argparse
import json
import math
import time
import warnings
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

warnings.filterwarnings("ignore")

ASPECT_RATIO_RANGE = (1.05, 2.2)
IDEAL_ELLIPSE_FILL = math.pi / 4  # exact fill ratio of any ellipse in its min-area bounding rect
FILL_RATIO_MIN = 0.75 * IDEAL_ELLIPSE_FILL

LAT_MIN, LAT_MAX = -30.0, 30.0

HERE = Path(__file__).parent
SHP_PATH = str(HERE / "data" / "land_polygons.shp")
DEFAULT_IN = HERE / "candidates_coral.json"
DEFAULT_OUT = HERE / "candidates_oval.json"


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
    return gdf.geometry.loc[row_idx]


def aspect_and_fill(geom):
    mrr = geom.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    s1 = math.hypot(coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    s2 = math.hypot(coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
    long_side, short_side = max(s1, s2), min(s1, s2)
    if short_side < 1e-9:
        return None, None
    return long_side / short_side, geom.area / mrr.area if mrr.area > 0 else 0


def main():
    t0 = time.time()
    args = parse_args()
    IN_PATH = Path(args.in_path)
    OUT_PATH = Path(args.out_path)
    candidates = json.loads(IN_PATH.read_text())
    print(f"loaded {len(candidates)} candidates")

    land = gpd.read_file(SHP_PATH, bbox=(-180, LAT_MIN, 180, LAT_MAX))
    sindex = land.sindex
    print(f"land polygons ready ({time.time()-t0:.1f}s)")

    survivors = []
    for c in candidates:
        lon, lat = c["P0"]
        geom = own_polygon(land, sindex, lon, lat)
        if geom is None:
            continue

        aspect, fill = aspect_and_fill(geom)
        if aspect is None:
            continue

        c["p0_aspect_ratio"] = round(aspect, 3)
        c["p0_fill_ratio"] = round(fill, 3)

        if ASPECT_RATIO_RANGE[0] <= aspect <= ASPECT_RATIO_RANGE[1] and fill >= FILL_RATIO_MIN:
            survivors.append(c)

    print(f"{len(survivors)}/{len(candidates)} survive "
          f"(aspect {ASPECT_RATIO_RANGE[0]}-{ASPECT_RATIO_RANGE[1]}, fill>={FILL_RATIO_MIN})")

    OUT_PATH.write_text(json.dumps(survivors, indent=2))
    print(f"saved -> {OUT_PATH}")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
