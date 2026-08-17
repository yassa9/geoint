#!/usr/bin/env python3
import argparse
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import rasterio
import requests
from rasterio.errors import RasterioIOError
from rasterio.warp import transform

warnings.filterwarnings("ignore")

NDVI_THRESHOLD = 0.6
MAX_WORKERS = 20
CLOUD_COVER_MAX = 20

HERE = Path(__file__).parent
DEFAULT_IN = HERE / "candidates_oval.json"
DEFAULT_OUT = HERE / "candidates_ndvi.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default=str(DEFAULT_IN), help="input candidates json")
    p.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT), help="output candidates json")
    return p.parse_args()


def get_ndvi(lon, lat):
    query = {
        "collections": ["sentinel-2-l2a"],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "query": {"eo:cloud_cover": {"lt": CLOUD_COVER_MAX}},
        "limit": 1,
    }
    for attempt in range(4):
        try:
            r = requests.post("https://earth-search.aws.element84.com/v1/search",
                               json=query, timeout=15)
            if r.status_code == 429:
                time.sleep(2 + attempt)
                continue
            if r.status_code != 200:
                return None, f"stac_http_{r.status_code}"
            feats = r.json().get("features", [])
            if not feats:
                return None, "no_scene"
            f = feats[0]
            red_url = f["assets"]["red"]["href"]
            nir_url = f["assets"]["nir"]["href"]
            with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"):
                with rasterio.open("/vsicurl/" + red_url) as rds, \
                     rasterio.open("/vsicurl/" + nir_url) as nds:
                    # sentinel-2 tiles are utm, not wgs84, must reproject the point first
                    xs, ys = transform("EPSG:4326", rds.crs, [lon], [lat])
                    rv = float(list(rds.sample([(xs[0], ys[0])]))[0][0])
                    nv = float(list(nds.sample([(xs[0], ys[0])]))[0][0])
                    if rv + nv == 0:
                        return None, "nodata_pixel"
                    return (nv - rv) / (nv + rv), None
        except (requests.RequestException, RasterioIOError, Exception) as e:
            if attempt == 3:
                return None, f"error_{type(e).__name__}"
            time.sleep(2 + attempt)
    return None, "exhausted_retries"


def main():
    t0 = time.time()
    args = parse_args()
    IN_PATH = Path(args.in_path)
    OUT_PATH = Path(args.out_path)
    candidates = json.loads(IN_PATH.read_text())
    n = len(candidates)
    print(f"loaded {n} candidates", flush=True)

    ndvi_vals = [None] * n
    fail_reasons = [None] * n
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(get_ndvi, c["P0"][0], c["P0"][1]): i for i, c in enumerate(candidates)}
        for fut in as_completed(futures):
            i = futures[fut]
            ndvi, reason = fut.result()
            ndvi_vals[i] = ndvi
            fail_reasons[i] = reason
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (n - done) / rate if rate > 0 else 0
            tag = f"ndvi={ndvi:.3f}" if ndvi is not None else f"fail={reason}"
            print(f"  [{done}/{n}] {tag}  rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)

    survivors = []
    for c, ndvi in zip(candidates, ndvi_vals):
        if ndvi is not None:
            c["p0_ndvi"] = round(ndvi, 3)
        if ndvi is not None and ndvi >= NDVI_THRESHOLD:
            survivors.append(c)

    n_fail = sum(1 for v in ndvi_vals if v is None)
    if n_fail:
        print(f"\nNOTE: {n_fail}/{n} candidates had no usable ndvi reading", flush=True)

    print(f"\n{len(survivors)}/{n} survive (ndvi >= {NDVI_THRESHOLD})", flush=True)

    survivors.sort(key=lambda c: -c["p0_ndvi"])
    OUT_PATH.write_text(json.dumps(survivors, indent=2))
    print(f"saved -> {OUT_PATH}")
    print(f"done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
