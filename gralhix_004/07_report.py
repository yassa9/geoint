#!/usr/bin/env python3
import argparse
import json
import warnings
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

warnings.filterwarnings("ignore")

ZOOM_M = 50

HERE = Path(__file__).parent
COUNTRIES_PATH = str(HERE / "data" / "ne_10m_countries.geojson")
DEFAULT_IN = HERE / "candidates_elevation.json"
DEFAULT_OUT = HERE / "report.html"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default=str(DEFAULT_IN), help="input candidates json")
    p.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT), help="output html")
    return p.parse_args()


def country_lookup(countries, lon, lat):
    pt = Point(lon, lat)
    hit = countries[countries.contains(pt)]
    if len(hit):
        return hit.iloc[0]["NAME"]
    nearest = countries.geometry.distance(pt).idxmin()
    return countries.loc[nearest, "NAME"]


def maps_url(lon, lat):
    return f"https://www.google.com/maps/@{lat},{lon},{ZOOM_M}m/data=!3m1!1e3"


def main():
    args = parse_args()
    IN_PATH = Path(args.in_path)
    OUT_PATH = Path(args.out_path)
    candidates = json.loads(IN_PATH.read_text())
    countries = gpd.read_file(COUNTRIES_PATH)

    rows = []
    for i, c in enumerate(candidates):
        lon0, lat0 = c["P0"]
        lon1, lat1 = c["P1"]
        lon2, lat2 = c["P2"]
        name = country_lookup(countries, lon0, lat0)
        rows.append(f"""
    <tr>
      <td>{i:02d}</td>
      <td>{name}</td>
      <td><a href="{maps_url(lon0, lat0)}" target="_blank">{lat0:.6f}, {lon0:.6f}</a></td>
      <td><a href="{maps_url(lon1, lat1)}" target="_blank">{lat1:.6f}, {lon1:.6f}</a></td>
      <td><a href="{maps_url(lon2, lat2)}" target="_blank">{lat2:.6f}, {lon2:.6f}</a></td>
    </tr>""")

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>OSINT 004</title>
<style>
html {{
  background: #111;
}}
body {{
  margin: 40px auto;
  max-width: 650px;
  line-height: 1.6;
  font-size: 18px;
  color: #ccc;
  background: #111;
  padding: 0 10px;
}}
h1 {{
  line-height: 1.2;
  color: #eee;
}}
table {{
  border-collapse: collapse;
  width: 100%;
}}
td, th {{
  text-align: left;
  padding: 6px 10px 6px 0;
  border-bottom: 1px solid #333;
}}
a {{
  color: #6cb2eb;
}}
</style>
</head>
<body>
<h1>OSINT 004</h1>
<table>
  <tr><th>idx</th><th>country</th><th>P0</th><th>P1</th><th>P2</th></tr>{"".join(rows)}
</table>
</body>
</html>
"""
    OUT_PATH.write_text(html)
    print(f"{len(candidates)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
