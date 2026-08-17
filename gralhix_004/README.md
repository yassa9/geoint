# gralhix 004

Challenge: [OSINT Exercise #004 by Sofia Santos, gralhix](https://gralhix.com/list-of-osint-exercises/osint-exercise-004/)

Writeup: [placeholder]

## Setup

```sh
git clone https://github.com/yassa9/geoint.git
cd geoint/gralhix_004

pip install numpy scipy geopandas shapely matplotlib rasterio requests
mkdir data
```

Download these two datasets into `data/` (not included, too large for the repo):

```sh
cd data

# OSM split land polygons (WGS84, split), ~880MB zipped
wget https://osmdata.openstreetmap.de/download/land-polygons-split-4326.zip
unzip land-polygons-split-4326.zip
mv land-polygons-split-4326/land_polygons.* .

# Natural Earth admin-0 countries, 10m, shipped as shapefile, converted to GeoJSON
wget https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip
unzip ne_10m_admin_0_countries.zip -d ne_10m_countries
ogr2ogr -f GeoJSON ne_10m_countries.geojson ne_10m_countries/ne_10m_admin_0_countries.shp

cd ..
```

`ogr2ogr` ships with GDAL (`apt install gdal-bin` / `brew install gdal`), which geopandas/rasterio already depend on.

## Running

Each stage reads the previous stage's output and writes its own, in order:

```sh
python3 01_triangle_gui.py                                    # click P0, P1, P2 -> triangle_*.json
python3 02_search.py                                          # -> candidates_search.json
python3 03_coral_filter.py                                    # -> candidates_coral.json
python3 04_oval_filter.py                                     # -> candidates_oval.json
python3 05_ndvi_filter.py                                     # -> candidates_ndvi.json
python3 06_elevation_filter.py                                # -> candidates_elevation.json
python3 07_report.py                                          # -> report.html
```

Every stage takes `--in`/`--out` to point at a different file instead of the default.

## CUDA kernel

`02_search.py` auto-compiles and runs `gpu_kernel/match_kernel.cu` on first use (needs `nvcc`). To build/run it by hand instead:

```sh
nvcc -O3 -arch=native -o gpu_kernel/match_kernel gpu_kernel/match_kernel.cu
gpu_kernel/match_kernel input.bin output.bin
```

The binary I/O format is written/read in `run_kernel()` / `main()` in `02_search.py` and `match_kernel.cu` respectively.
