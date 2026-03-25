# MOHID Particle Viewer

Python script for inspecting and plotting MOHID Lagrangian particle outputs stored in HDF5 files.

The project focuses on quick scientific visualization of particle tracks, with optional:

- bathymetry background from HDF5, NetCDF, CSV, or XYZ
- local DEM/topography rasters
- trail coloring by time
- frame export and GIF/MP4 animation

## Main Script

- `mohid_lagrangian_trails.py`

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Optional readers:

- `pandas` for CSV/XYZ bathymetry
- `xarray` for NetCDF bathymetry
- `rasterio` for local DEM/topography rasters

## Usage

Quick inspection of an HDF5 file:

```bash
python3 mohid_lagrangian_trails.py Lagrangian_1.hdf5 --inspect
```

Show the latest frame:

```bash
python3 mohid_lagrangian_trails.py Lagrangian_1.hdf5 --show
```

Save a map:

```bash
python3 mohid_lagrangian_trails.py Lagrangian_1.hdf5 --save mapa.png
```

Show bathymetry and local DEM:

```bash
python3 mohid_lagrangian_trails.py Lagrangian_1.hdf5 \
  --bathymetry batimetria.nc \
  --topography dem_s29_w049.tif dem_s28_w049.tif dem_s27_w049.tif \
  --show
```

Generate an animation:

```bash
python3 mohid_lagrangian_trails.py Lagrangian_1.hdf5 --animate animacao.gif
```

## Notes

- Large input datasets are intentionally ignored by Git in this repository.
- `NorthArrow.png` is kept under version control because it is part of the map layout.
- If you want to publish sample datasets, it is better to use a separate release asset or external storage.
