# MOHID Lagrangian Map

Python tool for inspecting and plotting MOHID Lagrangian particle outputs stored in HDF5 files.

It is designed for quick scientific visualization of particle trajectories with optional bathymetry and local DEM/topography support, plus export of static maps and animations.

## Example

![Example map](docs/assets/example_map.png)

## Features

- automatic discovery of particle coordinate groups in MOHID-style HDF5 files
- trail visualization with time-colored trajectories
- quick HDF5 inspection mode
- bathymetry support from HDF5, NetCDF, CSV, TXT, or XYZ
- local DEM/topography support from raster files
- north arrow, scale bar, and cartographic finishing
- export to PNG, frame sequences, GIF, and MP4

## Main File

- `mohid_lagrangian_trails.py`

## Installation

Install the base dependencies:

```bash
pip install -r requirements.txt
```

Optional packages:

- `pandas` for CSV/XYZ bathymetry
- `xarray` for NetCDF bathymetry
- `rasterio` for DEM/topography rasters

Example:

```bash
pip install pandas xarray rasterio
```

## Basic Usage

Inspect the structure of a MOHID HDF5 file:

```bash
python3 mohid_lagrangian_trails.py Lagrangian_1.hdf5 --inspect
```

Show the latest frame:

```bash
python3 mohid_lagrangian_trails.py Lagrangian_1.hdf5 --show
```

Save a single map:

```bash
python3 mohid_lagrangian_trails.py Lagrangian_1.hdf5 --save mapa.png
```

Save all frames:

```bash
python3 mohid_lagrangian_trails.py Lagrangian_1.hdf5 --save-frames frames
```

Generate an animation:

```bash
python3 mohid_lagrangian_trails.py Lagrangian_1.hdf5 --animate animacao.gif
```

## Bathymetry Examples

Using an external NetCDF bathymetry file:

```bash
python3 mohid_lagrangian_trails.py Lagrangian_1.hdf5 \
  --bathymetry batimetria.nc \
  --show
```

Using CSV/XYZ bathymetry:

```bash
python3 mohid_lagrangian_trails.py Lagrangian_1.hdf5 \
  --bathymetry batimetria.xyz \
  --show
```

## DEM / Topography Example

Using local DEM tiles together with bathymetry:

```bash
python3 mohid_lagrangian_trails.py Lagrangian_1.hdf5 \
  --bathymetry batimetria.nc \
  --topography dem_s29_w049.tif dem_s28_w049.tif dem_s27_w049.tif \
  --show
```

## Outputs

The script can:

- display the figure interactively
- save a static PNG map
- save a PNG sequence for all time steps
- create GIF or MP4 animations

## Repository Notes

- large scientific input files are ignored by Git
- generated outputs such as maps and animations are ignored by Git
- `NorthArrow.png` is versioned because it is part of the cartographic layout

## License

This project is distributed under the MIT License. See `LICENSE` for details.
