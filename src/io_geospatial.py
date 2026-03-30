from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np

from config import TOPOGRAPHY_ASSETS_DIR
from .io_mohid import is_invalid, pick_first_existing

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import xarray as xr
except Exception:
    xr = None

try:
    import rasterio
except Exception:
    rasterio = None


def read_bathy_from_hdf(hdf_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    arrays = {}
    with h5py.File(hdf_path, "r") as h5:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                lname = name.lower()
                if any(tok in lname for tok in ["bathymetry", "bottom", "depth", "longitude", "latitude", "grid"]):
                    try:
                        arrays[name] = np.asarray(obj[()])
                    except Exception:
                        pass

        h5.visititems(visitor)

    if not arrays:
        return None

    lon = pick_first_existing(arrays, ["Grid/Longitude", "Longitude", "Results/Longitude"])
    lat = pick_first_existing(arrays, ["Grid/Latitude", "Latitude", "Results/Latitude"])
    dep = None
    for k, v in arrays.items():
        lk = k.lower()
        if any(tok in lk for tok in ["bathymetry", "bottom", "depth"]):
            dep = np.asarray(v)
            break

    if lon is None or lat is None or dep is None:
        return None

    return np.asarray(lon), np.asarray(lat), np.asarray(dep)


def _read_bathy_table(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pd is None:
        raise ImportError("pandas não está instalado, necessário para ler CSV/XYZ.")
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, header=None)

    cols_low = [str(c).strip().lower() for c in df.columns]
    if {"lon", "lat", "depth"}.issubset(set(cols_low)):
        lon = df[df.columns[cols_low.index("lon")]].to_numpy()
        lat = df[df.columns[cols_low.index("lat")]].to_numpy()
        dep = df[df.columns[cols_low.index("depth")]].to_numpy()
    elif {"longitude", "latitude", "depth"}.issubset(set(cols_low)):
        lon = df[df.columns[cols_low.index("longitude")]].to_numpy()
        lat = df[df.columns[cols_low.index("latitude")]].to_numpy()
        dep = df[df.columns[cols_low.index("depth")]].to_numpy()
    elif df.shape[1] >= 3:
        lon = df.iloc[:, 0].to_numpy()
        lat = df.iloc[:, 1].to_numpy()
        dep = df.iloc[:, 2].to_numpy()
    else:
        raise ValueError("CSV/XYZ precisa ter pelo menos 3 colunas: lon, lat, depth.")

    xu = np.unique(lon)
    yu = np.unique(lat)
    if xu.size * yu.size == lon.size:
        z_grid = np.full((yu.size, xu.size), np.nan, dtype=float)
        xi = {v: i for i, v in enumerate(xu)}
        yi = {v: i for i, v in enumerate(yu)}
        for x, y, z in zip(lon, lat, dep):
            z_grid[yi[y], xi[x]] = z
        lon_grid, lat_grid = np.meshgrid(xu, yu)
        return lon_grid, lat_grid, z_grid

    return lon, lat, dep


def _read_bathy_netcdf(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if xr is None:
        raise ImportError("xarray não está instalado, necessário para ler NetCDF.")

    with xr.open_dataset(path) as ds:
        lon_name = next((c for c in ds.coords if c.lower() in ["lon", "longitude", "x"]), None)
        lat_name = next((c for c in ds.coords if c.lower() in ["lat", "latitude", "y"]), None)
        depth_name = next((v for v in ds.data_vars if v.lower() in ["depth", "bathymetry", "h", "bottom_depth"]), None)

        if depth_name is None:
            for v in ds.data_vars:
                if ds[v].ndim >= 2:
                    depth_name = v
                    break

        if lon_name is None or lat_name is None or depth_name is None:
            raise ValueError("Não consegui identificar lon/lat/depth no NetCDF de batimetria.")

        dep = ds[depth_name].values
        lon = ds[lon_name].values
        lat = ds[lat_name].values

    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    return np.asarray(lon), np.asarray(lat), np.asarray(dep)


def read_bathy(bathy_path: Optional[str], fallback_hdf: Optional[str] = None) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if bathy_path is None:
        return read_bathy_from_hdf(fallback_hdf) if fallback_hdf is not None else None

    path = Path(bathy_path)
    suffix = path.suffix.lower()

    if suffix in [".csv", ".txt", ".xyz"]:
        return _read_bathy_table(path)

    if suffix in [".nc", ".nc4", ".cdf"]:
        return _read_bathy_netcdf(path)

    if suffix in [".h5", ".hdf5"]:
        out = read_bathy_from_hdf(str(path))
        if out is None:
            raise ValueError("Não encontrei lon/lat/batimetria no HDF5 informado.")
        return out

    raise ValueError(f"Formato de batimetria não suportado: {suffix}")


def resolve_topography_path(path: str) -> Path:
    raster_path = Path(path)
    if raster_path.exists():
        return raster_path

    fallback_path = TOPOGRAPHY_ASSETS_DIR / raster_path.name
    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(f"Raster de topografia não encontrado: {path}")


def read_topography_raster(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rasterio is None:
        raise ImportError("rasterio não está instalado, necessário para ler raster local de altimetria.")

    raster_path = resolve_topography_path(path)
    with rasterio.open(raster_path) as src:
        elev = src.read(1).astype(float)
        nodata = src.nodata
        bounds = src.bounds
        width = src.width
        height = src.height

    if nodata is not None:
        elev = np.where(np.isclose(elev, nodata), np.nan, elev)
    elev = np.where(is_invalid(elev), np.nan, elev)

    dx = (bounds.right - bounds.left) / max(width, 1)
    dy = (bounds.top - bounds.bottom) / max(height, 1)
    xs = bounds.left + (np.arange(width) + 0.5) * dx
    ys = bounds.top - (np.arange(height) + 0.5) * dy
    lon, lat = np.meshgrid(xs, ys)
    return lon, lat, elev


def read_topography_rasters(paths: Sequence[str]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [read_topography_raster(path) for path in paths]
