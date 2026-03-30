#!/usr/bin/env python3
# %%
"""
MOHID Lagrangian particle viewer
--------------------------------
Reads MOHID-style HDF5 outputs and plots particle positions through time,
with optional bathymetry background and particle trails colored by time.

Main features
- Reads time steps from /Time/Time_XXXXX
- Tries to find particle X/Y or Longitude/Latitude datasets automatically
- Filters invalid values such as -9.9e15 and -9999999
- Plots trails with color varying through time
- Optional bathymetry background from:
    1) a separate NetCDF file
    2) a CSV/XYZ file with lon,lat,depth columns
    3) the same HDF5 file, if lon/lat/bathymetry grids exist
- Can save a single figure, all frames, or an animated GIF/MP4

Examples
--------
python mohid_lagrangian_trails.py Lagrangian_1.hdf5 --inspect / inspecao rapida do arquivo
python mohid_lagrangian_trails.py Lagrangian.hdf5 --show / basico
python mohid_lagrangian_trails.py Lagrangian.hdf5 --save mapa.png / salva um frame
python mohid_lagrangian_trails.py Lagrangian.hdf5 --save-frames frames / salva todos os frames
python mohid_lagrangian_trails.py lagrangian.hdf5 --bathymetry batimetria.nc --show / 
python mohid_lagrangian_trails.py lagrangian.hdf5 --bathymetry batimetria.xyz --animate animacao.gif / gera uma animacao
python mohid_lagrangian_trails.py Lagrangian.hdf5 --bathymetry batimetria.nc --topography dem_s29_w049.tif dem_s28_w049.tif dem_s27_w049.tif --show / mostra batimetria e MDE local
python mohid_lagrangian_trails.py Lagrangian.hdf5 --bathymetry batimetria.nc --animate animacao.gif / gera animacao com batimetria externa NetCDF, CSV ou XYZ
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

#import matplotlib
#matplotlib.use("TkAgg")
# %%

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


INVALID_LIMIT = 1e15
DEFAULT_BAD_VALUES = (-9.9e15, -9999999.0, -99999.0)
LAND_CMAP = LinearSegmentedColormap.from_list(
    "soft_land",
    ["#f7f4ec", "#ece4d2", "#d8d4b3", "#bcc79e", "#8faa83"],
)


def is_invalid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    bad = ~np.isfinite(arr) | (np.abs(arr) > INVALID_LIMIT)
    for v in DEFAULT_BAD_VALUES:
        bad |= np.isclose(arr, v, rtol=0, atol=max(1e-12, abs(v) * 1e-12))
    return bad


def parse_mohid_time(vec: np.ndarray) -> str:
    v = np.asarray(vec).astype(int).ravel()
    if v.size < 6:
        return "tempo_desconhecido"
    y, m, d, hh, mm, ss = v[:6]
    return f"{y:04d}-{m:02d}-{d:02d} {hh:02d}:{mm:02d}:{ss:02d}"


def sort_mohid_keys(keys: Iterable[str]) -> List[str]:
    def keyfun(name: str) -> Tuple[str, int]:
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], int(parts[1])
        digits = "".join(ch for ch in name if ch.isdigit())
        return name, int(digits) if digits else -1
    return sorted(keys, key=keyfun)


def candidate_series_groups(h5: h5py.File) -> Dict[str, List[str]]:
    found = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            lname = name.lower()
            if any(tok in lname for tok in ["longitude", "latitude", "coord", "position", "x ", "y ", "/x", "/y"]):
                found[name] = list(obj.keys())
    h5.visititems(visitor)
    return found


def get_time_strings(h5: h5py.File) -> List[str]:
    if "Time" not in h5:
        raise KeyError("Grupo /Time não encontrado no HDF5.")
    keys = sort_mohid_keys(h5["Time"].keys())
    return [parse_mohid_time(h5["Time"][k][()]) for k in keys]


def find_coordinate_groups(h5: h5py.File) -> Tuple[str, str]:
    """
    Tries to find two groups representing X/Y or lon/lat particle coordinates.
    Returns paths to the two groups.
    """
    groups = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            lname = name.lower()
            if any(tok in lname for tok in ["longitude", "latitude", "x coord", "y coord", "position x", "position y", "/x", "/y"]):
                groups.append(name)

    h5.visititems(visitor)

    # Direct pair search by common parent
    by_parent = {}
    for g in groups:
        parent = g.rsplit("/", 1)[0] if "/" in g else ""
        by_parent.setdefault(parent, []).append(g)

    pairs = []
    for parent, gs in by_parent.items():
        for i, gi in enumerate(gs):
            for j, gj in enumerate(gs):
                if i >= j:
                    continue
                li, lj = gi.lower(), gj.lower()
                pair_ok = (
                    ("longitude" in li and "latitude" in lj) or
                    ("latitude" in li and "longitude" in lj) or
                    (li.endswith("/x") and lj.endswith("/y")) or
                    (li.endswith("/y") and lj.endswith("/x")) or
                    ("position x" in li and "position y" in lj) or
                    ("position y" in li and "position x" in lj)
                )
                if pair_ok:
                    pairs.append((gi, gj))

    if pairs:
        a, b = pairs[0]
        la, lb = a.lower(), b.lower()
        if "longitude" in la or la.endswith("/x") or "position x" in la:
            return a, b
        return b, a

    raise KeyError(
        "Não encontrei no HDF5 um par de grupos com coordenadas de partículas "
        "(por exemplo Longitude/Latitude ou X/Y)."
    )


def read_series_group(h5: h5py.File, group_path: str) -> List[np.ndarray]:
    g = h5[group_path]
    keys = sort_mohid_keys(g.keys())
    return [np.asarray(g[k][()], dtype=float).ravel() for k in keys]


def read_particle_tracks(hdf_path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Returns
    -------
    times : list[str]
    x : array [nt, nparticles]
    y : array [nt, nparticles]
    """
    with h5py.File(hdf_path, "r") as h5:
        times = get_time_strings(h5)
        gx, gy = find_coordinate_groups(h5)
        xs = read_series_group(h5, gx)
        ys = read_series_group(h5, gy)

    if len(xs) != len(ys):
        raise ValueError("Os grupos X e Y têm número diferente de passos de tempo.")
    if len(xs) != len(times):
        n = min(len(xs), len(times))
        times = times[:n]
        xs = xs[:n]
        ys = ys[:n]

    nsteps = len(xs)
    npart = max(arr.size for arr in xs)
    X = np.full((nsteps, npart), np.nan, dtype=float)
    Y = np.full((nsteps, npart), np.nan, dtype=float)

    for i in range(nsteps):
        xi = np.asarray(xs[i], dtype=float).ravel()
        yi = np.asarray(ys[i], dtype=float).ravel()
        n = min(xi.size, yi.size, npart)
        bad = is_invalid(xi[:n]) | is_invalid(yi[:n])
        xi = xi[:n]
        yi = yi[:n]
        xi[bad] = np.nan
        yi[bad] = np.nan
        X[i, :n] = xi
        Y[i, :n] = yi

    return times, X, Y


def _pick_first_existing(d: Dict[str, np.ndarray], names: Sequence[str]) -> Optional[np.ndarray]:
    for name in names:
        if name in d:
            return d[name]
    low_map = {k.lower(): k for k in d.keys()}
    for name in names:
        if name.lower() in low_map:
            return d[low_map[name.lower()]]
    return None


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

    lon = _pick_first_existing(arrays, ["Grid/Longitude", "Longitude", "Results/Longitude"])
    lat = _pick_first_existing(arrays, ["Grid/Latitude", "Latitude", "Results/Latitude"])
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


def read_topography_raster(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rasterio is None:
        raise ImportError("rasterio não está instalado, necessário para ler raster local de altimetria.")

    raster_path = Path(path)
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


def nice_limits(x: np.ndarray, y: np.ndarray, pad_fraction: float = 0.08) -> Tuple[float, float, float, float]:
    xx = x[np.isfinite(x)]
    yy = y[np.isfinite(y)]
    if xx.size == 0 or yy.size == 0:
        return -1, 1, -1, 1
    xmin, xmax = xx.min(), xx.max()
    ymin, ymax = yy.min(), yy.max()
    dx = xmax - xmin
    dy = ymax - ymin
    if dx == 0:
        dx = max(abs(xmin) * 0.01, 1e-6)
    if dy == 0:
        dy = max(abs(ymin) * 0.01, 1e-6)
    return (
        xmin - dx * pad_fraction,
        xmax + dx * pad_fraction,
        ymin - dy * pad_fraction,
        ymax + dy * pad_fraction,
    )


def _looks_like_lonlat(xmin: float, xmax: float, ymin: float, ymax: float) -> bool:
    return (
        -180.0 <= xmin <= 180.0 and
        -180.0 <= xmax <= 180.0 and
        -90.0 <= ymin <= 90.0 and
        -90.0 <= ymax <= 90.0
    )


def _nice_number(value: float) -> float:
    if value <= 0:
        return 0.0
    exponent = np.floor(np.log10(value))
    fraction = value / (10 ** exponent)
    if fraction < 1.5:
        nice_fraction = 1.0
    elif fraction < 3.5:
        nice_fraction = 2.0
    elif fraction < 7.5:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * (10 ** exponent)


def add_north_arrow(ax, xpos: float = 0.89, ypos: float = 0.80, size: float = 0.20) -> None:
    image_path = Path(__file__).with_name("NorthArrow.png")
    if not image_path.exists():
        return

    try:
        image = plt.imread(image_path)
    except Exception:
        return

    if image.ndim < 2 or image.shape[0] == 0 or image.shape[1] == 0:
        return

    height_px, width_px = image.shape[:2]
    height = size
    width = size * (width_px / height_px)
    x0 = xpos
    y0 = ypos
    inset = ax.inset_axes([x0, y0, width, height], transform=ax.transAxes)
    inset.imshow(image)
    inset.set_axis_off()


def choose_scale_bar_position(
    X: np.ndarray,
    Y: np.ndarray,
    frame_idx: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Tuple[float, float]:
    xspan = xmax - xmin
    yspan = ymax - ymin
    x_right_min = xmin + 0.74 * xspan
    x_right_max = xmin + 0.98 * xspan
    x_left_min = xmin + 0.02 * xspan
    x_left_max = xmin + 0.32 * xspan
    y_bottom_min = ymin + 0.02 * yspan
    y_bottom_max = ymin + 0.12 * yspan

    xp = np.asarray(X[: frame_idx + 1], dtype=float)
    yp = np.asarray(Y[: frame_idx + 1], dtype=float)
    xp = xp[np.isfinite(xp)]
    yp = yp[np.isfinite(yp)]

    particle_hits = np.count_nonzero(
        (xp >= x_right_min) & (xp <= x_right_max) & (yp >= y_bottom_min) & (yp <= y_bottom_max)
    )
    right_busy = particle_hits >= 3

    if right_busy:
        return 0.08, 0.08
    return 0.72, 0.08


def add_scale_bar(
    ax,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xpos: float = 0.08,
    ypos: float = 0.08,
) -> None:
    xspan = xmax - xmin
    yspan = ymax - ymin
    if xspan <= 0 or yspan <= 0:
        return

    is_lonlat = _looks_like_lonlat(xmin, xmax, ymin, ymax)
    mean_lat = 0.5 * (ymin + ymax)

    if is_lonlat:
        meters_per_degree_x = 111_320.0 * max(np.cos(np.deg2rad(mean_lat)), 1e-6)
        target_meters = 0.20 * xspan * meters_per_degree_x
        bar_meters = _nice_number(target_meters)
        bar_dx = bar_meters / meters_per_degree_x
        if bar_meters >= 1000:
            label = f"{bar_meters / 1000:.0f} km"
        else:
            label = f"{bar_meters:.0f} m"
    else:
        target_units = 0.20 * xspan
        bar_dx = _nice_number(target_units)
        label = f"{bar_dx:g} units"

    x0 = xmin + xpos * xspan
    y0 = ymin + ypos * yspan
    tick_h = 0.008 * yspan
    bar_h = 0.012 * yspan
    segment_count = 4
    segment_dx = bar_dx / segment_count

    for i in range(segment_count):
        xi = x0 + i * segment_dx
        face = "black" if i % 2 == 0 else "white"
        rect = plt.Rectangle(
            (xi, y0),
            segment_dx,
            bar_h,
            facecolor=face,
            edgecolor="#202020",
            lw=0.7,
            zorder=6,
        )
        ax.add_patch(rect)

    ax.plot([x0, x0 + bar_dx], [y0 + bar_h, y0 + bar_h], color="#202020", lw=0.7, zorder=6)
    ax.plot([x0, x0], [y0, y0 + bar_h + tick_h], color="#202020", lw=0.7, zorder=6)
    ax.plot([x0 + bar_dx, x0 + bar_dx], [y0, y0 + bar_h + tick_h], color="#202020", lw=0.7, zorder=6)
    ax.text(
        x0 + 0.5 * bar_dx,
        y0 + bar_h + 1.4 * tick_h,
        label,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#303030",
        bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.72),
        zorder=6,
    )


def style_map_axes(ax) -> None:
    ax.set_xlabel("Longitude", fontsize=9, color="#4a4a4a", labelpad=6)
    ax.set_ylabel("Latitude", fontsize=9, color="#4a4a4a", labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=8, colors="#6a6a6a", width=0.6, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#5a5a5a")


def add_bathy_to_axis(ax, bathy, alpha=0.9):
    if bathy is None:
        return None
    lon, lat, dep = bathy
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    dep = np.asarray(dep, dtype=float)

    dep = np.where(is_invalid(dep), np.nan, dep)
    finite = dep[np.isfinite(dep)]
    if finite.size == 0:
        return None

    pos_count = np.count_nonzero(finite > 0)
    neg_count = np.count_nonzero(finite < 0)
    if pos_count >= neg_count:
        dep = np.where(dep > 0, dep, np.nan)
    else:
        dep = np.where(dep < 0, np.abs(dep), np.nan)

    cmap = plt.get_cmap("Blues_r").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    dep_max = np.nanmax(dep) if np.isfinite(dep).any() else 0.0

    if lon.ndim == 2 and lat.ndim == 2 and dep.ndim == 2:
        m = ax.pcolormesh(
            lon,
            lat,
            np.ma.masked_invalid(dep),
            shading="auto",
            cmap=cmap,
            alpha=alpha,
            vmin=0,
            vmax=dep_max,
        )
        try:
            ax.contour(lon, lat, dep, levels=8, linewidths=0.4, alpha=0.4)
        except Exception:
            pass
        return m

    # scattered
    if lon.ndim == 1 and lat.ndim == 1 and dep.ndim == 1:
        good = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(dep)
        if np.count_nonzero(good) < 3:
            return None
        levels = np.linspace(0, dep_max, 20) if dep_max > 0 else 20
        m = ax.tricontourf(
            lon[good],
            lat[good],
            dep[good],
            levels=levels,
            cmap=cmap,
            alpha=alpha,
            vmin=0,
            vmax=dep_max,
        )
        try:
            ax.tricontour(lon[good], lat[good], dep[good], levels=8, linewidths=0.4, alpha=0.4)
        except Exception:
            pass
        return m

    return None


def _as_topography_list(topography) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if topography is None:
        return []
    if isinstance(topography, list):
        return topography
    return [topography]


def add_topography_to_axis(
    ax,
    topography,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    alpha=0.55,
):
    artist = None
    cmap = LAND_CMAP.copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    for lon, lat, elev in _as_topography_list(topography):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        elev = np.asarray(elev, dtype=float)
        elev = np.where(is_invalid(elev), np.nan, elev)
        elev = np.where(elev > 0, elev, np.nan)

        if lon.ndim == 2 and lat.ndim == 2 and elev.ndim == 2:
            in_domain = (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)
            elev = np.where(in_domain, elev, np.nan)
        elif lon.ndim == 1 and lat.ndim == 1 and elev.ndim == 1:
            in_domain = (lon >= xmin) & (lon <= xmax) & (lat >= ymin) & (lat <= ymax)
            elev = np.where(in_domain, elev, np.nan)

        if not np.isfinite(elev).any():
            continue

        vmax = np.nanpercentile(elev, 95)
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else np.nanmax(elev)
        norm = Normalize(vmin=0, vmax=vmax if vmax > 0 else 1.0)

        if lon.ndim == 2 and lat.ndim == 2 and elev.ndim == 2:
            filled = np.where(np.isfinite(elev), elev, 0.0)
            gy, gx = np.gradient(filled)
            slope = np.hypot(gx, gy)
            slope = np.where(np.isfinite(elev), slope, np.nan)
            shade = 1.0 - 0.35 * slope / np.nanpercentile(slope, 95)
            shade = np.clip(shade, 0.72, 1.0)
            rgba = cmap(norm(np.ma.masked_invalid(elev)))
            rgba[..., :3] *= np.where(np.isfinite(shade)[..., None], shade[..., None], 1.0)
            rgba[..., 3] = np.where(np.isfinite(elev), alpha, 0.0)
            x0 = float(np.nanmin(lon))
            x1 = float(np.nanmax(lon))
            y0 = float(np.nanmin(lat))
            y1 = float(np.nanmax(lat))
            origin = "upper" if lat[0, 0] > lat[-1, 0] else "lower"
            artist = ax.imshow(
                rgba,
                extent=[x0, x1, y0, y1],
                origin=origin,
                interpolation="bilinear",
                zorder=2,
                aspect="auto",
            )
            continue

        if lon.ndim == 1 and lat.ndim == 1 and elev.ndim == 1:
            good = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(elev)
            if np.count_nonzero(good) < 3:
                continue
            artist = ax.tricontourf(lon[good], lat[good], elev[good], levels=18, cmap=cmap, norm=norm, alpha=alpha, zorder=2)

    return artist


def _expand_limits_with_bathy(
    limits: Tuple[float, float, float, float],
    bathy: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = limits
    if bathy is None:
        return limits

    try:
        bx = np.asarray(bathy[0], dtype=float)
        by = np.asarray(bathy[1], dtype=float)
        if np.isfinite(bx).any() and np.isfinite(by).any():
            xmin = min(xmin, np.nanmin(bx))
            xmax = max(xmax, np.nanmax(bx))
            ymin = min(ymin, np.nanmin(by))
            ymax = max(ymax, np.nanmax(by))
    except Exception:
        pass
    return xmin, xmax, ymin, ymax


def create_frame_figure(
    times: List[str],
    X: np.ndarray,
    Y: np.ndarray,
    frame_idx: int,
    bathy=None,
    topography=None,
    output: Optional[str] = None,
    title_prefix: str = "Partículas Lagrangianas",
    dpi: int = 140,
):
    nsteps, npart = X.shape
    fig, ax = plt.subplots(figsize=(9, 7))
    xmin, xmax, ymin, ymax = nice_limits(X, Y)
    xmin, xmax, ymin, ymax = _expand_limits_with_bathy((xmin, xmax, ymin, ymax), bathy)
    bathy_artist = add_bathy_to_axis(ax, bathy)
    add_topography_to_axis(ax, topography, xmin, xmax, ymin, ymax)
    norm = Normalize(vmin=0, vmax=max(nsteps - 1, 1))

    # Trails
    for p in range(npart):
        xp = X[: frame_idx + 1, p]
        yp = Y[: frame_idx + 1, p]
        good = np.isfinite(xp) & np.isfinite(yp)
        if np.count_nonzero(good) < 2:
            continue
        idx = np.where(good)[0]
        pts = np.column_stack([xp[good], yp[good]])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", norm=norm, linewidths=2.2)
        lc.set_array(idx[1:])
        ax.add_collection(lc)
        ax.plot(pts[-1, 0], pts[-1, 1], "o", ms=5, zorder=5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    style_map_axes(ax)
    ax.set_title(f"{title_prefix}\n{times[frame_idx]}", fontsize=11, color="#2f2f2f", pad=8)
    ax.grid(True, alpha=0.18, color="#9aa4ad", linewidth=0.55)
    ax.set_aspect("equal" if abs((xmax - xmin) / max(ymax - ymin, 1e-12)) < 8 else "auto")
    add_north_arrow(ax)
    scale_xpos, scale_ypos = choose_scale_bar_position(X, Y, frame_idx, xmin, xmax, ymin, ymax)
    add_scale_bar(ax, xmin, xmax, ymin, ymax, xpos=scale_xpos, ypos=scale_ypos)

    # Colorbar for trail time
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    divider = make_axes_locatable(ax)
    cax_time = divider.append_axes("right", size="3%", pad=0.20)
    cbar = fig.colorbar(sm, cax=cax_time)
    cbar.set_label("Índice do tempo")
    if bathy_artist is not None:
        try:
            cax_bathy = divider.append_axes("right", size="3%", pad=0.55)
            cbar2 = fig.colorbar(bathy_artist, cax=cax_bathy)
            cbar2.set_label("Batimetria")
        except Exception:
            pass

    fig.tight_layout()
    return fig


def plot_frame(
    times: List[str],
    X: np.ndarray,
    Y: np.ndarray,
    frame_idx: int,
    bathy=None,
    topography=None,
    output: Optional[str] = None,
    title_prefix: str = "Partículas Lagrangianas",
    dpi: int = 140,
):
    fig = create_frame_figure(
        times,
        X,
        Y,
        frame_idx,
        bathy=bathy,
        topography=topography,
        title_prefix=title_prefix,
        dpi=dpi,
    )

    if output:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_animation(
    times: List[str],
    X: np.ndarray,
    Y: np.ndarray,
    bathy,
    topography,
    out_path: str,
    fps: int = 3,
    dpi: int = 110,
):
    tmp_dir = Path("_tmp_mohid_frames")
    tmp_dir.mkdir(exist_ok=True)
    frames = []
    for i in range(len(times)):
        frame_path = tmp_dir / f"frame_{i:05d}.png"
        plot_frame(times, X, Y, i, bathy=bathy, topography=topography, output=str(frame_path), dpi=dpi)
        frames.append(imageio.imread(frame_path))

    out = Path(out_path)
    if out.suffix.lower() == ".gif":
        imageio.mimsave(out, frames, fps=fps, loop=0)
    elif out.suffix.lower() in [".mp4", ".m4v"]:
        imageio.mimsave(out, frames, fps=fps)
    else:
        raise ValueError("Use .gif ou .mp4 para animação.")

    for p in tmp_dir.glob("frame_*.png"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass


def inspect_hdf(hdf_path: str):
    print(f"\nInspeção rápida de: {hdf_path}")
    with h5py.File(hdf_path, "r") as h5:
        print("\nGrupos principais:")
        for k in h5.keys():
            print(f"  - {k}")

        if "Time" in h5:
            tkeys = sort_mohid_keys(h5["Time"].keys())
            print(f"\nPassos de tempo: {len(tkeys)}")
            if tkeys:
                print("Primeiro tempo:", parse_mohid_time(h5["Time"][tkeys[0]][()]))
                print("Último tempo  :", parse_mohid_time(h5["Time"][tkeys[-1]][()]))

        print("\nGrupos candidatos a coordenadas:")
        cand = candidate_series_groups(h5)
        for name, ks in list(cand.items())[:20]:
            print(f"  - {name} ({len(ks)} datasets)")
        try:
            gx, gy = find_coordinate_groups(h5)
            print("\nPar de coordenadas detectado:")
            print("  X/Lon:", gx)
            print("  Y/Lat:", gy)
        except Exception as e:
            print("\nNenhum par de coordenadas detectado automaticamente.")
            print("Motivo:", e)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualização de partículas lagrangianas do MOHID")
    p.add_argument("hdf5", help="Arquivo HDF5 do resultado lagrangiano")
    p.add_argument("--bathymetry", help="Arquivo opcional de batimetria (.nc, .csv, .xyz, .hdf5)")
    p.add_argument("--topography", nargs="+", help="Um ou mais rasters locais opcionais de altimetria/elevação terrestre (ex.: GeoTIFF)")
    p.add_argument("--frame", type=int, default=None, help="Índice do passo de tempo a plotar")
    p.add_argument("--show", action="store_true", help="Exibe a figura na tela")
    p.add_argument("--save", help="Salva uma figura única em PNG")
    p.add_argument("--save-frames", help="Diretório para salvar todos os frames em PNG")
    p.add_argument("--animate", help="Salva animação .gif ou .mp4")
    p.add_argument("--fps", type=int, default=3, help="FPS da animação")
    p.add_argument("--inspect", action="store_true", help="Inspeciona a estrutura do HDF5 e sai")
    return p


def main():
    args = build_parser().parse_args()

    if args.inspect:
        inspect_hdf(args.hdf5)
        return

    times, X, Y = read_particle_tracks(args.hdf5)
    bathy = None
    topography = None
    try:
        bathy = read_bathy(args.bathymetry, fallback_hdf=args.hdf5 if args.bathymetry is None else None)
    except Exception as e:
        print(f"[aviso] Não foi possível carregar batimetria: {e}")
    if args.topography:
        try:
            topography = read_topography_rasters(args.topography)
        except Exception as e:
            print(f"[aviso] Não foi possível carregar altimetria: {e}")

    if args.save_frames:
        outdir = Path(args.save_frames)
        outdir.mkdir(parents=True, exist_ok=True)
        for i in range(len(times)):
            out = outdir / f"particles_{i:05d}.png"
            plot_frame(times, X, Y, i, bathy=bathy, topography=topography, output=str(out))
        print(f"Frames salvos em: {outdir}")

    if args.animate:
        save_animation(times, X, Y, bathy, topography, args.animate, fps=args.fps)
        print(f"Animação salva em: {args.animate}")

    frame_idx = args.frame if args.frame is not None else len(times) - 1
    frame_idx = max(0, min(frame_idx, len(times) - 1))

    if args.save:
        plot_frame(times, X, Y, frame_idx, bathy=bathy, topography=topography, output=args.save)
        print(f"Figura salva em: {args.save}")

    if args.show or (not args.save and not args.save_frames and not args.animate):
        plot_frame(times, X, Y, frame_idx, bathy=bathy, topography=topography, output=None)


if __name__ == "__main__":
    main()
