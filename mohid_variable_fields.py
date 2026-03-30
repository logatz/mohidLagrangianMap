#!/usr/bin/env python3
"""
MOHID field viewer for hydrodynamic and water-quality variables.

Examples
--------
python3 mohid_variable_fields.py --curr --save correntes.png
python3 mohid_variable_fields.py --curr --layer 0 --save correntes_superficie.png
python3 mohid_variable_fields.py --wlev --save nivel_mar.png
python3 mohid_variable_fields.py --sali --layer 0 --save salinidade.png
python3 mohid_variable_fields.py --temp --layer 0 --topography dem_s29_w049.tif dem_s28_w049.tif dem_s27_w049.tif --save temperatura.png
python3 mohid_variable_fields.py --oxy --layer 5 --animate oxigenio.gif
python3 mohid_variable_fields.py --var phytoplankton --input WaterProperties_2.hdf5 --layer 0 --save fito.png
python3 mohid_variable_fields.py --inspect
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

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
COLIFORM_CMAP = LinearSegmentedColormap.from_list(
    "coliforms",
    [
        (0.0, (0.72, 0.89, 1.00, 0.00)),
        (0.18, (0.72, 0.89, 1.00, 0.55)),
        (0.45, (0.46, 0.75, 0.98, 0.85)),
        (0.75, (0.99, 0.63, 0.28, 0.95)),
        (1.0, (0.80, 0.08, 0.12, 1.00)),
    ],
)


@dataclass(frozen=True)
class VariableSpec:
    key: str
    title: str
    label: str
    source: str
    mode: str
    group: Optional[str] = None
    group_u: Optional[str] = None
    group_v: Optional[str] = None
    group_mag: Optional[str] = None
    cmap: str = "viridis"
    center_zero: bool = False
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    log_scale: bool = False


def scalar_spec(
    key: str,
    title: str,
    label: str,
    group: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center_zero: bool = False,
    log_scale: bool = False,
) -> VariableSpec:
    return VariableSpec(
        key=key,
        title=title,
        label=label,
        source="water",
        mode="scalar",
        group=group,
        cmap=cmap,
        center_zero=center_zero,
        vmin=vmin,
        vmax=vmax,
        log_scale=log_scale,
    )


def hydro_scalar_spec(
    key: str,
    title: str,
    label: str,
    group: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center_zero: bool = False,
    log_scale: bool = False,
) -> VariableSpec:
    return VariableSpec(
        key=key,
        title=title,
        label=label,
        source="hydro",
        mode="scalar",
        group=group,
        cmap=cmap,
        center_zero=center_zero,
        vmin=vmin,
        vmax=vmax,
        log_scale=log_scale,
    )


def vector_spec(
    key: str,
    title: str,
    label: str,
    group_u: str,
    group_v: str,
    group_mag: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> VariableSpec:
    return VariableSpec(
        key=key,
        title=title,
        label=label,
        source="hydro",
        mode="vector",
        group_u=group_u,
        group_v=group_v,
        group_mag=group_mag,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


VARIABLE_SPECS: Dict[str, VariableSpec] = {
    "curr": vector_spec(
        "curr",
        "Correntes",
        "Velocidade (m/s)",
        "velocity U",
        "velocity V",
        "velocity modulus",
        cmap="turbo",
        vmin=0.0,
        vmax=0.40,
    ),
    "wlev": hydro_scalar_spec(
        "wlev",
        "Nível do Mar",
        "Nível da água (m)",
        "water level",
        cmap="RdBu_r",
        vmin=-1.5,
        vmax=1.5,
        center_zero=True,
    ),
    "sali": scalar_spec("sali", "Salinidade", "Salinidade", "salinity", cmap="viridis", vmin=28.0, vmax=36.0),
    "temp": scalar_spec("temp", "Temperatura", "Temperatura (°C)", "temperature", cmap="Spectral_r", vmin=17.0, vmax=31.0),
    "oxy": scalar_spec("oxy", "Oxigênio Dissolvido", "Oxigênio dissolvido (mg/L)", "oxygen", cmap="YlGnBu", vmin=0.0, vmax=8.5),
    "t90": scalar_spec("t90", "T90", "T90", "T90", cmap="magma", vmin=0.0, vmax=216000.0),
    "ammo": scalar_spec("ammo", "Amônia", "Amônia", "ammonia", cmap="magma", vmin=0.0, vmax=0.10),
    "carb": scalar_spec("carb", "Dióxido de Carbono", "Dióxido de carbono", "carbon dioxide", cmap="cividis", vmin=0.0, vmax=1.0),
    "csed": scalar_spec("csed", "Sedimento Coesivo", "Sedimento coesivo", "cohesive sediment", cmap="gist_earth", vmin=0.0, vmax=2.0),
    "dnrn": scalar_spec("dnrn", "Nitrogênio Orgânico Dissolvido Não Refratário", "N orgânico dissolvido não refratário", "dissolved non-refractory organic nitrogen", cmap="YlGn", vmin=0.0, vmax=0.06),
    "dnrp": scalar_spec("dnrp", "Fósforo Orgânico Dissolvido Não Refratário", "P orgânico dissolvido não refratário", "dissolved non-refractory organic phosphorus", cmap="YlGnBu", vmin=0.0, vmax=0.010),
    "drrn": scalar_spec("drrn", "Nitrogênio Orgânico Dissolvido Refratário", "N orgânico dissolvido refratário", "dissolved refractory organic nitrogen", cmap="BuGn", vmin=0.0, vmax=0.030),
    "drrp": scalar_spec("drrp", "Fósforo Orgânico Dissolvido Refratário", "P orgânico dissolvido refratário", "dissolved refractory organic phosphorus", cmap="PuBuGn", vmin=0.0, vmax=0.003),
    "ecol": scalar_spec("ecol", "Escherichia coli", "Escherichia coli", "escherichia coli", cmap=COLIFORM_CMAP, vmin=1e-6, vmax=1e5, log_scale=True),
    "fcol": scalar_spec("fcol", "Coliformes Fecais", "Coliformes fecais", "fecal coliforms", cmap=COLIFORM_CMAP, vmin=1e-6, vmax=1e6, log_scale=True),
    "ipho": scalar_spec("ipho", "Fósforo Inorgânico", "Fósforo inorgânico", "inorganic phosphorus", cmap="PuBu", vmin=0.0, vmax=0.006),
    "nitr": scalar_spec("nitr", "Nitrato", "Nitrato", "nitrate", cmap="PuBuGn", vmin=0.0, vmax=0.030),
    "niti": scalar_spec("niti", "Nitrito", "Nitrito", "nitrite", cmap="BuPu", vmin=0.0, vmax=0.006),
    "pon": scalar_spec("pon", "Nitrogênio Orgânico Particulado", "N orgânico particulado", "particulate organic nitrogen", cmap="YlOrBr", vmin=0.0, vmax=0.020),
    "pop": scalar_spec("pop", "Fósforo Orgânico Particulado", "P orgânico particulado", "particulate organic phosphorus", cmap="OrRd", vmin=0.0, vmax=0.0015),
    "phyt": scalar_spec("phyt", "Fitoplâncton", "Fitoplâncton", "phytoplankton", cmap="YlGn", vmin=0.0, vmax=1.0),
    "zoop": scalar_spec("zoop", "Zooplâncton", "Zooplâncton", "zooplankton", cmap="PuBuGn", vmin=0.0, vmax=0.0004),
}


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


def choose_frame_index(times: Sequence[str], frame: Optional[int], time_str: Optional[str]) -> int:
    if not times:
        raise ValueError("Não há passos de tempo disponíveis no arquivo.")
    if time_str:
        normalized = time_str.strip()
        if normalized in times:
            return times.index(normalized)
        raise ValueError(
            f"Tempo '{time_str}' não encontrado no arquivo. "
            "Use --list-times para ver os instantes disponíveis."
        )
    if frame is None:
        return len(times) - 1
    return max(0, min(frame, len(times) - 1))


def print_times(times: Sequence[str]) -> None:
    print("Tempos disponíveis:")
    for idx, value in enumerate(times):
        print(f"  [{idx:02d}] {value}")


def sort_mohid_keys(keys: Iterable[str]) -> List[str]:
    def keyfun(name: str) -> Tuple[str, int]:
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], int(parts[1])
        digits = "".join(ch for ch in name if ch.isdigit())
        return name, int(digits) if digits else -1

    return sorted(keys, key=keyfun)


def get_time_strings(h5: h5py.File) -> List[str]:
    if "Time" not in h5:
        raise KeyError("Grupo /Time não encontrado no HDF5.")
    keys = sort_mohid_keys(h5["Time"].keys())
    return [parse_mohid_time(h5["Time"][k][()]) for k in keys]


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
    inset = ax.inset_axes([xpos, ypos, width, height], transform=ax.transAxes)
    inset.imshow(image)
    inset.set_axis_off()


def add_scale_bar(
    ax,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xpos: float = 0.73,
    ypos: float = 0.06,
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
        label = f"{bar_meters / 1000:.0f} km" if bar_meters >= 1000 else f"{bar_meters:.0f} m"
    else:
        target_units = 0.20 * xspan
        bar_dx = _nice_number(target_units)
        label = f"{bar_dx:g} units"

    x0 = xmin + xpos * xspan
    x0 = min(x0, xmax - bar_dx - 0.02 * xspan)
    y0 = ymin + ypos * yspan
    tick_h = 0.008 * yspan
    bar_h = 0.012 * yspan
    segment_count = 4
    segment_dx = bar_dx / segment_count

    for i in range(segment_count):
        xi = x0 + i * segment_dx
        face = "black" if i % 2 == 0 else "white"
        rect = plt.Rectangle((xi, y0), segment_dx, bar_h, facecolor=face, edgecolor="#202020", lw=0.7, zorder=6)
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
    if dep_max <= 0:
        return None

    if lon.ndim == 2 and lat.ndim == 2 and dep.ndim == 2:
        artist = ax.pcolormesh(
            lon,
            lat,
            np.ma.masked_invalid(dep),
            shading="auto",
            cmap=cmap,
            alpha=alpha,
            vmin=0,
            vmax=dep_max,
            zorder=1,
        )
        try:
            ax.contour(lon[:-1, :-1], lat[:-1, :-1], dep, levels=8, linewidths=0.4, alpha=0.4, zorder=1.5)
        except Exception:
            pass
        return artist
    return None


def _as_topography_list(topography) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if topography is None:
        return []
    if isinstance(topography, list):
        return topography
    return [topography]


def add_topography_to_axis(ax, topography, xmin: float, xmax: float, ymin: float, ymax: float, alpha=0.55):
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

    return artist


def resolve_layer_index(layer: int, nlayers: Optional[int]) -> Optional[int]:
    if nlayers is None:
        return None
    idx = layer if layer >= 0 else nlayers + layer
    if idx < 0 or idx >= nlayers:
        raise IndexError(f"Camada {layer} fora do intervalo válido 0..{nlayers - 1} (ou negativos equivalentes).")
    return idx


def infer_default_file(source: str, hydro_path: str, water_path: str) -> str:
    if source == "hydro":
        return hydro_path
    if source == "water":
        return water_path
    raise ValueError(f"Fonte desconhecida: {source}")


def find_group_name(results_group: h5py.Group, wanted: str) -> str:
    if wanted in results_group:
        return wanted
    low_map = {name.lower(): name for name in results_group.keys()}
    if wanted.lower() in low_map:
        return low_map[wanted.lower()]
    raise KeyError(f"Variável '{wanted}' não encontrada em /Results.")


def read_grid(hdf_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(hdf_path, "r") as h5:
        lon = np.asarray(h5["Grid/Longitude"][()], dtype=float)
        lat = np.asarray(h5["Grid/Latitude"][()], dtype=float)
        bathy = np.asarray(h5["Grid/Bathymetry"][()], dtype=float)
    return lon, lat, bathy


def read_water_points(hdf_path: str) -> Optional[np.ndarray]:
    with h5py.File(hdf_path, "r") as h5:
        if "Grid/WaterPoints3D" not in h5:
            return None
        return np.asarray(h5["Grid/WaterPoints3D"][()], dtype=float)


def read_scalar_series(hdf_path: str, group_name: str) -> Tuple[List[str], List[np.ndarray]]:
    with h5py.File(hdf_path, "r") as h5:
        times = get_time_strings(h5)
        group_name = find_group_name(h5["Results"], group_name)
        group = h5["Results"][group_name]
        keys = sort_mohid_keys(group.keys())
        data = [np.asarray(group[key][()], dtype=float) for key in keys]
    return times, data


def read_vector_series(
    hdf_path: str,
    group_u: str,
    group_v: str,
    group_mag: Optional[str] = None,
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    times, u_series = read_scalar_series(hdf_path, group_u)
    _, v_series = read_scalar_series(hdf_path, group_v)
    if len(u_series) != len(v_series):
        raise ValueError("Os grupos U e V possuem números diferentes de passos de tempo.")
    if group_mag is not None:
        _, mag_series = read_scalar_series(hdf_path, group_mag)
    else:
        mag_series = [np.hypot(u, v) for u, v in zip(u_series, v_series)]
    return times, u_series, v_series, mag_series


def select_layer(data: np.ndarray, layer_idx: Optional[int]) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if layer_idx is None:
            layer_idx = 0
        return arr[layer_idx]
    raise ValueError(f"Formato de dado não suportado: ndim={arr.ndim}")


def compute_limits(
    series: Sequence[np.ndarray],
    center_zero: bool = False,
    fixed_vmin: Optional[float] = None,
    fixed_vmax: Optional[float] = None,
) -> Tuple[float, float]:
    if fixed_vmin is not None and fixed_vmax is not None:
        return fixed_vmin, fixed_vmax

    vals = []
    for arr in series:
        finite = np.asarray(arr, dtype=float)
        finite = finite[np.isfinite(finite) & ~is_invalid(finite)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return 0.0, 1.0

    joined = np.concatenate(vals)
    vmin = float(np.nanpercentile(joined, 2))
    vmax = float(np.nanpercentile(joined, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(joined))
        vmax = float(np.nanmax(joined))
    if center_zero:
        vmax_abs = max(abs(vmin), abs(vmax), 1e-12)
        return -vmax_abs, vmax_abs
    if vmin == vmax:
        vmax = vmin + 1e-12
    return vmin, vmax


def extract_scalar_frames(series: Sequence[np.ndarray], layer: int) -> Tuple[List[np.ndarray], Optional[int]]:
    first = np.asarray(series[0])
    nlayers = first.shape[0] if first.ndim == 3 else None
    layer_idx = resolve_layer_index(layer, nlayers)
    frames = [np.where(is_invalid(select_layer(arr, layer_idx)), np.nan, select_layer(arr, layer_idx)) for arr in series]
    return frames, layer_idx


def select_water_mask(water_points: Optional[np.ndarray], layer_idx: Optional[int], target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if water_points is None:
        return None
    arr = np.asarray(water_points, dtype=float)
    if arr.ndim == 3:
        if layer_idx is None:
            return None
        mask = arr[layer_idx] > 0
    elif arr.ndim == 2:
        mask = arr > 0
    else:
        return None
    if mask.shape != target_shape:
        return None
    return mask


def apply_water_mask(field: np.ndarray, water_mask: Optional[np.ndarray]) -> np.ndarray:
    arr = np.asarray(field, dtype=float).copy()
    if water_mask is None:
        return arr
    arr[~water_mask] = np.nan
    return arr


def compute_quiver_scale(mag: np.ndarray, target_fraction: float = 0.10) -> float:
    finite = np.asarray(mag, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    ref = float(np.nanpercentile(finite, 95))
    if not np.isfinite(ref) or ref <= 0:
        ref = float(np.nanmax(finite))
    if not np.isfinite(ref) or ref <= 0:
        return 1.0
    return ref / max(target_fraction, 1e-6)


def grid_limits_from_grid(lon: np.ndarray, lat: np.ndarray) -> Tuple[float, float, float, float]:
    xx = lon[np.isfinite(lon)]
    yy = lat[np.isfinite(lat)]
    if xx.size == 0 or yy.size == 0:
        raise ValueError("A grade do modelo não possui coordenadas válidas para definir os limites do mapa.")
    xmin, xmax = float(np.nanmin(xx)), float(np.nanmax(xx))
    ymin, ymax = float(np.nanmin(yy)), float(np.nanmax(yy))
    return xmin, xmax, ymin, ymax


def create_norm(vmin: float, vmax: float, center_zero: bool, log_scale: bool = False):
    if log_scale:
        return LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, max(vmin, 1e-12) * 1.000001))
    if center_zero:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def resolve_cmap(cmap_spec):
    if isinstance(cmap_spec, str):
        return plt.get_cmap(cmap_spec).copy()
    try:
        return cmap_spec.copy()
    except Exception:
        return cmap_spec


def plot_scalar_frame(
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    times: List[str],
    frame_idx: int,
    spec: VariableSpec,
    layer_idx: Optional[int],
    vmin: float,
    vmax: float,
    bathy=None,
    topography=None,
    output: Optional[str] = None,
    dpi: int = 140,
):
    fig, ax = plt.subplots(figsize=(9, 7))
    xmin, xmax, ymin, ymax = grid_limits_from_grid(lon, lat)
    add_bathy_to_axis(ax, bathy)
    add_topography_to_axis(ax, topography, xmin, xmax, ymin, ymax)

    cmap = resolve_cmap(spec.cmap)
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    plot_field = np.asarray(field, dtype=float)
    if spec.log_scale:
        plot_field = np.where(plot_field > 0, plot_field, np.nan)
    norm = create_norm(vmin, vmax, spec.center_zero, log_scale=spec.log_scale)
    mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(plot_field), shading="auto", cmap=cmap, norm=norm, zorder=3)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    style_map_axes(ax)
    layer_text = "" if layer_idx is None else f" | camada {layer_idx}"
    ax.set_title(f"{spec.title}{layer_text}\n{times[frame_idx]}", fontsize=11, color="#2f2f2f", pad=8)
    ax.grid(True, alpha=0.18, color="#9aa4ad", linewidth=0.55)
    ax.set_aspect("equal" if abs((xmax - xmin) / max(ymax - ymin, 1e-12)) < 8 else "auto")
    add_north_arrow(ax)
    add_scale_bar(ax, xmin, xmax, ymin, ymax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.20)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(spec.label)

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_vector_frame(
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    mag: np.ndarray,
    times: List[str],
    frame_idx: int,
    spec: VariableSpec,
    layer_idx: Optional[int],
    vmin: float,
    vmax: float,
    quiver_step: int,
    water_mask: Optional[np.ndarray],
    bathy=None,
    topography=None,
    output: Optional[str] = None,
    dpi: int = 140,
):
    fig, ax = plt.subplots(figsize=(9, 7))
    xmin, xmax, ymin, ymax = grid_limits_from_grid(lon, lat)
    add_bathy_to_axis(ax, bathy)
    add_topography_to_axis(ax, topography, xmin, xmax, ymin, ymax)

    cmap = resolve_cmap(spec.cmap)
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    mesh = ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_invalid(mag),
        shading="auto",
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        zorder=3,
    )

    lonc = lon[:-1, :-1] if lon.shape != mag.shape else lon
    latc = lat[:-1, :-1] if lat.shape != mag.shape else lat
    good = np.isfinite(u) & np.isfinite(v) & np.isfinite(mag)
    if water_mask is not None and water_mask.shape == good.shape:
        good &= water_mask
    uq = np.where(good, u, np.nan)[::quiver_step, ::quiver_step]
    vq = np.where(good, v, np.nan)[::quiver_step, ::quiver_step]
    xq = lonc[::quiver_step, ::quiver_step]
    yq = latc[::quiver_step, ::quiver_step]
    qmag = np.where(good, mag, np.nan)[::quiver_step, ::quiver_step]
    qgood = np.isfinite(qmag) & (qmag > 0)
    quiver_scale = compute_quiver_scale(mag)
    ax.quiver(
        xq[qgood],
        yq[qgood],
        uq[qgood],
        vq[qgood],
        color="black",
        angles="uv",
        scale_units="width",
        scale=quiver_scale,
        width=0.0028,
        headwidth=3.8,
        headlength=5.0,
        headaxislength=4.5,
        pivot="middle",
        zorder=4,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    style_map_axes(ax)
    layer_text = "" if layer_idx is None else f" | camada {layer_idx}"
    ax.set_title(f"{spec.title}{layer_text}\n{times[frame_idx]}", fontsize=11, color="#2f2f2f", pad=8)
    ax.grid(True, alpha=0.18, color="#9aa4ad", linewidth=0.55)
    ax.set_aspect("equal" if abs((xmax - xmin) / max(ymax - ymin, 1e-12)) < 8 else "auto")
    add_north_arrow(ax)
    add_scale_bar(ax, xmin, xmax, ymin, ymax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.20)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(spec.label)

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_animation(render_frame, nframes: int, out_path: str, fps: int = 3):
    tmp_dir = Path("_tmp_mohid_variable_frames")
    tmp_dir.mkdir(exist_ok=True)
    frames = []
    for i in range(nframes):
        frame_path = tmp_dir / f"frame_{i:05d}.png"
        render_frame(i, str(frame_path))
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


def inspect_available_variables(hydro_path: str, water_path: str):
    print("Atalhos disponíveis:")
    for key, spec in VARIABLE_SPECS.items():
        source = hydro_path if spec.source == "hydro" else water_path
        target = spec.group if spec.group is not None else f"{spec.group_u} + {spec.group_v}"
        print(f"  --{key:5s} -> {target} [{source}]")

    for path in [hydro_path, water_path]:
        print(f"\nArquivo: {path}")
        with h5py.File(path, "r") as h5:
            times = get_time_strings(h5)
            print(f"  Passos de tempo: {len(times)}")
            if times:
                print(f"  Primeiro tempo : {times[0]}")
                print(f"  Último tempo   : {times[-1]}")
            print("  Variáveis em /Results:")
            for name in h5["Results"].keys():
                group = h5["Results"][name]
                first_key = sort_mohid_keys(group.keys())[0]
                ds = np.asarray(group[first_key][()])
                print(f"    - {name} :: shape={ds.shape}")


def _existing_candidates(args) -> List[str]:
    seen = set()
    candidates: List[str] = []
    for path in list(args.hdf5 or []) + [args.input, args.hydro, args.water]:
        if not path or path in seen:
            continue
        if Path(path).exists():
            candidates.append(path)
            seen.add(path)
    return candidates


def _file_has_groups(path: str, groups: Sequence[str]) -> bool:
    try:
        with h5py.File(path, "r") as h5:
            if "Results" not in h5:
                return False
            results = h5["Results"]
            for group in groups:
                find_group_name(results, group)
            return True
    except Exception:
        return False


def _pick_source_file(args, groups: Sequence[str], requested_name: str) -> str:
    for candidate in _existing_candidates(args):
        if _file_has_groups(candidate, groups):
            return candidate
    raise ValueError(
        f"Não encontrei um HDF5 compatível para '{requested_name}'. "
        "Informe o arquivo como argumento posicional, ou use --input, --hydro ou --water."
    )


def resolve_spec(args) -> Tuple[VariableSpec, str]:
    selected = [key for key in VARIABLE_SPECS if getattr(args, key)]
    if args.var:
        if selected:
            raise ValueError("Use um atalho como --temp ou uma variável genérica com --var, não ambos ao mesmo tempo.")
        source = _pick_source_file(args, [args.var], args.var)
        spec = VariableSpec(
            key="custom",
            title=args.var,
            label=args.var,
            source="custom",
            mode="scalar",
            group=args.var,
            cmap=args.cmap or "viridis",
            center_zero=args.center_zero,
        )
        return spec, source

    if len(selected) != 1:
        raise ValueError("Escolha exatamente uma variável por atalho, ou use --var para um nome genérico.")

    spec = VARIABLE_SPECS[selected[0]]
    required_groups = [spec.group] if spec.group is not None else [spec.group_u or "", spec.group_v or ""]
    source_path = _pick_source_file(args, required_groups, spec.title)
    return spec, source_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualização de campos hidrodinâmicos e de qualidade da água do MOHID")
    p.add_argument("hdf5", nargs="*", help="Um ou mais arquivos HDF5 a considerar na busca da variável solicitada")
    p.add_argument("--hydro", default="Hydrodynamic_2.hdf5", help="Arquivo HDF5 hidrodinâmico padrão")
    p.add_argument("--water", default="WaterProperties_2.hdf5", help="Arquivo HDF5 de propriedades da água padrão")
    p.add_argument("--input", help="Arquivo HDF5 explícito para uso com --var ou para sobrescrever o padrão do atalho")
    for key, spec in VARIABLE_SPECS.items():
        if spec.mode == "vector":
            help_text = f"Mapa de {spec.title.lower()} com setas e magnitude em cores"
        else:
            help_text = f"Mapa de {spec.title.lower()}"
        p.add_argument(f"--{key}", action="store_true", help=help_text)
    p.add_argument("--var", help="Nome de uma variável em /Results para plot sob demanda")
    p.add_argument("--layer", type=int, default=0, help="Índice da camada vertical para variáveis 3D (default: 0, superfície)")
    p.add_argument("--frame", type=int, default=None, help="Índice do passo de tempo a plotar")
    p.add_argument("--time", help='Momento exato a plotar no formato "AAAA-MM-DD HH:MM:SS"')
    p.add_argument("--list-times", action="store_true", help="Lista os tempos disponíveis com seus índices e sai")
    p.add_argument("--bathymetry", help="Arquivo opcional de batimetria (.nc, .csv, .xyz, .hdf5)")
    p.add_argument("--topography", nargs="+", help="Um ou mais rasters locais opcionais de altimetria/elevação terrestre")
    p.add_argument("--show", action="store_true", help="Exibe a figura na tela")
    p.add_argument("--save", help="Salva uma figura única em PNG")
    p.add_argument("--save-frames", help="Diretório para salvar todos os frames em PNG")
    p.add_argument("--animate", help="Salva animação .gif ou .mp4")
    p.add_argument("--fps", type=int, default=3, help="FPS da animação")
    p.add_argument("--inspect", action="store_true", help="Lista variáveis e estrutura resumida dos HDF5")
    p.add_argument("--quiver-step", type=int, default=12, help="Espaçamento das setas de corrente")
    p.add_argument("--cmap", help="Colormap customizado para uso com --var")
    p.add_argument("--center-zero", action="store_true", help="Centraliza a escala de cores em zero para uso com --var")
    return p


def main():
    args = build_parser().parse_args()

    if args.inspect:
        inspect_available_variables(args.hydro, args.water)
        return

    spec, hdf_path = resolve_spec(args)
    lon, lat, bathy_hdf = read_grid(hdf_path)
    water_points = read_water_points(hdf_path)
    bathy = (lon, lat, bathy_hdf)
    if args.bathymetry:
        bathy = read_bathy(args.bathymetry, fallback_hdf=None)
    topography = read_topography_rasters(args.topography) if args.topography else None

    if spec.mode == "scalar":
        times, raw_series = read_scalar_series(hdf_path, spec.group or "")
        if args.list_times:
            print_times(times)
            return
        frames, layer_idx = extract_scalar_frames(raw_series, args.layer)
        water_mask = select_water_mask(water_points, layer_idx, frames[0].shape)
        if water_mask is not None:
            frames = [apply_water_mask(frame, water_mask) for frame in frames]
        vmin, vmax = compute_limits(
            frames,
            center_zero=spec.center_zero,
            fixed_vmin=spec.vmin,
            fixed_vmax=spec.vmax,
        )

        def render_frame(i: int, output: Optional[str]):
            plot_scalar_frame(
                lon,
                lat,
                frames[i],
                times,
                i,
                spec,
                layer_idx,
                vmin,
                vmax,
                bathy=bathy,
                topography=topography,
                output=output,
            )

    else:
        times, raw_u, raw_v, raw_mag = read_vector_series(hdf_path, spec.group_u or "", spec.group_v or "", spec.group_mag)
        if args.list_times:
            print_times(times)
            return
        u_frames, layer_idx = extract_scalar_frames(raw_u, args.layer)
        v_frames, _ = extract_scalar_frames(raw_v, args.layer)
        mag_frames, _ = extract_scalar_frames(raw_mag, args.layer)
        water_mask = select_water_mask(water_points, layer_idx, mag_frames[0].shape)
        if water_mask is not None:
            u_frames = [apply_water_mask(frame, water_mask) for frame in u_frames]
            v_frames = [apply_water_mask(frame, water_mask) for frame in v_frames]
            mag_frames = [apply_water_mask(frame, water_mask) for frame in mag_frames]
        vmin, vmax = compute_limits(
            mag_frames,
            center_zero=False,
            fixed_vmin=spec.vmin,
            fixed_vmax=spec.vmax,
        )

        def render_frame(i: int, output: Optional[str]):
            plot_vector_frame(
                lon,
                lat,
                u_frames[i],
                v_frames[i],
                mag_frames[i],
                times,
                i,
                spec,
                layer_idx,
                vmin,
                vmax,
                max(args.quiver_step, 1),
                water_mask,
                bathy=bathy,
                topography=topography,
                output=output,
            )

    if args.save_frames:
        outdir = Path(args.save_frames)
        outdir.mkdir(parents=True, exist_ok=True)
        for i in range(len(times)):
            out = outdir / f"{spec.key}_{i:05d}.png"
            render_frame(i, str(out))
        print(f"Frames salvos em: {outdir}")

    if args.animate:
        save_animation(render_frame, len(times), args.animate, fps=args.fps)
        print(f"Animação salva em: {args.animate}")

    frame_idx = choose_frame_index(times, args.frame, args.time)

    if args.save:
        render_frame(frame_idx, args.save)
        print(f"Figura salva em: {args.save}")

    if args.show or (not args.save and not args.save_frames and not args.animate):
        render_frame(frame_idx, None)


if __name__ == "__main__":
    main()
