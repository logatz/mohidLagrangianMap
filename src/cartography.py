from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

from config import NORTH_ARROW_PATH
from .io_mohid import is_invalid


LAND_CMAP = LinearSegmentedColormap.from_list(
    "soft_land",
    ["#f7f4ec", "#ece4d2", "#d8d4b3", "#bcc79e", "#8faa83"],
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
    image_path = NORTH_ARROW_PATH
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
            contour_lon = lon[:-1, :-1] if lon.shape != dep.shape else lon
            contour_lat = lat[:-1, :-1] if lat.shape != dep.shape else lat
            ax.contour(contour_lon, contour_lat, dep, levels=8, linewidths=0.4, alpha=0.4, zorder=1.5)
        except Exception:
            pass
        return artist

    if lon.ndim == 1 and lat.ndim == 1 and dep.ndim == 1:
        good = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(dep)
        if np.count_nonzero(good) < 3:
            return None
        levels = np.linspace(0, dep_max, 20) if dep_max > 0 else 20
        artist = ax.tricontourf(
            lon[good],
            lat[good],
            dep[good],
            levels=levels,
            cmap=cmap,
            alpha=alpha,
            vmin=0,
            vmax=dep_max,
            zorder=1,
        )
        try:
            ax.tricontour(lon[good], lat[good], dep[good], levels=8, linewidths=0.4, alpha=0.4, zorder=1.5)
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
            slope_scale = np.nanpercentile(slope, 95)
            if not np.isfinite(slope_scale) or slope_scale <= 0:
                slope_scale = 1.0
            shade = 1.0 - 0.35 * slope / slope_scale
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
            artist = ax.tricontourf(
                lon[good],
                lat[good],
                elev[good],
                levels=18,
                cmap=cmap,
                norm=norm,
                alpha=alpha,
                zorder=2,
            )

    return artist
