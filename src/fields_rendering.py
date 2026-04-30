from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from .cartography import add_bathy_to_axis, add_north_arrow, add_scale_bar, add_topography_to_axis, style_map_axes
from .domain import FieldDataset, FieldRenderContext
from .fields_processing import compute_quiver_scale
from .specs import VariableSpec


def grid_limits_from_grid(lon: np.ndarray, lat: np.ndarray) -> Tuple[float, float, float, float]:
    xx = lon[np.isfinite(lon)]
    yy = lat[np.isfinite(lat)]
    if xx.size == 0 or yy.size == 0:
        raise ValueError("A grade do modelo não possui coordenadas válidas para definir os limites do mapa.")
    xmin, xmax = float(np.nanmin(xx)), float(np.nanmax(xx))
    ymin, ymax = float(np.nanmin(yy)), float(np.nanmax(yy))
    return xmin, xmax, ymin, ymax


def create_norm(
    vmin: float,
    vmax: float,
    center_zero: bool,
    log_scale: bool = False,
    vcenter: Optional[float] = None,
):
    if log_scale:
        return LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, max(vmin, 1e-12) * 1.000001))
    if center_zero:
        center_value = 0.0 if vcenter is None else vcenter
        return TwoSlopeNorm(vmin=vmin, vcenter=center_value, vmax=vmax)
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
    figsize: Tuple[float, float] = (9.0, 7.0),
):
    fig, ax = plt.subplots(figsize=figsize)
    xmin, xmax, ymin, ymax = grid_limits_from_grid(lon, lat)
    add_bathy_to_axis(ax, bathy)
    add_topography_to_axis(ax, topography, xmin, xmax, ymin, ymax)

    cmap = resolve_cmap(spec.cmap)
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    plot_field = np.asarray(field, dtype=float)
    if spec.log_scale:
        plot_field = np.where(plot_field > 0, plot_field, np.nan)
    norm = create_norm(vmin, vmax, spec.center_zero, log_scale=spec.log_scale, vcenter=spec.vcenter)
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
    quiver_scale_override: Optional[float],
    water_mask: Optional[np.ndarray],
    bathy=None,
    topography=None,
    output: Optional[str] = None,
    dpi: int = 140,
    figsize: Tuple[float, float] = (9.0, 7.0),
):
    fig, ax = plt.subplots(figsize=figsize)
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
    quiver_scale = (
        quiver_scale_override
        if quiver_scale_override is not None
        else spec.quiver_scale
        if spec.quiver_scale is not None
        else compute_quiver_scale(mag)
    )
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


def render_scalar_dataset_frame(
    dataset: FieldDataset,
    context: FieldRenderContext,
    frame_idx: int,
    output: Optional[str] = None,
):
    if dataset.scalar_frames is None:
        raise ValueError("FieldDataset escalar sem frames escalares.")
    return plot_scalar_frame(
        dataset.lon,
        dataset.lat,
        dataset.scalar_frames[frame_idx],
        dataset.times,
        frame_idx,
        dataset.spec,
        dataset.layer_idx,
        context.vmin,
        context.vmax,
        bathy=dataset.bathy,
        topography=context.topography,
        output=output,
        dpi=context.dpi,
        figsize=context.figsize,
    )


def render_vector_dataset_frame(
    dataset: FieldDataset,
    context: FieldRenderContext,
    frame_idx: int,
    output: Optional[str] = None,
):
    if dataset.u_frames is None or dataset.v_frames is None or dataset.mag_frames is None:
        raise ValueError("FieldDataset vetorial sem frames U/V/magnitude.")
    return plot_vector_frame(
        dataset.lon,
        dataset.lat,
        dataset.u_frames[frame_idx],
        dataset.v_frames[frame_idx],
        dataset.mag_frames[frame_idx],
        dataset.times,
        frame_idx,
        dataset.spec,
        dataset.layer_idx,
        context.vmin,
        context.vmax,
        context.quiver_step,
        context.quiver_scale_override,
        dataset.water_mask,
        bathy=dataset.bathy,
        topography=context.topography,
        output=output,
        dpi=context.dpi,
        figsize=context.figsize,
    )
