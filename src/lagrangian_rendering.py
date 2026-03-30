from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from .cartography import add_bathy_to_axis, add_north_arrow, add_scale_bar, add_topography_to_axis, style_map_axes
from .domain import LagrangianDataset, LagrangianRenderContext
from .lagrangian_processing import nice_limits


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
    title_prefix: str = "Partículas Lagrangianas",
):
    nsteps, npart = X.shape
    fig, ax = plt.subplots(figsize=(9, 7))
    xmin, xmax, ymin, ymax = nice_limits(X, Y)
    xmin, xmax, ymin, ymax = _expand_limits_with_bathy((xmin, xmax, ymin, ymax), bathy)
    bathy_artist = add_bathy_to_axis(ax, bathy)
    add_topography_to_axis(ax, topography, xmin, xmax, ymin, ymax)
    norm = Normalize(vmin=0, vmax=max(nsteps - 1, 1))

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
    )

    if output:
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def render_lagrangian_frame(
    dataset: LagrangianDataset,
    context: LagrangianRenderContext,
    frame_idx: int,
    output: Optional[str] = None,
):
    return plot_frame(
        dataset.times,
        dataset.X,
        dataset.Y,
        frame_idx,
        bathy=dataset.bathy,
        topography=dataset.topography,
        output=output,
        title_prefix=context.title_prefix,
        dpi=context.dpi,
    )
