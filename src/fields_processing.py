from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np

from .domain import FieldDataset
from .io_mohid import get_time_strings, is_invalid, sort_mohid_keys


def resolve_layer_index(layer: int, nlayers: Optional[int]) -> Optional[int]:
    if nlayers is None:
        return None
    idx = layer if layer >= 0 else nlayers + layer
    if idx < 0 or idx >= nlayers:
        raise IndexError(f"Camada {layer} fora do intervalo válido 0..{nlayers - 1} (ou negativos equivalentes).")
    return idx


def find_group_name(results_group: h5py.Group, wanted: str) -> str:
    if wanted in results_group:
        return wanted
    low_map = {name.lower(): name for name in results_group.keys()}
    if wanted.lower() in low_map:
        return low_map[wanted.lower()]
    raise KeyError(f"Variável '{wanted}' não encontrada em /Results.")


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


def build_scalar_dataset(
    hdf_path: str,
    spec,
    lon: np.ndarray,
    lat: np.ndarray,
    bathy,
    water_points: Optional[np.ndarray],
    layer: int,
) -> FieldDataset:
    times, raw_series = read_scalar_series(hdf_path, spec.group or "")
    frames, layer_idx = extract_scalar_frames(raw_series, layer)
    water_mask = select_water_mask(water_points, layer_idx, frames[0].shape)
    if water_mask is not None:
        frames = [apply_water_mask(frame, water_mask) for frame in frames]
    return FieldDataset(
        hdf_path=hdf_path,
        spec=spec,
        lon=lon,
        lat=lat,
        bathy=bathy,
        water_points=water_points,
        times=times,
        mode="scalar",
        layer_idx=layer_idx,
        scalar_frames=frames,
        water_mask=water_mask,
    )


def build_vector_dataset(
    hdf_path: str,
    spec,
    lon: np.ndarray,
    lat: np.ndarray,
    bathy,
    water_points: Optional[np.ndarray],
    layer: int,
) -> FieldDataset:
    times, raw_u, raw_v, raw_mag = read_vector_series(hdf_path, spec.group_u or "", spec.group_v or "", spec.group_mag)
    u_frames, layer_idx = extract_scalar_frames(raw_u, layer)
    v_frames, _ = extract_scalar_frames(raw_v, layer)
    mag_frames, _ = extract_scalar_frames(raw_mag, layer)
    water_mask = select_water_mask(water_points, layer_idx, mag_frames[0].shape)
    if water_mask is not None:
        u_frames = [apply_water_mask(frame, water_mask) for frame in u_frames]
        v_frames = [apply_water_mask(frame, water_mask) for frame in v_frames]
        mag_frames = [apply_water_mask(frame, water_mask) for frame in mag_frames]
    return FieldDataset(
        hdf_path=hdf_path,
        spec=spec,
        lon=lon,
        lat=lat,
        bathy=bathy,
        water_points=water_points,
        times=times,
        mode="vector",
        layer_idx=layer_idx,
        u_frames=u_frames,
        v_frames=v_frames,
        mag_frames=mag_frames,
        water_mask=water_mask,
    )
