from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple, List

import h5py
import numpy as np


INVALID_LIMIT = 1e15
DEFAULT_BAD_VALUES = (-9.9e15, -9999999.0, -99999.0)


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


def get_time_strings(h5: h5py.File) -> List[str]:
    if "Time" not in h5:
        raise KeyError("Grupo /Time não encontrado no HDF5.")
    keys = sort_mohid_keys(h5["Time"].keys())
    return [parse_mohid_time(h5["Time"][k][()]) for k in keys]


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


def pick_first_existing(d: Dict[str, np.ndarray], names: Sequence[str]) -> Optional[np.ndarray]:
    for name in names:
        if name in d:
            return d[name]
    low_map = {k.lower(): k for k in d.keys()}
    for name in names:
        if name.lower() in low_map:
            return d[low_map[name.lower()]]
    return None


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
