from __future__ import annotations

from typing import Dict, List, Tuple

import h5py
import numpy as np

from .domain import LagrangianDataset
from .io_mohid import get_time_strings, is_invalid, parse_mohid_time, sort_mohid_keys


def candidate_series_groups(h5: h5py.File) -> Dict[str, List[str]]:
    found = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            lname = name.lower()
            if any(tok in lname for tok in ["longitude", "latitude", "coord", "position", "x ", "y ", "/x", "/y"]):
                found[name] = list(obj.keys())

    h5.visititems(visitor)
    return found


def find_coordinate_groups(h5: h5py.File) -> Tuple[str, str]:
    groups = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            lname = name.lower()
            if any(tok in lname for tok in ["longitude", "latitude", "x coord", "y coord", "position x", "position y", "/x", "/y"]):
                groups.append(name)

    h5.visititems(visitor)

    by_parent = {}
    for g in groups:
        parent = g.rsplit("/", 1)[0] if "/" in g else ""
        by_parent.setdefault(parent, []).append(g)

    pairs = []
    for _, gs in by_parent.items():
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
        la = a.lower()
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


def build_lagrangian_dataset(
    hdf_path: str,
    bathy=None,
    topography=None,
) -> LagrangianDataset:
    times, X, Y = read_particle_tracks(hdf_path)
    return LagrangianDataset(
        hdf_path=hdf_path,
        times=times,
        X=X,
        Y=Y,
        bathy=bathy,
        topography=topography,
    )
