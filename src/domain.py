from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .specs import VariableSpec

GridTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]
TopoList = Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]


@dataclass
class FieldDataset:
    hdf_path: str
    spec: VariableSpec
    lon: np.ndarray
    lat: np.ndarray
    bathy: Optional[GridTuple]
    water_points: Optional[np.ndarray]
    times: List[str]
    mode: str
    layer_idx: Optional[int]
    scalar_frames: Optional[List[np.ndarray]] = None
    u_frames: Optional[List[np.ndarray]] = None
    v_frames: Optional[List[np.ndarray]] = None
    mag_frames: Optional[List[np.ndarray]] = None
    water_mask: Optional[np.ndarray] = None


@dataclass
class FieldRenderContext:
    topography: TopoList = None
    vmin: float = 0.0
    vmax: float = 1.0
    quiver_step: int = 12
    quiver_scale_override: Optional[float] = None
    dpi: int = 140
    figsize: Tuple[float, float] = (9.0, 7.0)


@dataclass
class LagrangianDataset:
    hdf_path: str
    times: List[str]
    X: np.ndarray
    Y: np.ndarray
    bathy: Optional[GridTuple] = None
    topography: TopoList = None


@dataclass
class LagrangianRenderContext:
    title_prefix: str = "Partículas Lagrangianas"
    dpi: int = 140
