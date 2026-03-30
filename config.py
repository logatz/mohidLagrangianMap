from __future__ import annotations

from pathlib import Path

DEFAULT_HYDRO_FILE = "Hydrodynamic_2.hdf5"
DEFAULT_WATER_FILE = "WaterProperties_2.hdf5"

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
CARTOGRAPHY_ASSETS_DIR = ASSETS_DIR / "cartography"
TOPOGRAPHY_ASSETS_DIR = ASSETS_DIR / "topography"
NORTH_ARROW_PATH = CARTOGRAPHY_ASSETS_DIR / "NorthArrow.png"
