from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

DEFAULT_HYDRO_FILE = "Hydrodynamic_2.hdf5"
DEFAULT_WATER_FILE = "WaterProperties_2.hdf5"
DEFAULT_FLORIANOPOLIS_DATA_ROOT = Path("/home/garbossa/mnt/qnap/saida/mohid/Florianopolis")
DEFAULT_SC_PR_SP_DATA_ROOT = Path("/home/garbossa/mnt/qnap/saida/mohid/SC_PR_SP")
DEFAULT_DOMAIN = "bisc"


@dataclass(frozen=True)
class DomainConfig:
    key: str
    label: str
    data_root: Path
    quiver_step: int = 12
    quiver_scale: Optional[float] = None
    figsize: Tuple[float, float] = (9.0, 7.0)
    dpi: int = 140


DOMAIN_CONFIGS: Dict[str, DomainConfig] = {
    "florianopolis": DomainConfig(
        key="florianopolis",
        label="BISC/SCIB - Baía da Ilha de Santa Catarina",
        data_root=DEFAULT_FLORIANOPOLIS_DATA_ROOT,
        quiver_step=12,
        quiver_scale=None,
        figsize=(9.0, 7.0),
        dpi=140,
    ),
    "sc_pr_sp": DomainConfig(
        key="sc_pr_sp",
        label="SC/PR/SP",
        data_root=DEFAULT_SC_PR_SP_DATA_ROOT,
        quiver_step=14,
        quiver_scale=None,
        figsize=(8.2, 9.2),
        dpi=140,
    ),
}

DOMAIN_ALIASES: Dict[str, str] = {
    "bisc": "florianopolis",
    "scib": "florianopolis",
    "sc": "sc_pr_sp",
}

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
CARTOGRAPHY_ASSETS_DIR = ASSETS_DIR / "cartography"
TOPOGRAPHY_ASSETS_DIR = ASSETS_DIR / "topography"
NORTH_ARROW_PATH = CARTOGRAPHY_ASSETS_DIR / "NorthArrow.png"
