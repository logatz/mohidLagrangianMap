from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

from matplotlib.colors import LinearSegmentedColormap


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
    cmap: Any = "viridis"
    center_zero: bool = False
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    vcenter: Optional[float] = None
    log_scale: bool = False
    quiver_scale: Optional[float] = None


def scalar_spec(
    key: str,
    title: str,
    label: str,
    group: str,
    cmap: Any,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = None,
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
        vcenter=vcenter,
        log_scale=log_scale,
    )


def hydro_scalar_spec(
    key: str,
    title: str,
    label: str,
    group: str,
    cmap: Any,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vcenter: Optional[float] = None,
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
        vcenter=vcenter,
        log_scale=log_scale,
    )


def vector_spec(
    key: str,
    title: str,
    label: str,
    group_u: str,
    group_v: str,
    group_mag: str,
    cmap: Any,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    quiver_scale: Optional[float] = None,
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
        quiver_scale=quiver_scale,
    )


VARIABLE_SPECS: Dict[str, VariableSpec] = {
    "curr": vector_spec("curr", "Correntes", "Velocidade [m.s$^{-1}$]", "velocity U", "velocity V", "velocity modulus", cmap="turbo", vmin=0.0, vmax=0.80, quiver_scale=10),
    "wlev": hydro_scalar_spec("wlev", "Nível do Mar", "Nível da água [m]", "water level", cmap="RdBu_r", vmin=0.0, vmax=2.5, vcenter=1.25, center_zero=True),
    "sali": scalar_spec("sali", "Salinidade", "Salinidade", "salinity", cmap="YlGnBu_r", vmin=0, vmax=36.0),
    "temp": scalar_spec("temp", "Temperatura", "Temperatura [°C]", "temperature", cmap="Spectral_r", vmin=13.0, vmax=30.0),
    "oxy": scalar_spec("oxy", "Oxigênio Dissolvido", "Oxigênio dissolvido [mg.L$^{-1}$]", "oxygen", cmap="Spectral", vmin=0.0, vmax=9.0),
    "t90": scalar_spec("t90", "T90", "T90 [h]", "T90", cmap="magma", vmin=0.0, vmax=216000.0),
    "ammo": scalar_spec("ammo", "Amônia", "Amônia", "ammonia", cmap="magma", vmin=0.0, vmax=0.10),
    "carb": scalar_spec("carb", "Dióxido de Carbono", "Dióxido de carbono", "carbon dioxide", cmap="cividis", vmin=0.0, vmax=1.0),
    "csed": scalar_spec("csed", "Sedimento Coesivo", "Sedimento coesivo", "cohesive sediment", cmap="gist_earth", vmin=0.0, vmax=2.0),
    "dnrn": scalar_spec("dnrn", "Nitrogênio Orgânico Dissolvido Não Refratário", "N orgânico dissolvido não refratário", "dissolved non-refractory organic nitrogen", cmap="YlGn", vmin=0.0, vmax=0.06),
    "dnrp": scalar_spec("dnrp", "Fósforo Orgânico Dissolvido Não Refratário", "P orgânico dissolvido não refratário", "dissolved non-refractory organic phosphorus", cmap="YlGnBu", vmin=0.0, vmax=0.010),
    "drrn": scalar_spec("drrn", "Nitrogênio Orgânico Dissolvido Refratário", "N orgânico dissolvido refratário", "dissolved refractory organic nitrogen", cmap="BuGn", vmin=0.0, vmax=0.030),
    "drrp": scalar_spec("drrp", "Fósforo Orgânico Dissolvido Refratário", "P orgânico dissolvido refratário", "dissolved refractory organic phosphorus", cmap="PuBuGn", vmin=0.0, vmax=0.003),
    "ecol": scalar_spec("ecol", "Escherichia coli", "Escherichia coli [NMP]", "escherichia coli", cmap=COLIFORM_CMAP, vmin=1e-6, vmax=1e5, log_scale=True),
    "fcol": scalar_spec("fcol", "Coliformes Fecais", "Coliformes fecais [NMP]", "fecal coliforms", cmap=COLIFORM_CMAP, vmin=1e-6, vmax=1e6, log_scale=True),
    "ipho": scalar_spec("ipho", "Fósforo Inorgânico", "Fósforo inorgânico", "inorganic phosphorus", cmap="PuBu", vmin=0.0, vmax=0.006),
    "nitr": scalar_spec("nitr", "Nitrato", "Nitrato", "nitrate", cmap="PuBuGn", vmin=0.0, vmax=0.030),
    "niti": scalar_spec("niti", "Nitrito", "Nitrito", "nitrite", cmap="BuPu", vmin=0.0, vmax=0.006),
    "pon": scalar_spec("pon", "Nitrogênio Orgânico Particulado", "N orgânico particulado", "particulate organic nitrogen", cmap="YlOrBr", vmin=0.0, vmax=0.020),
    "pop": scalar_spec("pop", "Fósforo Orgânico Particulado", "P orgânico particulado", "particulate organic phosphorus", cmap="OrRd", vmin=0.0, vmax=0.0015),
    "phyt": scalar_spec("phyt", "Fitoplâncton", "Fitoplâncton [mg.m$^{-3}$]", "phytoplankton", cmap="YlGn", vmin=0.0, vmax=1.0),
    "zoop": scalar_spec("zoop", "Zooplâncton", "Zooplâncton", "zooplankton", cmap="PuBuGn", vmin=0.0, vmax=0.0004),
}