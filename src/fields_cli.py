from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np

from config import DEFAULT_HYDRO_FILE, DEFAULT_WATER_FILE
from .animations import save_rendered_animation
from .domain import FieldRenderContext
from .fields_processing import (
    build_scalar_dataset,
    build_vector_dataset,
    compute_limits,
    find_group_name,
)
from .fields_rendering import render_scalar_dataset_frame, render_vector_dataset_frame
from .io_geospatial import read_bathy, read_topography_rasters
from .io_mohid import choose_frame_index, get_time_strings, print_times, read_grid, read_water_points, sort_mohid_keys
from .specs import VARIABLE_SPECS, VariableSpec


def inspect_available_variables(hydro_path: str, water_path: str):
    print("Atalhos disponíveis:")
    for key, spec in VARIABLE_SPECS.items():
        source = hydro_path if spec.source == "hydro" else water_path
        target = spec.group if spec.group is not None else f"{spec.group_u} + {spec.group_v}"
        print(f"  --{key:5s} -> {target} [{source}]")

    for path in [hydro_path, water_path]:
        print(f"\nArquivo: {path}")
        with h5py.File(path, "r") as h5:
            times = get_time_strings(h5)
            print(f"  Passos de tempo: {len(times)}")
            if times:
                print(f"  Primeiro tempo : {times[0]}")
                print(f"  Último tempo   : {times[-1]}")
            print("  Variáveis em /Results:")
            for name in h5["Results"].keys():
                group = h5["Results"][name]
                first_key = sort_mohid_keys(group.keys())[0]
                ds = np.asarray(group[first_key][()])
                print(f"    - {name} :: shape={ds.shape}")


def _existing_candidates(args) -> List[str]:
    seen = set()
    candidates: List[str] = []
    for path in list(args.hdf5 or []) + [args.input, args.hydro, args.water]:
        if not path or path in seen:
            continue
        if Path(path).exists():
            candidates.append(path)
            seen.add(path)
    return candidates


def _file_has_groups(path: str, groups: Sequence[str]) -> bool:
    try:
        with h5py.File(path, "r") as h5:
            if "Results" not in h5:
                return False
            results = h5["Results"]
            for group in groups:
                find_group_name(results, group)
            return True
    except Exception:
        return False


def _pick_source_file(args, groups: Sequence[str], requested_name: str) -> str:
    for candidate in _existing_candidates(args):
        if _file_has_groups(candidate, groups):
            return candidate
    raise ValueError(
        f"Não encontrei um HDF5 compatível para '{requested_name}'. "
        "Informe o arquivo como argumento posicional, ou use --input, --hydro ou --water."
    )


def resolve_spec(args) -> Tuple[VariableSpec, str]:
    selected = [key for key in VARIABLE_SPECS if getattr(args, key)]
    if args.var:
        if selected:
            raise ValueError("Use um atalho como --temp ou uma variável genérica com --var, não ambos ao mesmo tempo.")
        source = _pick_source_file(args, [args.var], args.var)
        spec = VariableSpec(
            key="custom",
            title=args.var,
            label=args.var,
            source="custom",
            mode="scalar",
            group=args.var,
            cmap=args.cmap or "viridis",
            center_zero=args.center_zero,
        )
        return spec, source

    if len(selected) != 1:
        raise ValueError("Escolha exatamente uma variável por atalho, ou use --var para um nome genérico.")

    spec = VARIABLE_SPECS[selected[0]]
    required_groups = [spec.group] if spec.group is not None else [spec.group_u or "", spec.group_v or ""]
    source_path = _pick_source_file(args, required_groups, spec.title)
    return spec, source_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualização de campos hidrodinâmicos e de qualidade da água do MOHID")
    p.add_argument("hdf5", nargs="*", help="Um ou mais arquivos HDF5 a considerar na busca da variável solicitada")
    p.add_argument("--hydro", default=DEFAULT_HYDRO_FILE, help="Arquivo HDF5 hidrodinâmico padrão")
    p.add_argument("--water", default=DEFAULT_WATER_FILE, help="Arquivo HDF5 de propriedades da água padrão")
    p.add_argument("--input", help="Arquivo HDF5 explícito para uso com --var ou para sobrescrever o padrão do atalho")
    for key, spec in VARIABLE_SPECS.items():
        if spec.mode == "vector":
            help_text = f"Mapa de {spec.title.lower()} com setas e magnitude em cores"
        else:
            help_text = f"Mapa de {spec.title.lower()}"
        p.add_argument(f"--{key}", action="store_true", help=help_text)
    p.add_argument("--var", help="Nome de uma variável em /Results para plot sob demanda")
    p.add_argument("--layer", type=int, default=0, help="Índice da camada vertical para variáveis 3D (default: 0, superfície)")
    p.add_argument("--frame", type=int, default=None, help="Índice do passo de tempo a plotar")
    p.add_argument("--time", help='Momento exato a plotar no formato "AAAA-MM-DD HH:MM:SS"')
    p.add_argument("--list-times", action="store_true", help="Lista os tempos disponíveis com seus índices e sai")
    p.add_argument("--bathymetry", help="Arquivo opcional de batimetria (.nc, .csv, .xyz, .hdf5)")
    p.add_argument("--topography", nargs="+", help="Um ou mais rasters locais opcionais de altimetria/elevação terrestre")
    p.add_argument("--show", action="store_true", help="Exibe a figura na tela")
    p.add_argument("--save", help="Salva uma figura única em PNG")
    p.add_argument("--save-frames", help="Diretório para salvar todos os frames em PNG")
    p.add_argument("--animate", help="Salva animação .gif ou .mp4")
    p.add_argument("--fps", type=int, default=3, help="FPS da animação")
    p.add_argument("--inspect", action="store_true", help="Lista variáveis e estrutura resumida dos HDF5")
    p.add_argument("--quiver-step", type=int, default=12, help="Espaçamento das setas de corrente")
    p.add_argument("--quiver-scale", type=float, help="Escala fixa das setas de corrente; sobrescreve o padrão da variável")
    p.add_argument("--cmap", help="Colormap customizado para uso com --var")
    p.add_argument("--center-zero", action="store_true", help="Centraliza a escala de cores em zero para uso com --var")
    return p


def main(argv: Optional[Sequence[str]] = None):
    args = build_parser().parse_args(argv)

    if args.inspect:
        inspect_available_variables(args.hydro, args.water)
        return

    spec, hdf_path = resolve_spec(args)
    lon, lat, bathy_hdf = read_grid(hdf_path)
    water_points = read_water_points(hdf_path)
    bathy = (lon, lat, bathy_hdf)
    if args.bathymetry:
        bathy = read_bathy(args.bathymetry, fallback_hdf=None)
    topography = read_topography_rasters(args.topography) if args.topography else None

    if spec.mode == "scalar":
        dataset = build_scalar_dataset(hdf_path, spec, lon, lat, bathy, water_points, args.layer)
        if args.list_times:
            print_times(dataset.times)
            return
        vmin, vmax = compute_limits(
            dataset.scalar_frames or [],
            center_zero=spec.center_zero,
            fixed_vmin=spec.vmin,
            fixed_vmax=spec.vmax,
        )
        context = FieldRenderContext(topography=topography, vmin=vmin, vmax=vmax, dpi=140)

        def render_frame(i: int, output: Optional[str]):
            render_scalar_dataset_frame(dataset, context, i, output=output)

    else:
        dataset = build_vector_dataset(hdf_path, spec, lon, lat, bathy, water_points, args.layer)
        if args.list_times:
            print_times(dataset.times)
            return
        vmin, vmax = compute_limits(
            dataset.mag_frames or [],
            center_zero=False,
            fixed_vmin=spec.vmin,
            fixed_vmax=spec.vmax,
        )
        context = FieldRenderContext(
            topography=topography,
            vmin=vmin,
            vmax=vmax,
            quiver_step=max(args.quiver_step, 1),
            quiver_scale_override=args.quiver_scale,
            dpi=140,
        )

        def render_frame(i: int, output: Optional[str]):
            render_vector_dataset_frame(dataset, context, i, output=output)

    if args.save_frames:
        outdir = Path(args.save_frames)
        outdir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset.times)):
            out = outdir / f"{spec.key}_{i:05d}.png"
            render_frame(i, str(out))
        print(f"Frames salvos em: {outdir}")

    if args.animate:
        save_rendered_animation(
            render_frame,
            len(dataset.times),
            args.animate,
            fps=args.fps,
            tmp_dir_name="_tmp_mohid_variable_frames",
        )
        print(f"Animação salva em: {args.animate}")

    frame_idx = choose_frame_index(dataset.times, args.frame, args.time)

    if args.save:
        render_frame(frame_idx, args.save)
        print(f"Figura salva em: {args.save}")

    if args.show or (not args.save and not args.save_frames and not args.animate):
        render_frame(frame_idx, None)
