from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .animations import save_rendered_animation
from .domain import LagrangianRenderContext
from .io_geospatial import read_bathy, read_topography_rasters
from .lagrangian_processing import build_lagrangian_dataset, inspect_hdf
from .lagrangian_rendering import render_lagrangian_frame


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualização de partículas lagrangianas do MOHID")
    p.add_argument("hdf5", help="Arquivo HDF5 do resultado lagrangiano")
    p.add_argument("--bathymetry", help="Arquivo opcional de batimetria (.nc, .csv, .xyz, .hdf5)")
    p.add_argument("--topography", nargs="+", help="Um ou mais rasters locais opcionais de altimetria/elevação terrestre (ex.: GeoTIFF)")
    p.add_argument("--frame", type=int, default=None, help="Índice do passo de tempo a plotar")
    p.add_argument("--show", action="store_true", help="Exibe a figura na tela")
    p.add_argument("--save", help="Salva uma figura única em PNG")
    p.add_argument("--save-frames", help="Diretório para salvar todos os frames em PNG")
    p.add_argument("--animate", help="Salva animação .gif ou .mp4")
    p.add_argument("--fps", type=int, default=3, help="FPS da animação")
    p.add_argument("--inspect", action="store_true", help="Inspeciona a estrutura do HDF5 e sai")
    return p


def main(argv: Optional[Sequence[str]] = None):
    args = build_parser().parse_args(argv)

    if args.inspect:
        inspect_hdf(args.hdf5)
        return

    bathy = None
    topography = None
    try:
        bathy = read_bathy(args.bathymetry, fallback_hdf=args.hdf5 if args.bathymetry is None else None)
    except Exception as e:
        print(f"[aviso] Não foi possível carregar batimetria: {e}")
    if args.topography:
        try:
            topography = read_topography_rasters(args.topography)
        except Exception as e:
            print(f"[aviso] Não foi possível carregar altimetria: {e}")
    dataset = build_lagrangian_dataset(args.hdf5, bathy=bathy, topography=topography)
    context = LagrangianRenderContext()

    if args.save_frames:
        outdir = Path(args.save_frames)
        outdir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset.times)):
            out = outdir / f"particles_{i:05d}.png"
            render_lagrangian_frame(dataset, context, i, output=str(out))
        print(f"Frames salvos em: {outdir}")

    if args.animate:
        def render_frame(i: int, output: str):
            render_lagrangian_frame(dataset, context, i, output=output)

        save_rendered_animation(
            render_frame,
            len(dataset.times),
            args.animate,
            fps=args.fps,
            tmp_dir_name="_tmp_mohid_frames",
        )
        print(f"Animação salva em: {args.animate}")

    frame_idx = args.frame if args.frame is not None else len(dataset.times) - 1
    frame_idx = max(0, min(frame_idx, len(dataset.times) - 1))

    if args.save:
        render_lagrangian_frame(dataset, context, frame_idx, output=args.save)
        print(f"Figura salva em: {args.save}")

    if args.show or (not args.save and not args.save_frames and not args.animate):
        render_lagrangian_frame(dataset, context, frame_idx, output=None)
