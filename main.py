#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from src.fields_cli import main as fields_main
from src.lagrangian_cli import main as lagrangian_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI unificada para mapas MOHID")
    subparsers = parser.add_subparsers(dest="command")

    p_fields = subparsers.add_parser("fields", help="Mapas de campos hidrodinâmicos e variáveis da água")
    p_fields.add_argument("args", nargs=argparse.REMAINDER, help="Argumentos repassados para o módulo de fields")

    p_lagrangian = subparsers.add_parser("lagrangian", help="Mapas de partículas lagrangianas")
    p_lagrangian.add_argument("args", nargs=argparse.REMAINDER, help="Argumentos repassados para o módulo lagrangiano")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not raw_args:
        parser.print_help()
        return

    command, *command_args = raw_args
    if command == "fields":
        fields_main(command_args)
        return
    if command == "lagrangian":
        lagrangian_main(command_args)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
