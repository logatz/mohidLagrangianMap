from __future__ import annotations

from pathlib import Path
from typing import Callable

import imageio.v2 as imageio


def save_rendered_animation(
    render_frame: Callable[[int, str], None],
    nframes: int,
    out_path: str,
    fps: int = 3,
    tmp_dir_name: str = "_tmp_mohid_frames",
) -> None:
    tmp_dir = Path(tmp_dir_name)
    tmp_dir.mkdir(exist_ok=True)
    frames = []
    for i in range(nframes):
        frame_path = tmp_dir / f"frame_{i:05d}.png"
        render_frame(i, str(frame_path))
        frames.append(imageio.imread(frame_path))

    out = Path(out_path)
    if out.suffix.lower() == ".gif":
        imageio.mimsave(out, frames, fps=fps, loop=0)
    elif out.suffix.lower() in [".mp4", ".m4v"]:
        imageio.mimsave(out, frames, fps=fps)
    else:
        raise ValueError("Use .gif ou .mp4 para animação.")

    for p in tmp_dir.glob("frame_*.png"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass
