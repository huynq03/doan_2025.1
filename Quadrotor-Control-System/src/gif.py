import argparse
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image


FRAME_RE = re.compile(r"^frame_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)

# Skip these target indices when generating the GIF
EXCLUDE_TARGETS = {20, 160, 200, 240, 280}


def discover_frames(frame_dir: Path) -> dict[int, Path]:
    frames: dict[int, Path] = {}
    for p in frame_dir.iterdir():
        if not p.is_file():
            continue
        m = FRAME_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        frames[idx] = p
    return dict(sorted(frames.items(), key=lambda kv: kv[0]))


def pick_nearest(available: list[int], target: int) -> int:
    arr = np.asarray(available)
    return int(available[int(np.argmin(np.abs(arr - target)))])


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a GIF from animation frames")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=799)
    ap.add_argument("--step", type=int, default=40)
    ap.add_argument("--fps", type=float, default=10.0, help="Frames per second")
    ap.add_argument(
        "--frame_dir",
        type=str,
        default=None,
        help="Directory containing frame_XXXX.png (default: ../media/animation_frames)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output GIF path (default: ../media/animation_step40.gif)",
    )
    ap.add_argument(
        "--require_exact",
        action="store_true",
        help="If set: only keep exact indices; otherwise use nearest existing frame",
    )
    ap.add_argument(
        "--loop",
        type=int,
        default=0,
        help="GIF loop count (0 = infinite). PowerPoint-friendly default: 0",
    )
    args = ap.parse_args()

    src_dir = Path(__file__).resolve().parent
    project_dir = src_dir.parent
    frame_dir = Path(args.frame_dir).resolve() if args.frame_dir else (project_dir / "media" / "animation_frames")
    out_gif = Path(args.out).resolve() if args.out else (project_dir / "media" / f"animation_step{args.step}.gif")

    if not frame_dir.exists():
        raise SystemExit(f"Frame folder not found: {frame_dir}")

    frames = discover_frames(frame_dir)
    if not frames:
        raise SystemExit(f"No frames found in: {frame_dir}")

    available = list(frames.keys())
    targets = list(range(args.start, args.end + 1, args.step))
    if args.end not in targets:
        targets.append(args.end)
    targets = sorted(set(targets))
    targets = [t for t in targets if t not in EXCLUDE_TARGETS]

    chosen: list[Path] = []
    mapping: list[tuple[int, int]] = []
    last_picked: int | None = None
    for t in targets:
        if t in frames:
            picked = t
        else:
            if args.require_exact:
                continue
            picked = pick_nearest(available, t)

        if last_picked is not None and picked == last_picked:
            continue
        last_picked = picked

        chosen.append(frames[picked])
        mapping.append((t, picked))

    if not chosen:
        raise SystemExit("No frames selected. Check --start/--end/--step or remove --require_exact.")

    print(f"Frame dir: {frame_dir}")
    print(f"Available frames: {len(available)}")
    print(f"Selected frames: {len(chosen)}")
    print("Mapping (target -> picked):")
    for t, p in mapping:
        mark = "" if t == p else " (nearest)"
        print(f"  {t:>4} -> {p:>4}{mark}")

    # Load images and normalize to the same size (GIF requires consistent frame shape)
    base_size = None
    images: list[Image.Image] = []
    for i, p in enumerate(chosen):
        with Image.open(p) as im:
            im = im.convert("RGB")
            if base_size is None:
                base_size = im.size
            if im.size != base_size:
                im = im.resize(base_size, Image.Resampling.LANCZOS)
            images.append(im.copy())
    duration_ms = int(round(1000.0 / max(args.fps, 0.1)))

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = images
    first.save(
        str(out_gif),
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=args.loop,
        disposal=2,
    )
    print(f"\nĐã tạo GIF: {out_gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
