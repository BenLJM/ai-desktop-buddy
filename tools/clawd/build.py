#!/usr/bin/env python3
"""
Build the Clawd character pack from Claude Code sticker PNGs.

Takes the 5 source stickers in tools/clawd/src/ and emits a full
character pack at characters/clawd/ (manifest + sleep/idle_*/busy/
attention/celebrate/dizzy/heart GIFs, each 96x100 P-mode).

Re-run after editing any source sticker or tweaking animation parameters
below. After regenerating, stage and flash with:

    python3 tools/flash_character.py characters/clawd
    # or, if `pio` isn't on PATH:
    python3 -m platformio run -t uploadfs --upload-port COM5

Source stickers are scaled nearest-neighbor (pixel art friendly) and
padded onto a 96x100 transparent canvas. Animations are generated
programmatically — see the state sections below.

Usage:
    python3 tools/clawd/build.py
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SRC_DIR = HERE / "src"
OUT_DIR = REPO / "characters" / "clawd"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_PATH = HERE / "preview.png"

CANVAS = (96, 100)  # matches the bufo reference pack
BG = (0, 0, 0)      # character background on device


def load_and_clean(path: Path) -> Image.Image:
    """Load sticker, strip the light-grey sticker-sheet background, crop."""
    im = Image.open(path).convert("RGBA")
    arr = np.array(im)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    drg = np.abs(r.astype(int) - g.astype(int))
    dgb = np.abs(g.astype(int) - b.astype(int))
    is_grey = (drg < 6) & (dgb < 6) & (r >= 200) & (r <= 250)
    arr[is_grey, 3] = 0
    out = Image.fromarray(arr, "RGBA")
    bbox = out.getbbox()
    return out.crop(bbox) if bbox else out


def fit_to_canvas(im: Image.Image, canvas=CANVAS, margin=3) -> Image.Image:
    """Nearest-neighbor scale to fit, center on transparent canvas."""
    w, h = canvas
    iw, ih = im.size
    scale = min((w - 2 * margin) / iw, (h - 2 * margin) / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = im.resize((nw, nh), Image.NEAREST)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(resized, ((w - nw) // 2, (h - nh) // 2), resized)
    return out


def composite_on_bg(rgba: Image.Image, bg=BG) -> Image.Image:
    base = Image.new("RGB", rgba.size, bg)
    base.paste(rgba, mask=rgba.split()[-1])
    return base


def to_p(rgb: Image.Image) -> Image.Image:
    return rgb.convert("P", palette=Image.ADAPTIVE, colors=32)


def save_gif(rgba_frames, path: Path, duration=120, loop=0):
    p_frames = [to_p(composite_on_bg(f)) for f in rgba_frames]
    p_frames[0].save(
        path,
        save_all=True,
        append_images=p_frames[1:],
        duration=duration,
        loop=loop,
        disposal=2,
        optimize=True,
    )
    print(f"  {path.name}: {len(p_frames)} frame(s)")


def shifted(rgba: Image.Image, dx=0, dy=0) -> Image.Image:
    out = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    out.paste(rgba, (dx, dy), rgba)
    return out


def rotated(rgba: Image.Image, angle: float) -> Image.Image:
    return rgba.rotate(angle, resample=Image.NEAREST, expand=False)


def main() -> None:
    print(f"reading stickers from {SRC_DIR}")
    srcs = {}
    for name in ["disk", "sparkle", "heart", "dizzy", "skate"]:
        cleaned = load_and_clean(SRC_DIR / f"clawd_{name}.png")
        srcs[name] = fit_to_canvas(cleaned)
        print(f"  clawd_{name}: {cleaned.size} -> {srcs[name].size}")

    base = srcs["heart"]   # clean body + heart, doubles as idle base

    print(f"\nwriting character pack to {OUT_DIR}")
    save_gif([srcs["dizzy"]], OUT_DIR / "sleep.gif", duration=500)
    save_gif(
        [rotated(srcs["dizzy"], a) for a in (-6, 0, 6, 0)],
        OUT_DIR / "dizzy.gif",
        duration=150,
    )
    save_gif(
        [shifted(srcs["disk"], 0, dy) for dy in (0, -2)],
        OUT_DIR / "busy.gif",
        duration=200,
    )
    save_gif(
        [shifted(srcs["sparkle"], dx, dy) for dx, dy in ((-1, 0), (0, -1), (1, 0), (0, 0))],
        OUT_DIR / "attention.gif",
        duration=150,
    )
    bobs = (0, 0, -1, -2, -2, -1, 0, 1, 1, 0, 0)
    save_gif([shifted(srcs["heart"], 0, b) for b in bobs], OUT_DIR / "heart.gif", duration=100)

    celebrate = [
        shifted(srcs["skate"], int(6 * math.sin(2 * math.pi * i / 27)), 0)
        for i in range(27)
    ]
    save_gif(celebrate, OUT_DIR / "celebrate.gif", duration=60)

    for i in range(9):
        src = base if i % 2 == 0 else srcs["skate"]
        frames = []
        for f in range(4):
            dy = int(2 * math.sin(2 * math.pi * f / 4 + i * 0.7))
            dx = int(1 * math.cos(2 * math.pi * f / 4 + i * 0.5))
            frames.append(shifted(src, dx, dy))
        save_gif(frames, OUT_DIR / f"idle_{i}.gif", duration=220)

    manifest = {
        "name": "clawd",
        "colors": {
            "body": "#D97853",
            "bg": "#000000",
            "text": "#FFFFFF",
            "textDim": "#808080",
            "ink": "#000000",
        },
        "states": {
            "sleep": "sleep.gif",
            "idle": [f"idle_{i}.gif" for i in range(9)],
            "busy": "busy.gif",
            "attention": "attention.gif",
            "celebrate": "celebrate.gif",
            "dizzy": "dizzy.gif",
            "heart": "heart.gif",
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  manifest.json")

    preview = Image.new("RGB", (96 * 7, 100), (40, 40, 40))
    labels = [
        ("idle", base),
        ("busy", srcs["disk"]),
        ("attn", srcs["sparkle"]),
        ("cele", srcs["skate"]),
        ("dizz", srcs["dizzy"]),
        ("hert", srcs["heart"]),
        ("slep", srcs["dizzy"]),
    ]
    for idx, (_, im) in enumerate(labels):
        preview.paste(composite_on_bg(im), (96 * idx, 0))
    preview.save(PREVIEW_PATH)

    total = sum(f.stat().st_size for f in OUT_DIR.iterdir() if f.is_file())
    cap = 1_800_000
    print(
        f"\ntotal: {total:,} bytes / {cap:,} cap ({100 * total / cap:.1f}%)\n"
        f"preview: {PREVIEW_PATH}"
    )


if __name__ == "__main__":
    main()
