"""Build the Clawd character pack v2 from the bundled animation clip.

Source: tools/clawd/src/clawd_animation.mp4 — a Veo-generated 8-second,
24 fps, 1280x720 pixel-art loop that cycles through four action poses
interleaved with short return-to-rest windows:

    1..48    rocket / firework       -> celebrate
    49..100  heart on the left       -> heart
    108..144 idea bulb above head    -> attention
    162..188 happy ^_^ spin / walk   -> busy + idle motion
    48..60 and 146..156              -> clean body, base idle

Outputs: characters/clawd/*.gif + manifest.json (written in-place).

Frames are extracted to a scratch dir next to the script via the
imageio-ffmpeg binary (pip install imageio-ffmpeg). Run with:

    python tools/clawd/build_v2.py

Also consumes tools/clawd/src/clawd_dizzy.png for dizzy/sleep states,
which have no motion in the video.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


HERE     = Path(__file__).resolve().parent         # tools/clawd
REPO     = HERE.parent.parent                      # repo root
SRC_DIR  = HERE / "src"
VIDEO    = SRC_DIR / "clawd_animation.mp4"
FRAMES   = HERE / "_frames_scratch"                # ephemeral
OUT_DIR  = REPO / "characters" / "clawd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS = (96, 100)
BG = (0, 0, 0)


def ffmpeg_bin() -> str:
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "ffmpeg not found. Install system ffmpeg or `pip install imageio-ffmpeg`."
        )


def extract_frames():
    if FRAMES.exists():
        shutil.rmtree(FRAMES)
    FRAMES.mkdir()
    subprocess.run(
        [ffmpeg_bin(), "-y", "-i", str(VIDEO), "-vf", "fps=24",
         str(FRAMES / "f_%03d.png")],
        check=True, capture_output=True,
    )


def load_and_clean(path: Path) -> Image.Image:
    """Load, make the light-grey studio background transparent, crop to content."""
    im = Image.open(path).convert("RGBA")
    arr = np.array(im)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    drg = np.abs(r.astype(int) - g.astype(int))
    dgb = np.abs(g.astype(int) - b.astype(int))
    is_grey = (drg < 10) & (dgb < 10) & (r >= 210) & (r <= 250)
    arr[is_grey, 3] = 0
    out = Image.fromarray(arr, "RGBA")
    bbox = out.getbbox()
    if bbox:
        out = out.crop(bbox)
    return out


def fit_to_canvas(im: Image.Image, canvas=CANVAS, margin=2) -> Image.Image:
    w, h = canvas
    iw, ih = im.size
    scale = min((w - 2 * margin) / iw, (h - 2 * margin) / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = im.resize((nw, nh), Image.NEAREST)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(resized, ((w - nw) // 2, (h - nh) // 2), resized)
    return out


def composite_on_bg(rgba, bg=BG):
    base = Image.new("RGB", rgba.size, bg)
    base.paste(rgba, mask=rgba.split()[-1])
    return base


def to_p(im_rgb):
    return im_rgb.convert("P", palette=Image.ADAPTIVE, colors=64)


def save_gif(rgba_frames, path, duration=120, loop=0):
    p_frames = [to_p(composite_on_bg(f)) for f in rgba_frames]
    p_frames[0].save(
        path, save_all=True, append_images=p_frames[1:],
        duration=duration, loop=loop, disposal=2, optimize=True,
    )
    print(f"{path.name}: {len(p_frames)} fr")


def frames_in_range(start, end, step=4):
    out = []
    for i in range(start, end + 1, step):
        p = FRAMES / f"f_{i:03d}.png"
        if p.exists():
            out.append(fit_to_canvas(load_and_clean(p)))
    return out


def main():
    extract_frames()

    celebrate = frames_in_range(4, 42, step=4)
    save_gif(celebrate, OUT_DIR / "celebrate.gif", duration=90)

    heart = frames_in_range(65, 98, step=4)
    save_gif(heart, OUT_DIR / "heart.gif", duration=100)

    attention = frames_in_range(112, 142, step=4)
    save_gif(attention, OUT_DIR / "attention.gif", duration=110)

    busy = frames_in_range(162, 188, step=4)
    save_gif(busy, OUT_DIR / "busy.gif", duration=95)

    rest = frames_in_range(48, 60, step=2) + frames_in_range(146, 156, step=2)
    if not rest:
        rest = frames_in_range(150, 156, step=1)
    for i in range(9):
        offset = i % len(rest)
        idle_i = [rest[(offset + j) % len(rest)] for j in range(4)]
        save_gif(idle_i, OUT_DIR / f"idle_{i}.gif", duration=160)

    # Dizzy & sleep have no motion in the video; use the static sticker.
    dizzy_src = fit_to_canvas(load_and_clean(SRC_DIR / "clawd_dizzy.png"))
    dizzy = [dizzy_src.rotate(a, resample=Image.NEAREST, expand=False)
             for a in [-6, 0, 6, 0]]
    save_gif(dizzy, OUT_DIR / "dizzy.gif", duration=150)
    save_gif([dizzy_src], OUT_DIR / "sleep.gif", duration=500)

    manifest = {
        "name": "clawd",
        "colors": {
            "body": "#D97853", "bg": "#000000", "text": "#FFFFFF",
            "textDim": "#808080", "ink": "#000000",
        },
        "states": {
            "sleep":     "sleep.gif",
            "idle":      [f"idle_{i}.gif" for i in range(9)],
            "busy":      "busy.gif",
            "attention": "attention.gif",
            "celebrate": "celebrate.gif",
            "dizzy":     "dizzy.gif",
            "heart":     "heart.gif",
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum(p.stat().st_size for p in OUT_DIR.iterdir() if p.is_file())
    print(f"total pack: {total/1024:.1f} KB (limit 1800 KB)")

    shutil.rmtree(FRAMES, ignore_errors=True)


if __name__ == "__main__":
    main()
