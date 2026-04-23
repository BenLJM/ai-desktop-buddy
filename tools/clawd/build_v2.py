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


def save_gif(rgba_frames, path, duration=120, loop=0):
    """Write a multi-frame GIF the ESP32 AnimatedGIF decoder plays cleanly.

    The on-device decoder doesn't clear previous-frame pixels when PIL
    (or imageio) emits partial-patch frames — it crops each frame to
    the diff bounding box, so new poses stack on top of old ones.

    Force every frame to cover the full 96x100 canvas by:
      1. Reserving palette index 0 = pure black (bg) and index 1 =
         (1,1,1) — visually identical to black on the LCD.
      2. Alternating pixel (0,0) and (W-1,H-1) between those two
         indices per frame. PIL's diff bbox is forced to include both
         corners, which spans the entire canvas.
      3. Saving with disposal=1 (same as bufo) — at that point every
         frame is a full replacement so the disposal choice is moot.
    """
    rgb_frames = [composite_on_bg(f) for f in rgba_frames]

    # Shared palette across all frames so the decoder doesn't switch
    # local color tables mid-loop (causes flicker on ESP32).
    if len(rgb_frames) > 1:
        fw, fh = rgb_frames[0].size
        concat = Image.new("RGB", (fw * len(rgb_frames), fh))
        for i, f in enumerate(rgb_frames):
            concat.paste(f, (i * fw, 0))
        ref = concat.convert("P", palette=Image.ADAPTIVE, colors=62)
    else:
        ref = rgb_frames[0].convert("P", palette=Image.ADAPTIVE, colors=62)

    # Prepend two near-black entries so indices 0 and 1 are reserved for
    # the anti-ghost corner markers. The quantized frames had referenced
    # palette indices starting at 0; shift them up by 2 before writing.
    src_pal = ref.getpalette()[: 3 * 62]       # 62 colors * 3 channels
    new_pal = [0, 0, 0, 1, 1, 1] + src_pal     # 64 colors
    # Pad to 256 * 3 so Pillow accepts it as a full GIF palette
    new_pal = new_pal + [0] * (768 - len(new_pal))

    p_frames = []
    for rgb in rgb_frames:
        # Quantize against `ref` to get indices in [0..61]
        q = rgb.quantize(palette=ref, dither=Image.NONE)
        # Shift indices up by 2 so they align with the new palette
        data = bytes(b + 2 for b in q.tobytes())
        shifted = Image.frombytes("P", q.size, data)
        shifted.putpalette(new_pal)
        p_frames.append(shifted)

    # Alternate corner markers per frame so consecutive frames differ at
    # both (0,0) and (W-1,H-1), forcing PIL's diff bbox to full canvas.
    for i, f in enumerate(p_frames):
        marker = i % 2          # 0 or 1, both map to (near-)black in new_pal
        f.putpixel((0, 0), marker)
        f.putpixel((f.width - 1, f.height - 1), marker)

    p_frames[0].save(
        path,
        save_all=True,
        append_images=p_frames[1:],
        duration=duration,
        loop=loop,
        disposal=1,
        optimize=False,
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

    # ESP32 AnimatedGIF + full-frame 96x100 + SPI LCD + LittleFS I/O
    # leaves ~150ms/frame realistic budget. Past experience: too-fast
    # durations (60-90ms) cause visible jitter as the decoder falls
    # behind and catches up unevenly. 140ms per frame gives stable ~7
    # fps playback; source is 24 fps so picking every 3rd frame (step=3)
    # gives 8 fps capture, matched to display budget.
    celebrate = frames_in_range(4, 44, step=3)        # ~14 fr
    save_gif(celebrate, OUT_DIR / "celebrate.gif", duration=140)

    heart = frames_in_range(65, 98, step=3)           # ~12 fr
    save_gif(heart, OUT_DIR / "heart.gif", duration=140)

    attention = frames_in_range(112, 142, step=3)     # ~11 fr
    save_gif(attention, OUT_DIR / "attention.gif", duration=140)

    busy = frames_in_range(162, 188, step=3)          # ~9 fr
    save_gif(busy, OUT_DIR / "busy.gif", duration=140)

    rest = frames_in_range(48, 60, step=2) + frames_in_range(146, 156, step=2)
    if not rest:
        rest = frames_in_range(150, 156, step=1)
    for i in range(9):
        offset = (i * 2) % len(rest)
        idle_i = [rest[(offset + j) % len(rest)] for j in range(5)]
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
