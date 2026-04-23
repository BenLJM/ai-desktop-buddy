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


def load_rgba(path: Path) -> Image.Image:
    """Load frame, strip studio background to transparent, kill Veo watermark.

    Veo's watermark sits in the bottom-right corner — light grey "Veo" text
    that escapes the bg-grey detector and shows up as non-transparent
    content. Blanks out the bottom-right corner so bbox / union_bbox only
    tracks the actual sticker character.
    """
    im = Image.open(path).convert("RGBA")
    arr = np.array(im)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    drg = np.abs(r.astype(int) - g.astype(int))
    dgb = np.abs(g.astype(int) - b.astype(int))
    # Loosened upper bound to 255 to catch near-white watermark text too.
    is_bg = (drg < 12) & (dgb < 12) & (r >= 200)
    arr[is_bg, 3] = 0
    # Hard mask: watermark region at the bottom-right of a 1280x720 video.
    h, w = arr.shape[:2]
    mx, my = int(w * 0.85), int(h * 0.87)
    arr[my:, mx:, 3] = 0
    return Image.fromarray(arr, "RGBA")


def union_bbox(*paths: Path) -> tuple:
    """Compute the bbox that covers content in every listed frame.

    Each frame is cropped to its own bbox separately during quick
    preview, but the GIF pack needs a SHARED bbox — otherwise each
    frame's character gets scaled to fit 96x100 independently and the
    character visibly changes size between frames (the 'big-small
    jitter' the user reported).
    """
    left = top = None
    right = bottom = 0
    for p in paths:
        im = load_rgba(p)
        bb = im.getbbox()
        if bb is None:
            continue
        l, t, r, b = bb
        left   = l if left   is None else min(left,   l)
        top    = t if top    is None else min(top,    t)
        right  = max(right, r)
        bottom = max(bottom, b)
    return (left or 0, top or 0, right, bottom)


def fit_with_bbox(im: Image.Image, bbox, canvas=CANVAS, margin=2) -> Image.Image:
    """Crop to shared bbox and scale to canvas — constant scale across all frames."""
    w, h = canvas
    cropped = im.crop(bbox)
    cw, ch = cropped.size
    scale = min((w - 2 * margin) / cw, (h - 2 * margin) / ch)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized = cropped.resize((nw, nh), Image.NEAREST)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(resized, ((w - nw) // 2, (h - nh) // 2), resized)
    return out


# Legacy single-frame wrapper — still used for the static dizzy sticker.
def load_and_clean(path: Path) -> Image.Image:
    im = load_rgba(path)
    bbox = im.getbbox()
    if bbox:
        im = im.crop(bbox)
    return im


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


def frame_paths(start, end, step=4):
    return [FRAMES / f"f_{i:03d}.png"
            for i in range(start, end + 1, step)
            if (FRAMES / f"f_{i:03d}.png").exists()]


def frames_in_range(start, end, step=4, bbox=None):
    """Load frames cropped to a shared bbox (so character stays same size)."""
    out = []
    for p in frame_paths(start, end, step):
        rgba = load_rgba(p)
        if bbox is not None:
            out.append(fit_with_bbox(rgba, bbox))
        else:
            # Legacy self-cropping fallback
            bb = rgba.getbbox()
            if bb:
                rgba = rgba.crop(bb)
            out.append(fit_to_canvas(rgba))
    return out


def main():
    extract_frames()

    # Each state gets its OWN shared bbox — the union of every frame
    # inside that state's window. That fixes within-state jitter
    # (different poses no longer scale differently) without throwing
    # away the accessory in states with a big accessory. A single
    # whole-pack bbox had to choose: tight around the body (bulb /
    # heart / rocket got clipped), or wide enough for accessories
    # (body shrank to ~40 px, unusably small). Per-state bboxes let
    # each animation fill its own canvas naturally.
    def bbox_for(ranges):
        paths = []
        for start, end, step in ranges:
            paths += frame_paths(start, end, step)
        return union_bbox(*paths)

    celebrate_bbox = bbox_for([(4, 44, 3)])
    heart_bbox     = bbox_for([(65, 98, 3)])
    attention_bbox = bbox_for([(112, 142, 3)])
    busy_bbox      = bbox_for([(162, 188, 3)])
    rest_bbox      = bbox_for([(54, 58, 1), (150, 158, 1)])

    for name, bb in [("celebrate", celebrate_bbox), ("heart", heart_bbox),
                     ("attention", attention_bbox), ("busy", busy_bbox),
                     ("rest", rest_bbox)]:
        print(f"{name} bbox: {bb} size={bb[2]-bb[0]}x{bb[3]-bb[1]}")

    # Durations chosen to match ESP32 decode budget (~150 ms/frame).
    celebrate = frames_in_range(4, 44, step=3, bbox=celebrate_bbox)
    save_gif(celebrate, OUT_DIR / "celebrate.gif", duration=140)

    heart = frames_in_range(65, 98, step=3, bbox=heart_bbox)
    save_gif(heart, OUT_DIR / "heart.gif", duration=140)

    attention = frames_in_range(112, 142, step=3, bbox=attention_bbox)
    save_gif(attention, OUT_DIR / "attention.gif", duration=140)

    busy = frames_in_range(162, 188, step=3, bbox=busy_bbox)
    save_gif(busy, OUT_DIR / "busy.gif", duration=140)

    rest = (frames_in_range(54, 58, step=1, bbox=rest_bbox) +
            frames_in_range(150, 158, step=1, bbox=rest_bbox))
    if not rest:
        rest = frames_in_range(150, 156, step=1, bbox=rest_bbox)
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
