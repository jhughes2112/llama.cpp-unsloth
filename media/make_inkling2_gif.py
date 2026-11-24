#!/usr/bin/env python3
"""Renders media/inkling2-mac.gif (and .mp4): decode speed of Inkling 2 Mini on this branch, commit by commit.

Needs Pillow, numpy and ffmpeg. Usage: python media/make_inkling2_gif.py [ffmpeg]
The numbers are the llama-bench decode results recorded in the commit messages (M5 Ultra, UD-Q4_K_XL); the last one is
llama-server on a coding prompt at temperature 1.0 with the DFlash2 draft model.
"""
import os, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1200, 676, 25
T_DRAW0, T_DRAW1, T_END = 0.6, 6.0, 9.6
PTS = [37.0, 39.8, 42.5, 43.4, 50.1, 53.6, 55, 60, 62.6, 68, 71.5, 75.5, 80, 85.8, 88, 99, 103, 104, 108, 110, 112, 157]
NOTES = {0: "Unsloth release", 5: "fused kernels", 10: "graph ordered by depth", 13: "barrier fix",
         15: "commands pre-encoded", 20: "112 without a draft", 21: "DFlash2 draft, on code"}
CX0, CX1, CY0, CY1, VMAX = 90, 1110, 318, 590, 170.0
INK, DIM, LINE = (255, 255, 255), (176, 208, 255), (140, 228, 255)
GOLD = (255, 214, 96)
MID = ((255, 255, 255), (255, 240, 200))
COOL, HOT = ((225, 246, 255), (110, 205, 255)), ((255, 244, 150), (255, 138, 44))   # counter, top and bottom

def font(size, bold=False):
    path = "/System/Library/Fonts/HelveticaNeue.ttc"
    want = "Bold" if bold else "Regular"
    for i in range(16):
        try:
            f = ImageFont.truetype(path, size, index=i)
        except OSError:
            break
        if f.getname()[1] == want:
            return f
    return ImageFont.truetype(path, size)

F_TITLE, F_SUB, F_SMALL = font(84, True), font(50, True), font(22)
F_NUM, F_UNIT, F_NOTE, F_BADGE = font(176, True), font(40, True), font(20), font(34, True)

def px(i):
    return CX0 + (CX1 - CX0) * i / (len(PTS) - 1)

def py(v):
    return CY1 - (CY1 - CY0) * v / VMAX

def background():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    t = y / H
    top, bot = np.array([8, 30, 120], np.float32), np.array([14, 78, 214], np.float32)
    img = top + (bot - top) * t[..., None]
    r = np.hypot((x - W * 0.72) / W, (y - H * 0.30) / H)
    img += np.clip(1 - r / 0.75, 0, 1)[..., None] ** 2 * np.array([10, 40, 70], np.float32)
    return img

BG = background()

def frame(t):
    glow = Image.new("RGB", (W, H))   # everything that blooms
    g = ImageDraw.Draw(glow)
    dim = Image.new("RGB", (W, H))    # crisp only
    soft = Image.new("RGB", (W, H))   # title: a dark halo and little bloom, to stay readable
    st = ImageDraw.Draw(soft)
    d = ImageDraw.Draw(dim)

    u = min(max((t - T_DRAW0) / (T_DRAW1 - T_DRAW0), 0.0), 1.0)
    u = u * u * (3 - 2 * u)
    pos = u * (len(PTS) - 1)
    k = min(int(pos), len(PTS) - 2)
    val = PTS[k] + (PTS[k + 1] - PTS[k]) * (pos - k)
    head = (px(pos), py(val))

    for v in (40, 80, 120, 160):
        d.line([(CX0, py(v)), (CX1, py(v))], fill=(60, 110, 220), width=1)
        d.text((CX0 - 12, py(v)), str(v), font=F_NOTE, fill=DIM, anchor="rm")
    d.line([(CX0, CY1), (CX1, CY1)], fill=(120, 160, 240), width=2)
    d.text((CX0, CY1 + 14), "one stream; last point: code, temperature 1.0",
           font=F_SMALL, fill=DIM)
    d.text((CX1, CY1 + 14), "github.com/jhughes2112/llama.cpp-unsloth", font=F_SMALL, fill=DIM, anchor="ra")

    line = [(px(i), py(PTS[i])) for i in range(k + 1)] + [head]
    area = Image.new("L", (W, H))
    ImageDraw.Draw(area).polygon(line + [(head[0], CY1), (CX0, CY1)], fill=255)
    fade = np.clip((np.arange(H, dtype=np.float32)[:, None] - CY0) / (CY1 - CY0), 0, 1)
    area = np.asarray(area, np.float32) / 255 * (0.38 - 0.30 * fade)
    if len(line) > 1:
        g.line(line, fill=LINE, width=5, joint="curve")
    for i in range(k + 1):
        g.ellipse([px(i) - 4, py(PTS[i]) - 4, px(i) + 4, py(PTS[i]) + 4], fill=INK)
    g.ellipse([head[0] - 9, head[1] - 9, head[0] + 9, head[1] + 9], fill=INK)
    for i, s in NOTES.items():
        last = i >= len(PTS) - 2  # the two labels at the right edge sit left of their points
        if pos >= i or (i == len(PTS) - 1 and pos > i - 1):
            a = min(max(pos - (i - 1), 0.0), 1.0) if i == len(PTS) - 1 else min((pos - i) / 0.6, 1.0) if i else 1.0
            fill = tuple(int(c * a) for c in INK)
            if last:
                d.text((px(i) - 16, py(PTS[i]) + (-12 if i == len(PTS) - 1 else -30)), s, font=F_NOTE, fill=fill, anchor="ra")
            else:
                d.text((px(i) + 12, py(PTS[i]) + 16), s, font=F_NOTE, fill=fill)

    st.text((70, 40), "Inkling 2 Mini", font=F_TITLE, fill=INK)
    st.text((72, 138), "on a ", font=F_SUB, fill=(200, 222, 255))
    st.text((72 + F_SUB.getlength("on a "), 138), "Mac", font=F_SUB, fill=GOLD)
    d.text((72, 204), "M5 Ultra  ·  llama.cpp + Metal  ·  UD-Q4_K_XL", font=F_SMALL, fill=DIM)
    heat = (val - PTS[0]) / (PTS[-1] - PTS[0])   # the counter warms from ice blue to gold over orange
    lo, hi, h = (COOL, MID, heat * 2) if heat < 0.5 else (MID, HOT, heat * 2 - 1)   # through white, not through grey
    top = np.array(lo[0], np.float32) + (np.array(hi[0], np.float32) - lo[0]) * h
    bot = np.array(lo[1], np.float32) + (np.array(hi[1], np.float32) - lo[1]) * h
    ramp = np.clip((np.arange(H, dtype=np.float32) - 70) / 135, 0, 1)[:, None, None]
    ramp = np.broadcast_to(top + (bot - top) * ramp, (H, W, 3)).astype(np.uint8)
    mask = Image.new("L", (W, H))
    ImageDraw.Draw(mask).text((936, 44), str(int(round(val))), font=F_NUM, fill=255, anchor="ra")
    glow.paste(Image.fromarray(ramp), mask=mask)
    g.text((946, 132), "tok/sec", font=F_UNIT, fill=INK)
    b = min(max((t - T_DRAW1 - 0.1) / 0.5, 0.0), 1.0)
    if b > 0:
        g.text((946, 76), "4× on code", font=F_BADGE, fill=tuple(int(c * b) for c in (130, 255, 190)))

    pulse = 1.0 + 0.5 * np.exp(-((t - T_DRAW1 - 0.25) / 0.35) ** 2)
    ga = np.asarray(glow, np.float32)
    bloom = (np.asarray(glow.filter(ImageFilter.GaussianBlur(7)), np.float32) * 0.75
             + np.asarray(glow.filter(ImageFilter.GaussianBlur(26)), np.float32) * 0.85) * pulse
    halo = np.asarray(soft.convert("L").filter(ImageFilter.GaussianBlur(12)), np.float32)[..., None] / 255
    out = (BG + area[..., None] * np.array(LINE, np.float32)) * (1 - 0.55 * np.clip(halo * 1.6, 0, 1))
    out += bloom * 0.9 + np.asarray(soft.filter(ImageFilter.GaussianBlur(5)), np.float32) * 0.3
    for layer in (np.asarray(dim, np.float32), np.asarray(soft, np.float32), ga):
        a = layer.max(axis=2, keepdims=True) / 255
        out = out * (1 - a) + layer
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

def main():
    ffmpeg = sys.argv[1] if len(sys.argv) > 1 else "ffmpeg"
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        n = int(T_END * FPS)
        for i in range(n):
            frame(i / FPS).save(os.path.join(tmp, "f%04d.png" % i))
        frame(T_END).save(os.path.join(here, "inkling2-mac.png"))
        src = ["-framerate", str(FPS), "-i", os.path.join(tmp, "f%04d.png")]
        subprocess.run([ffmpeg, "-y", "-v", "error", *src, "-vf",
                        "split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                        "-loop", "0", os.path.join(here, "inkling2-mac.gif")], check=True)
        subprocess.run([ffmpeg, "-y", "-v", "error", *src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                        "-movflags", "+faststart", os.path.join(here, "inkling2-mac.mp4")], check=True)

if __name__ == "__main__":
    main()
