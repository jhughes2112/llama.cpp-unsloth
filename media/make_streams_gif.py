#!/usr/bin/env python3
"""Renders media/inkling2-streams.gif (and .mp4): decode speed of Inkling 2 Mini with 1, 2, 4 and 8 concurrent streams
as the context grows from 512 to 4K to 32K tokens. A bar is the total tok/sec, cut into one slice a stream.

Needs Pillow, numpy and ffmpeg. Usage: python media/make_streams_gif.py [ffmpeg]
The numbers: llama-server --parallel N --backend-sampling on an M5 Ultra, UD-Q4_K_M, temperature 1.0, each stream its
own real context of that many tokens, 256 tokens generated, two seeds (decode only; the prompts were cached first).
"""
import math, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1200, 676, 15
STREAMS = [1, 2, 4, 8]
CTX = ["512", "4K", "32K"]
# context -> (per stream, total) for 1, 2, 4, 8 streams
DATA = {"512": [(132, 132), (75, 151), (52, 207), (31, 253)],
        "4K":  [(125, 126), (72, 145), (49, 198), (30, 241)],
        "32K": [(109, 109), (62, 125), (41, 165), (24, 195)]}
HOLD, MOVE = 2.2, 0.9
SEQ = CTX + [CTX[0]]               # 512 -> 4K -> 32K -> 512
T_END = len(CTX) * (HOLD + MOVE)    # one loop
CY0, CY1, VMAX = 250, 590, 280.0
BX = [270, 490, 710, 930]           # bar centres
BW = 150
INK, DIM = (255, 255, 255), (176, 208, 255)
GOLD = (255, 214, 96)
SLICE = [(140, 228, 255), (110, 205, 255)]
TOP = (255, 214, 96)

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

F_TITLE, F_SUB, F_SMALL = font(60, True), font(34, True), font(22)
F_NUM, F_UNIT, F_NOTE, F_VAL = font(120, True), font(30, True), font(20), font(34, True)

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

def _phases():
    """how far the tokens of each bar have drifted at each frame: the integral of its per-stream speed, scaled so each
    bar makes a whole number of trips in a loop (seamless)"""
    n = int(round(T_END * FPS))
    cum = np.zeros((n + 1, 4))
    for f in range(n):
        cum[f + 1] = cum[f] + [state(f / FPS)[3][i][0] / 60.0 / FPS for i in range(4)]
    trips = np.maximum(np.round(cum[-1]), 1)
    return cum * (trips / cum[-1])

PH = None

def state(t):
    """values at time t: (context label shown, its blend toward the next, per-stream and totals)"""
    t = t % T_END
    seg = int(t // (HOLD + MOVE))
    u = t - seg * (HOLD + MOVE)
    a, b = SEQ[seg], SEQ[seg + 1]
    w = 0.0 if u < HOLD else (u - HOLD) / MOVE
    w = w * w * (3 - 2 * w)
    vals = [tuple(DATA[a][i][j] + (DATA[b][i][j] - DATA[a][i][j]) * w for j in (0, 1)) for i in range(4)]
    return a, b, w, vals

def frame(t):
    glow = Image.new("RGB", (W, H)); g = ImageDraw.Draw(glow)
    dim = Image.new("RGB", (W, H)); d = ImageDraw.Draw(dim)
    soft = Image.new("RGB", (W, H)); st = ImageDraw.Draw(soft)
    a, b, w, vals = state(t)

    for v in (100, 200):
        d.line([(150, py(v)), (1050, py(v))], fill=(60, 110, 220), width=1)
        d.text((138, py(v)), str(v), font=F_NOTE, fill=DIM, anchor="rm")
    d.line([(150, CY1), (1050, CY1)], fill=(120, 160, 240), width=2)

    for i, n in enumerate(STREAMS):
        per, tot = vals[i]
        x0, x1 = BX[i] - BW // 2, BX[i] + BW // 2
        top = py(tot)
        h = (CY1 - top) / n
        for s in range(n):   # one slice a stream, a hairline between them
            y1 = CY1 - s * h; y0 = y1 - h
            c = SLICE[s % 2]
            d.rectangle([x0, y0 + (1 if n > 1 else 0), x1, y1], fill=tuple(int(k * 0.55) for k in c))
            # tokens drift up through the slice, at the stream's speed
            k = max(2, int(BW / 26))
            for j in range(k):
                ph = (PH[int(round(t * FPS)) % (len(PH) - 1), i] + j * 0.618 + s * 0.37 + i * 0.13) % 1.0
                yy = y1 - 3 - ph * (h - 6)
                xx = x0 + 12 + (BW - 24) * ((j + 0.5) / k)
                r = 3 if n < 8 else 2
                d.ellipse([xx - r, yy - r, xx + r, yy + r], fill=c)
        g.line([(x0, top), (x1, top)], fill=TOP, width=4)
        g.text((BX[i], top - 12), "%d" % round(tot), font=F_VAL, fill=INK, anchor="ms")
        d.text((BX[i], CY1 + 16), "%d stream%s" % (n, "" if n == 1 else "s"), font=F_SMALL, fill=INK, anchor="ma")
        d.text((BX[i], CY1 + 44), "%d each" % round(per) if n > 1 else "", font=F_NOTE, fill=DIM, anchor="ma")

    st.text((70, 40), "Inkling 2 Mini", font=F_TITLE, fill=INK)
    st.text((72, 112), "concurrent streams", font=F_SUB, fill=GOLD)
    d.text((72, 160), "M5 Ultra  ·  llama-server  ·  UD-Q4_K_M  ·  temp 1.0", font=F_SMALL, fill=DIM)
    d.text((72, 186), "bar: total tok/sec, one slice a stream", font=F_SMALL, fill=DIM)

    # the context counter: the old label fades out, then the new one fades in
    for lab, al in ((a, max(0.0, 1.0 - 2*w)), (b, max(0.0, 2*w - 1.0))):
        if al <= 0.01:
            continue
        g.text((1120, 44), lab, font=F_NUM, fill=tuple(int(k * al) for k in GOLD), anchor="ra")
    g.text((1120, 172), "tokens of context", font=F_UNIT, fill=INK, anchor="ra")
    d.text((30, H - 16), "github.com/jhughes2112/llama.cpp-unsloth", font=F_NOTE, fill=DIM, anchor="ls")

    ga = np.asarray(glow, np.float32)
    bloom = (np.asarray(glow.filter(ImageFilter.GaussianBlur(6)), np.float32) * 0.7
             + np.asarray(glow.filter(ImageFilter.GaussianBlur(22)), np.float32) * 0.6)
    halo = np.asarray(soft.convert("L").filter(ImageFilter.GaussianBlur(12)), np.float32)[..., None] / 255
    out = BG * (1 - 0.55 * np.clip(halo * 1.6, 0, 1))
    out += bloom * 0.9 + np.asarray(soft.filter(ImageFilter.GaussianBlur(5)), np.float32) * 0.3
    for layer in (np.asarray(dim, np.float32), np.asarray(soft, np.float32), ga):
        al = layer.max(axis=2, keepdims=True) / 255
        out = out * (1 - al) + layer
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

def main():
    global PH
    PH = _phases()
    ffmpeg = sys.argv[1] if len(sys.argv) > 1 else "ffmpeg"
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        n = int(round(T_END * FPS))
        for i in range(n):
            frame(i / FPS).save(os.path.join(tmp, "f%04d.png" % i))
        frame(HOLD * 0.5).save(os.path.join(here, "inkling2-streams.png"))
        src = ["-framerate", str(FPS), "-i", os.path.join(tmp, "f%04d.png")]
        subprocess.run([ffmpeg, "-y", "-v", "error", *src, "-vf",
                        "scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                        "-loop", "0", os.path.join(here, "inkling2-streams.gif")], check=True)
        subprocess.run([ffmpeg, "-y", "-v", "error", *src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
                        "-movflags", "+faststart", os.path.join(here, "inkling2-streams.mp4")], check=True)

if __name__ == "__main__":
    main()
