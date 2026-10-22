"""Procedural animated scenes for The Dreamstate.

Pure numpy + Pillow, no network, no stock/AI images. Every frame is drawn from code, so there is nothing
downloaded and nothing watermarked. Each scene is a short seamless loop (rain, river, forest, ocean,
aurora, lanterns) in the Dreamstate palette; build_video() loops them under the voiceover.

A scene is `Scene(w, h, seed, sleep).frame(t) -> float32 HxWx3 in 0..1` where t is time in seconds.
frame() is continuous in t (and is also valid for negative t), which is what makes the loop crossfade seamless.
"""
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PAL = {"navy": (16, 21, 47), "blue": (32, 44, 91), "indigo": (53, 54, 111), "purple": (98, 81, 143),
       "lavender": (184, 166, 223), "lilac": (217, 200, 238), "pink": (233, 187, 216),
       "ivory": (255, 240, 217), "gold": (233, 206, 154)}
C = {k: np.array(v, np.float32) / 255 for k, v in PAL.items()}

FPS = 24
W, H = 1920, 1080          # final video size
RW, RH = 960, 540          # scenes are drawn at half size, then upscaled (soft, dreamy, 4x cheaper)
LOOP, XFADE = 24, 6        # loop length and seamless-loop crossfade, in seconds


# ----------------------------------------------------------------------------- helpers
def tnoise(w, h, rng, sx=8, sy=8, beta=1.6):
    """Tileable (seamlessly wrapping) smooth noise in 0..1. sx/sy = rough feature count across width/height."""
    fy = np.fft.fftfreq(h)[:, None] * h / sy
    fx = np.fft.rfftfreq(w)[None, :] * w / sx
    r = np.maximum(np.sqrt(fx ** 2 + fy ** 2), 1.0)
    amp = 1.0 / r ** beta
    amp[0, 0] = 0
    spec = (rng.normal(size=amp.shape) + 1j * rng.normal(size=amp.shape)) * amp
    n = np.fft.irfft2(spec, s=(h, w)).astype(np.float32)
    lo, hi = np.percentile(n, 1), np.percentile(n, 99)
    return np.clip((n - lo) / max(hi - lo, 1e-6), 0, 1)


def shift_x(a, dx):
    """Sub-pixel horizontal wrap-around shift (works on HxW and HxWx3)."""
    i = int(np.floor(dx)); f = np.float32(dx - i)
    return np.roll(a, -i, 1) * (1 - f) + np.roll(a, -i - 1, 1) * f


def row_shift(a, shifts):
    """Shift every row of `a` sideways by its own (sub-pixel) amount, wrapping at the edges."""
    h, w = a.shape[:2]
    i = np.floor(shifts).astype(np.int32)
    f = (shifts - i).astype(np.float32)[:, None]
    if a.ndim == 3:
        f = f[..., None]
    r = np.arange(h)[:, None]; xs = np.arange(w)[None, :] + i[:, None]
    return a[r, xs % w] * (1 - f) + a[r, (xs + 1) % w] * f


def gradient(h, stops):
    ys = np.linspace(0, 1, h, dtype=np.float32)
    pos = [p for p, _ in stops]
    out = np.stack([np.interp(ys, pos, [c[k] for _, c in stops]) for k in range(3)], 1)
    return out[:, None, :].astype(np.float32)


def glow(w, h, cx, cy, r, col, strength, power=2.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    return (np.clip(1 - d, 0, 1) ** power * strength)[..., None] * col[None, None, :]


def ridge(w, rng, base_y, amp, rough=0.5):
    xs = np.linspace(0, 1, w, dtype=np.float32)
    y = np.zeros(w, np.float32); a = 1.0
    for k in range(7):
        f = 1.6 * 2 ** k; ph = rng.random() * 6.283
        y += np.sin(xs * f * 6.283 + ph) * a * (rng.random() * 0.5 + 0.75); a *= rough
    y = (y - y.min()) / max(y.max() - y.min(), 1e-6)
    return base_y - y * amp


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Fireflies:
    def __init__(self, rng, n, w, h, y0, y1, k=1.0):
        self.w, self.h, self.n = w, h, n
        self.cx = rng.random(n) * w; self.cy = y0 + rng.random(n) * (y1 - y0)
        self.ax = (20 + rng.random(n) * 50) * k; self.ay = (8 + rng.random(n) * 22) * k
        self.f1 = 0.15 + rng.random(n) * 0.35; self.f2 = 0.12 + rng.random(n) * 0.3
        self.p1 = rng.random(n) * 6.283; self.p2 = rng.random(n) * 6.283
        self.bf = 0.12 + rng.random(n) * 0.22; self.bp = rng.random(n) * 6.283
        s = max(3, int(2.6 * k)); g = np.arange(-4 * s, 4 * s + 1, dtype=np.float32)
        self.sp = np.exp(-(g[None, :] ** 2 + g[:, None] ** 2) / (2 * s * s)); self.r = 4 * s

    def add(self, img, t):
        r = self.r
        for i in range(self.n):
            b = max(0.0, np.sin(self.bf[i] * t * 6.283 + self.bp[i])) ** 2
            if b < 0.03:
                continue
            x = int(self.cx[i] + self.ax[i] * np.sin(self.f1[i] * t * 6.283 + self.p1[i]))
            y = int(self.cy[i] + self.ay[i] * np.sin(self.f2[i] * t * 6.283 + self.p2[i]))
            if r <= x < self.w - r and r <= y < self.h - r:
                img[y - r:y + r + 1, x - r:x + r + 1] += self.sp[..., None] * (C["gold"] * 0.8 + C["ivory"] * 0.2) * (b * 0.9)


# ----------------------------------------------------------------------------- shared base
class Base:
    def __init__(self, w, h, seed, sleep=False):
        self.w, self.h, self.sleep = w, h, sleep
        self.k = w / 960.0
        self.rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        vig = np.clip(((xx - w / 2) / (w * 0.72)) ** 2 + ((yy - h / 2) / (h * 0.80)) ** 2, 0, 1)
        self.vig = (1 - 0.36 * vig)[..., None]
        self.dim = 0.80 if sleep else 1.0

    def make_sky(self, hz, moon=None, mr=0.06, stars=1.0):
        w, h, rng = self.w, self.h, self.rng
        self.hz = hz
        stops = [(0.0, C["navy"]), (0.35, C["blue"]), (0.68, C["indigo"]), (0.88, C["purple"]), (1.0, C["purple"] * 0.8 + C["pink"] * 0.2)]
        sky = np.broadcast_to(gradient(hz, stops), (hz, w, 3)).copy()
        mx = w * (0.28 + rng.random() * 0.44) if moon is None else w * moon[0]
        my = hz * (0.26 + rng.random() * 0.1) if moon is None else hz * moon[1]
        r = h * mr
        self.mx, self.my = mx, my
        sky += glow(w, hz, mx, hz, w * 0.55, C["pink"] * 0.6 + C["lavender"] * 0.4, 0.34, 1.8)
        sky += glow(w, hz, mx, hz, w * 0.22, C["gold"], 0.26, 1.8)
        sky += glow(w, hz, mx, my, r * 9, C["lavender"], 0.26, 2.4)
        sky += glow(w, hz, mx, my, r * 4, C["gold"] * 0.7 + C["ivory"] * 0.3, 0.36, 2.0)
        big = Image.new("L", (w, hz), 0); d = ImageDraw.Draw(big)
        d.ellipse([mx - r, my - r, mx + r, my + r], fill=255)
        d.ellipse([mx - r * 0.52, my - r * 1.12, mx + r * 1.5, my + r * 0.9], fill=0)
        moon_m = np.asarray(big.filter(ImageFilter.GaussianBlur(1.0)), np.float32)[..., None] / 255
        glowm = np.asarray(big.filter(ImageFilter.GaussianBlur(r * 0.35)), np.float32)[..., None] / 255
        sky = sky * (1 - moon_m) + C["ivory"] * moon_m + glowm * C["gold"] * 0.5
        # stars
        n = int(w * hz / 1500 * stars)
        self.sx = rng.integers(0, w - 2, n); self.sy = (rng.random(n) ** 1.5 * hz * 0.85).astype(int)
        self.sb = rng.random(n) ** 3 * 0.9 + 0.15
        self.sf = 0.08 + rng.random(n) * 0.4; self.sp = rng.random(n)
        st = np.zeros((hz, w), np.float32); st[self.sy, self.sx] = self.sb * 0.5
        halo = np.asarray(Image.fromarray((st * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.5)), np.float32) / 255
        fade = np.clip(1 - np.linspace(0, 1, hz, dtype=np.float32)[:, None] * 1.1, 0.05, 1)
        sky += (halo * 3.0 * fade)[..., None] * C["ivory"] * 0.5
        self.fade = fade[self.sy, 0]
        self.sky_base = np.clip(sky * self.dim, 0, 1.2)
        # clouds
        self.cloud = tnoise(w, hz, rng, sx=3, sy=2)
        ys = np.linspace(0, 1, hz, dtype=np.float32)[:, None]
        self.cband = np.clip(1.1 - np.abs(ys - 0.74) * 2.8, 0, 1)
        self.cloud_col = (C["lavender"] * 0.55 + C["pink"] * 0.2) * 0.30 * self.dim

    def sky_frame(self, t, clouds=True):
        sky = self.sky_base.copy(); w, hz = self.w, self.hz
        tw = 0.55 + 0.45 * np.sin(2 * np.pi * (t * self.sf + self.sp))
        b = (self.sb * tw * self.fade * self.dim)[:, None] * C["ivory"][None, :]
        sky[self.sy, self.sx] += b
        sky[self.sy, self.sx + 1] += b * 0.4
        sky[np.minimum(self.sy + 1, hz - 1), self.sx] += b * 0.4
        if clouds:
            c = shift_x(self.cloud, t * 6.0 * self.k)
            sky += (np.clip((c - 0.52) * 2.4, 0, 1) * self.cband)[..., None] * self.cloud_col
        return sky

    def mountains(self, hz, spec):
        """Layered ridges -> (premultiplied colour, alpha) to composite as sky*(1-A)+P."""
        w, rng = self.w, self.rng
        yidx = np.arange(hz, dtype=np.float32)[:, None]
        P = np.zeros((hz, w, 3), np.float32); A = np.zeros((hz, w, 1), np.float32)
        for base, amp, col, opa, rough in spec:
            top = ridge(w, rng, hz * base, hz * amp, rough)
            m = np.clip((yidx - top[None, :]) / 1.6 + 0.5, 0, 1)
            depth = np.clip((yidx - top[None, :]) / (hz * 0.22), 0, 1)[..., None]
            shade = col * (1 - 0.22 * depth) + C["lavender"] * 0.06 * (1 - depth)
            m = m[..., None] * opa
            P = P * (1 - m) + shade * self.dim * m; A = A * (1 - m) + m
        return P, A

    def finish(self, img):
        return np.clip(img * self.vig, 0, 1)


def reflect(top, lh, blur=2.0, dark=0.55, tint=0.18):
    """Mirror the scene above the horizon into the water below it."""
    hz = top.shape[0]
    refl = top[::-1][:lh] if lh <= hz else np.concatenate([top[::-1], np.repeat(top[:1], lh - hz, 0)])[:lh]
    img = Image.fromarray((np.clip(refl, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
    refl = np.asarray(img, np.float32) / 255
    fall = np.linspace(1.0, 0.55, lh, dtype=np.float32)[:, None, None]
    return refl * dark * fall * (1 - tint) + C["navy"] * tint


# ----------------------------------------------------------------------------- 1. rain over a moonlit lake
class Rain(Base):
    def __init__(self, w, h, seed, sleep=False):
        super().__init__(w, h, seed, sleep)
        k, rng = self.k, self.rng
        hz = int(h * 0.60); self.make_sky(hz, mr=0.05, stars=0.6)
        self.P, self.A = self.mountains(hz, [(0.985, 0.20, C["purple"] * 0.8 + C["lavender"] * 0.2, 0.60, 0.52),
                                             (1.005, 0.14, C["indigo"] * 0.9 + C["navy"] * 0.1, 0.92, 0.58)])
        self.lh = h - hz
        top = self.sky_base * (1 - self.A) + self.P
        self.lake_base = reflect(top, self.lh)
        self.mist = tnoise(w, hz, rng, sx=3, sy=3)
        ys = np.arange(hz, dtype=np.float32)[:, None]
        self.mband = np.clip(1 - np.abs(ys - hz * 0.84) / (hz * 0.14), 0, 1) ** 2
        self.shim = tnoise(w, self.lh, rng, sx=26, sy=5, beta=1.3)
        # rain layers: (count, length, speed px/s, strength)
        self.layers = []
        for n, ln, sp, a in [(int(340 * k * k), 7, 520, 0.18), (int(200 * k * k), 12, 760, 0.30), (int(70 * k * k), 22, 1050, 0.48)]:
            ln = max(3, int(ln * k))
            self.layers.append(dict(n=n, ln=ln, sp=sp * k, a=a, x=rng.random(n) * w, y=rng.random(n) * (h + ln),
                                    ramp=np.linspace(0.1, 1, ln, dtype=np.float32)))
        self.slant = 0.10
        # ripples on the lake
        self.M, self.K = int(46 * k), 200
        self.rp = rng.uniform(1.5, 2.8, self.M); self.rph = rng.random(self.M)
        self.rx = rng.random((self.M, self.K)) * w; self.ry = rng.random((self.M, self.K)) ** 0.8

    def frame(self, t):
        k, w, h, hz, lh = self.k, self.w, self.h, self.hz, self.lh
        top = self.sky_frame(t) * (1 - self.A) + self.P
        top += (shift_x(self.mist, t * 3.5 * k) * self.mband)[..., None] * (C["lilac"] * 0.5 + C["pink"] * 0.2) * 0.30 * self.dim
        # lake: reflection + drifting shimmer + moon glints + ripple rings
        s = shift_x(self.shim, -t * 5 * k)
        lake = self.lake_base + ((s - 0.5) * 0.04)[..., None] * C["lavender"]
        ys = np.linspace(0, 1, lh, dtype=np.float32)[:, None]
        xs = np.arange(w, dtype=np.float32)[None, :]
        col = np.exp(-(((xs - self.mx) / (w * (0.012 + 0.03 * ys))) ** 2)) * np.clip(1 - ys * 1.1, 0, 1)
        lake += (col * np.clip((s - 0.5) * 3.2 + 0.35, 0, 1) * 0.55)[..., None] * (C["gold"] * 0.8 + C["ivory"] * 0.2)
        ring = Image.new("L", (w, lh), 0); d = ImageDraw.Draw(ring)
        for i in range(self.M):
            c = t / self.rp[i] + self.rph[i]; age = c % 1.0; kk = int(np.floor(c)) % self.K
            cx, fy = self.rx[i, kk], self.ry[i, kk]
            rr = age * (6 + 34 * (0.35 + fy)) * k; ry = rr * 0.22
            cy = fy * (lh - 3) + 2
            v = int(255 * (1 - age) ** 1.6 * (0.35 + 0.65 * fy))
            if v > 6 and rr > 0.8:
                d.ellipse([cx - rr, cy - ry, cx + rr, cy + ry], outline=v, width=1)
        ring = np.asarray(ring.filter(ImageFilter.GaussianBlur(0.7)), np.float32) / 255
        lake += (ring * 1.4)[..., None] * (C["lilac"] * 0.7 + C["ivory"] * 0.3)
        img = np.concatenate([top, lake], 0)
        # rain streaks
        rain = np.zeros((h, w), np.float32)
        for L in self.layers:
            tot = L["y"] + L["sp"] * t; cyc = np.floor(tot / (h + L["ln"])); pos = tot - cyc * (h + L["ln"])
            x0 = (L["x"] + self.slant * pos + cyc * 317.0) % w
            off = np.arange(L["ln"], dtype=np.float32)
            yi = (pos[:, None] - L["ln"] + off[None, :]).astype(np.int32)
            xi = (x0[:, None] + self.slant * off[None, :]).astype(np.int32) % w
            m = (yi >= 0) & (yi < h)
            val = np.broadcast_to(L["ramp"][None, :] * L["a"], yi.shape)
            rain[yi[m], xi[m]] = np.maximum(rain[yi[m], xi[m]], val[m])
        img += rain[..., None] * (C["lilac"] * 0.8 + C["ivory"] * 0.2) * self.dim
        img *= 0.96
        return self.finish(img)


# ----------------------------------------------------------------------------- 2. moonlit river
class River(Base):
    def __init__(self, w, h, seed, sleep=False):
        super().__init__(w, h, seed, sleep)
        k, rng = self.k, self.rng
        hz = int(h * 0.47); self.make_sky(hz, mr=0.055)
        self.P, self.A = self.mountains(hz, [(0.99, 0.22, C["purple"] * 0.75 + C["lavender"] * 0.25, 0.60, 0.52),
                                             (1.0, 0.12, C["navy"] * 0.9 + C["indigo"] * 0.1, 1.0, 0.62)])
        lh = self.lh = h - hz
        d = (np.arange(lh, dtype=np.float32) + 1) / lh; self.d = d
        step = 1.7 - 1.35 * d                      # texture is stretched toward the viewer (perspective)
        xs = np.arange(w, dtype=np.float32)[None, :]
        self.base_idx = (xs - w / 2) * step[:, None] + w / 2
        self.t1 = 1 - np.abs(2 * tnoise(w, lh, rng, sx=16, sy=5, beta=1.4) - 1)
        self.t2 = 1 - np.abs(2 * tnoise(w, lh, rng, sx=30, sy=9, beta=1.3) - 1)
        wb = gradient(lh, [(0.0, C["purple"] * 0.55 + C["pink"] * 0.12), (0.45, C["indigo"] * 0.55), (1.0, C["navy"] * 0.85)])
        self.wbase = wb
        ys = d[:, None]
        self.col = np.exp(-(((xs - self.mx) / (w * (0.015 + 0.05 * ys))) ** 2)) * np.clip(1 - ys * 0.7, 0, 1)
        # reeds along the near bank
        rows = int(h * 0.20); self.rrows = rows
        m = Image.new("L", (w, rows), 0); dr = ImageDraw.Draw(m)
        for _ in range(int(110 * k)):
            x = rng.random() * w; ht = rows * (0.25 + 0.75 * rng.random() ** 1.5); lean = (rng.random() - 0.5) * 18 * k
            wd = (1.5 + rng.random() * 2) * k
            dr.polygon([(x - wd, rows), (x + wd, rows), (x + lean, rows - ht)], fill=255)
        self.reeds = np.asarray(m.filter(ImageFilter.GaussianBlur(0.6)), np.float32)[..., None] / 255
        self.sway_w = (np.linspace(1, 0, rows, dtype=np.float32) ** 1.4)
        self.ff = Fireflies(rng, int(26), w, h, hz * 0.85, h * 0.92, k)

    def frame(self, t):
        k, w, h, hz, lh = self.k, self.w, self.h, self.hz, self.lh
        top = self.sky_frame(t) * (1 - self.A) + self.P
        flow = (t * 34 * k) * (0.2 + self.d)
        v = np.zeros((lh, w), np.float32)
        r = np.arange(lh)[:, None]
        for tex, mul, wt in ((self.t1, 1.0, 0.6), (self.t2, 1.8, 0.4)):
            idx = self.base_idx + (flow * mul)[:, None]
            i0 = np.floor(idx).astype(np.int32); f = (idx - i0).astype(np.float32)
            v += wt * (tex[r, i0 % w] * (1 - f) + tex[r, (i0 + 1) % w] * f)
        water = self.wbase + ((v - 0.5) * 0.30)[..., None] * C["lavender"]
        spark = np.clip((v - 0.70) * 6.0, 0, 1)
        water += (spark * (self.col * 1.0 + 0.10))[..., None] * (C["gold"] * 0.75 + C["ivory"] * 0.25) * self.dim
        img = np.concatenate([top, water * self.dim], 0)
        # reeds sway
        rows = self.rrows
        sh = self.sway_w * (3.2 * k) * np.sin(2 * np.pi * t / 7.0 + np.linspace(0, 2, rows, dtype=np.float32))
        rs = row_shift(self.reeds, sh)
        band = img[h - rows:]
        img[h - rows:] = band * (1 - rs) + C["navy"] * 0.45 * rs
        self.ff.add(img, t)
        return self.finish(img)


# ----------------------------------------------------------------------------- 3. misty forest with moonbeams
class Forest(Base):
    def __init__(self, w, h, seed, sleep=False):
        super().__init__(w, h, seed, sleep)
        k, rng = self.k, self.rng
        hz = int(h * 0.62); self.make_sky(hz, moon=(0.30 + rng.random() * 0.4, 0.30), mr=0.06)
        cols = [C["purple"] * 0.78 + C["lavender"] * 0.22, C["indigo"] * 0.95, C["blue"] * 0.75 + C["navy"] * 0.25, C["navy"] * 0.8]
        specs = [(0.66, 0.30, 26), (0.74, 0.40, 20), (0.84, 0.52, 14), (0.97, 0.66, 9)]    # baseline, tree height, count
        self.layers = []
        yy = np.arange(h, dtype=np.float32)
        for i, (base, ht, cnt) in enumerate(specs):
            m = Image.new("L", (w, h), 0); dr = ImageDraw.Draw(m)
            yb = h * base
            for _ in range(int(cnt * 1.6)):
                x = rng.random() * w; th = h * ht * (0.55 + 0.45 * rng.random()); tw = th * (0.16 + 0.05 * rng.random())
                for tier in range(5):
                    f0 = tier / 5; f1 = (tier + 1.35) / 5
                    y_top = yb - th * (1 - f0); y_bot = yb - th * (1 - f1) + th * 0.07
                    half = tw * (0.25 + 0.75 * f1)
                    dr.polygon([(x, y_top), (x - half, y_bot), (x + half, y_bot)], fill=255)
                dr.rectangle([x - tw * 0.04, yb - th * 0.08, x + tw * 0.04, yb + 2], fill=255)
            dr.rectangle([0, yb, w, h], fill=255)
            mask = np.asarray(m.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
            y_top_row = max(0, int(yb - h * ht - 4)); y_bot_row = min(h, int(yb + 6))
            sway = np.clip((yb - yy[y_top_row:y_bot_row]) / (h * 0.45), 0, 1) ** 1.3
            self.layers.append(dict(mask=mask, col=cols[i], r0=y_top_row, r1=y_bot_row, sway=sway,
                                    A=(2.6 - 0.5 * i) * k, T=9 + 2.3 * i, fog=tnoise(w, h, rng, sx=3, sy=3),
                                    fy=yb - h * ht * 0.35, speed=(3.0 + 1.6 * i) * k))
        self.ff = Fireflies(rng, int(36), w, h, h * 0.45, h * 0.92, k)
        n1 = tnoise(w, 1, rng, sx=9, sy=1, beta=1.4)[0]
        self.rays_n = n1
        ys = np.arange(h, dtype=np.float32)[:, None]
        self.ray_idx = (np.arange(w)[None, :] + 0.55 * ys).astype(np.int32)
        self.ray_fade = np.clip(1 - ys / (h * 0.85), 0, 1) ** 1.2

    def frame(self, t):
        k, w, h, hz = self.k, self.w, self.h, self.hz
        sky = self.sky_frame(t)
        img = np.empty((h, w, 3), np.float32)
        img[:hz] = sky
        img[hz:] = sky[-1][None]
        # moonbeams drifting through the trees
        off = int(t * 3.0 * k)
        rays = np.clip(self.rays_n[(self.ray_idx + off) % w] * 1.6 - 0.55, 0, 1) * self.ray_fade
        pulse = 0.75 + 0.25 * np.sin(2 * np.pi * t / 19.0)
        img += (rays * 0.05 * pulse)[..., None] * (C["lilac"] * 0.6 + C["gold"] * 0.4) * self.dim
        yy = np.arange(h, dtype=np.float32)[:, None]
        for L in self.layers:
            fog = shift_x(L["fog"], t * L["speed"])
            band = np.clip(1 - np.abs(yy - L["fy"]) / (h * 0.16), 0, 1) ** 2
            img += (np.clip((fog - 0.35) * 1.8, 0, 1) * band)[..., None] * (C["lilac"] * 0.5 + C["pink"] * 0.2) * 0.17 * self.dim
            r0, r1 = L["r0"], L["r1"]
            sh = L["A"] * L["sway"] * np.sin(2 * np.pi * t / L["T"] + 0.010 * np.arange(r0, r1, dtype=np.float32))
            m = np.concatenate([L["mask"][:r0], row_shift(L["mask"][r0:r1], sh), L["mask"][r1:]], 0)[..., None]
            img = img * (1 - m) + (L["col"] * self.dim) * m
        self.ff.add(img, t)
        return self.finish(img)


# ----------------------------------------------------------------------------- 4. night ocean
class Ocean(Base):
    def __init__(self, w, h, seed, sleep=False):
        super().__init__(w, h, seed, sleep)
        k, rng = self.k, self.rng
        hz = int(h * 0.50); self.make_sky(hz, mr=0.06)
        lh = self.lh = h - hz
        d = (np.arange(lh, dtype=np.float32) + 1) / lh; self.d = d
        self.u = (1.0 / (d + 0.07))[:, None]
        self.nz = tnoise(w, lh, rng, sx=7, sy=4, beta=1.4)
        self.nz2 = tnoise(w, lh, rng, sx=18, sy=8, beta=1.3)
        self.wbase = gradient(lh, [(0.0, C["purple"] * 0.6 + C["pink"] * 0.14), (0.35, C["indigo"] * 0.55), (1.0, C["navy"] * 0.8)])
        xs = np.arange(w, dtype=np.float32)[None, :]; ys = d[:, None]
        self.col = np.exp(-(((xs - self.mx) / (w * (0.02 + 0.06 * ys))) ** 2)) * np.clip(1 - ys * 0.55, 0, 1)

    def frame(self, t):
        k, w, hz, lh = self.k, self.w, self.hz, self.lh
        top = self.sky_frame(t)
        nz = shift_x(self.nz, t * 5 * k); nz2 = shift_x(self.nz2, -t * 8 * k)
        phase = self.u * 7.5 + t * 1.5 + (nz - 0.5) * 2.4 + (nz2 - 0.5) * 0.7
        crest = (0.5 + 0.5 * np.sin(phase)) ** 3
        water = self.wbase + ((nz2 - 0.5) * 0.10)[..., None] * C["lavender"]
        water += (crest * (0.05 + 0.10 * (1 - self.d[:, None])))[..., None] * C["lavender"]
        water += (crest * self.col * 1.1)[..., None] * (C["gold"] * 0.75 + C["ivory"] * 0.25) * self.dim
        # soft light on the horizon
        water[:6] += np.linspace(0.12, 0, 6, dtype=np.float32)[:, None, None] * C["pink"]
        return self.finish(np.concatenate([top, water * self.dim], 0))


# ----------------------------------------------------------------------------- 5. aurora over still water
class Aurora(Base):
    def __init__(self, w, h, seed, sleep=False):
        super().__init__(w, h, seed, sleep)
        k, rng = self.k, self.rng
        hz = int(h * 0.74); self.make_sky(hz, moon=(0.84, 0.16), mr=0.035, stars=1.5)
        self.P, self.A = self.mountains(hz, [(0.99, 0.15, C["indigo"] * 0.9 + C["purple"] * 0.1, 0.80, 0.55),
                                             (1.0, 0.09, C["navy"] * 0.95, 1.0, 0.62)])
        self.lh = h - hz
        self.rays = [tnoise(w, 1, rng, sx=n, sy=1, beta=1.1)[0] for n in (13, 17, 10)]
        self.ph = rng.random((3, 4)) * 6.283
        self.ripple = tnoise(w, self.lh, rng, sx=30, sy=6, beta=1.3)
        self.ys = np.arange(hz, dtype=np.float32)[:, None]
        self.cols = [C["lavender"] * 0.9 + C["lilac"] * 0.1, C["pink"] * 0.8 + C["lavender"] * 0.2, C["lilac"] * 0.7 + C["gold"] * 0.3]

    def frame(self, t):
        k, w, h, hz, lh = self.k, self.w, self.h, self.hz, self.lh
        sky = self.sky_frame(t, clouds=False)
        xs = np.linspace(0, 1, w, dtype=np.float32)
        for c in range(3):
            p = self.ph[c]
            yk = hz * (0.34 + 0.09 * c) + hz * 0.05 * np.sin(2 * np.pi * (xs * (1.1 + 0.3 * c)) + 0.22 * t + p[0]) \
                + hz * 0.025 * np.sin(2 * np.pi * (xs * 3.2) - 0.35 * t + p[1])
            ray = np.clip(shift_x(self.rays[c][None, :], t * (4 + 2 * c) * k)[0] * 1.25 - 0.10, 0, 1)
            ray *= 0.65 + 0.35 * np.sin(2 * np.pi * t / (11.0 + 3 * c) + p[2] + xs * 9)
            dy = self.ys - yk[None, :]
            up = np.exp(np.clip(dy, -1e4, 0) / (hz * 0.17))          # soft, long glow above the lower edge
            lower = np.exp(-(np.clip(dy, 0, 1e4) / (hz * 0.020)) ** 2)  # crisp lower edge
            prof = np.where(dy <= 0, up, lower)
            sky += (prof * ray[None, :] * 0.46 * (0.9 - 0.12 * c))[..., None] * self.cols[c] * self.dim
        top = sky * (1 - self.A) + self.P
        refl = top[::-1][:lh]
        lake = refl * np.linspace(0.45, 0.20, lh, dtype=np.float32)[:, None, None] + C["navy"] * 0.14
        lake += ((self.ripple - 0.5) * 0.06)[..., None] * C["lavender"]
        return self.finish(np.concatenate([top, lake], 0))


# ----------------------------------------------------------------------------- 6. rising lanterns
class Lanterns(Base):
    def __init__(self, w, h, seed, sleep=False):
        super().__init__(w, h, seed, sleep)
        k, rng = self.k, self.rng
        hz = int(h * 0.64); self.make_sky(hz, mr=0.05)
        self.P, self.A = self.mountains(hz, [(0.99, 0.18, C["purple"] * 0.75 + C["lavender"] * 0.25, 0.62, 0.52),
                                             (1.0, 0.10, C["navy"] * 0.9 + C["indigo"] * 0.1, 0.95, 0.60)])
        self.lh = h - hz
        top = self.sky_base * (1 - self.A) + self.P
        self.lake_base = reflect(top, self.lh, blur=2.4, dark=0.6)
        self.shim = tnoise(w, self.lh, rng, sx=24, sy=5, beta=1.3)
        n = int(30 * k) + 8; self.n = n
        self.x0 = rng.random(n) * w; self.sc = 0.45 + rng.random(n) * 0.9            # depth/size
        self.lsp = (7 + 12 * self.sc) * k * (0.8 + rng.random(n) * 0.4)
        self.y0 = rng.random(n); self.sw = (6 + rng.random(n) * 14) * k; self.sph = rng.random(n) * 6.283
        self.span = hz + 110 * k
        self.sprites = {}
        self.ff = None

    def sprite(self, s):
        key = round(s * 8)
        if key not in self.sprites:
            r = int(max(6, 34 * self.k * key / 8)); g = np.arange(-r, r + 1, dtype=np.float32)
            rr = (g[None, :] ** 2 + g[:, None] ** 2) / (r * r)
            halo = np.exp(-rr * 7.0); core = np.exp(-rr * 90.0)
            self.sprites[key] = (r, halo, core)
        return self.sprites[key]

    def frame(self, t):
        k, w, h, hz, lh = self.k, self.w, self.h, self.hz, self.lh
        top = self.sky_frame(t) * (1 - self.A) + self.P
        s = shift_x(self.shim, -t * 4 * k)
        lake = self.lake_base + ((s - 0.5) * 0.09)[..., None] * C["lavender"]
        refl = np.zeros((lh, w, 3), np.float32)
        warm = C["gold"] * 0.85 + C["pink"] * 0.15
        for i in range(self.n):
            life = ((self.y0[i] + self.lsp[i] * t / self.span) % 1.0)
            y = hz + 40 * k - life * self.span
            x = self.x0[i] + self.sw[i] * np.sin(2 * np.pi * t / (9 + 4 * self.sc[i]) + self.sph[i])
            a = smooth(life / 0.12) * smooth((1 - life) / 0.18)
            r, halo, core = self.sprite(self.sc[i])
            xi, yi = int(x), int(y)
            if a < 0.02 or xi < r or xi >= w - r:
                continue
            flick = 0.88 + 0.12 * np.sin(2 * np.pi * t * (0.7 + 0.5 * self.sc[i]) + self.sph[i] * 3)
            sp = (halo * 0.55 + core * 1.1)[..., None] * (warm * a * flick)
            for yc, tgt, mul in ((yi, top, 1.0),):
                if r <= yc < hz - r:
                    tgt[yc - r:yc + r + 1, xi - r:xi + r + 1] += sp * self.dim
                elif -r < yc < hz + r:
                    y0 = max(0, yc - r); y1 = min(hz, yc + r + 1)
                    tgt[y0:y1, xi - r:xi + r + 1] += sp[y0 - (yc - r):y1 - (yc - r)] * self.dim
            ry = int(2 * 0 + (hz - y) * 0.55) if y < hz else 0   # mirrored in the water, squashed
            if ry > 0 and ry < lh - r // 2 and r <= xi < w - r:
                y0 = max(0, ry - r // 2); y1 = min(lh, ry + r // 2 + 1)
                sq = sp[::2][: y1 - y0]
                refl[y0:y0 + sq.shape[0], xi - r:xi + r + 1] += sq * 0.45
        lake += refl * self.dim * (0.8 + 0.4 * (s - 0.5))[..., None]
        return self.finish(np.concatenate([top, lake], 0))


SCENES = {"rain": Rain, "river": River, "forest": Forest, "ocean": Ocean, "aurora": Aurora, "lanterns": Lanterns}
THEME_SCENES = {
    "dreamstate": ["forest", "rain", "river", "aurora", "lanterns", "ocean"],
    "cosmos": ["aurora", "ocean", "lanterns", "rain", "forest"],
    "ocean": ["ocean", "rain", "river", "aurora"],
    "nature": ["forest", "river", "rain", "lanterns", "ocean"],
    "zen": ["lanterns", "forest", "river", "rain", "aurora"],
}


# ----------------------------------------------------------------------------- video assembly
def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *[str(a) for a in args]], check=True)


def render_loop(name, seed, path, sleep=False, secs=LOOP, xf=XFADE):
    """Render one seamless loop: the last `xf` seconds dissolve back into the first frame."""
    sc = SCENES[name](RW, RH, seed, sleep)
    n, nx = int(secs * FPS), int(xf * FPS)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{RW}x{RH}", "-r", str(FPS), "-i", "-",
           "-vf", f"scale={W}:{H}:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
           "-g", str(FPS * 2), str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(n):
            t = i / FPS
            a = sc.frame(t)
            if i >= n - nx:
                al = smooth((i - (n - nx)) / nx)
                a = a * (1 - al) + sc.frame(t - secs) * al
            p.stdin.write((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
    finally:
        p.stdin.close(); p.wait()
    if p.returncode:
        raise RuntimeError(f"ffmpeg failed while rendering {name}")


def pick_scenes(theme, n, job=None):
    names = [s for s in (job or {}).get("scenes", []) if s in SCENES] or THEME_SCENES.get(theme, THEME_SCENES["dreamstate"])
    return [(names[i % len(names)], (i // len(names)) % 2) for i in range(n)]


def build_video(theme, n, dur, audio, out, tmp, job=None, seed0=1):
    """n scenes spread over `dur` seconds: looped animation + 3 s fades, muxed with the audio."""
    tmp = Path(tmp); seq = pick_scenes(theme, n, job); loops = {}; clips = []
    d = dur / n
    for i, key in enumerate(seq):
        if key not in loops:
            lp = tmp / f"loop_{key[0]}_{key[1]}.mp4"
            print(f"rendering animated scene: {key[0]} (variant {key[1]})", flush=True)
            render_loop(key[0], seed0 * 1000 + hash(key[0]) % 997 + key[1] * 31, lp, bool((job or {}).get("sleep", False)))
            loops[key] = lp
        c = tmp / f"clip{i}.mp4"
        fd = min(3.0, d / 3)
        _ff("-stream_loop", "-1", "-i", loops[key], "-t", f"{d:.3f}",
            "-vf", f"fade=t=in:st=0:d={fd:.2f},fade=t=out:st={d - fd:.2f}:d={fd:.2f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-r", FPS, c)
        clips.append(c)
    lst = tmp / "list.txt"; lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    _ff("-f", "concat", "-safe", "0", "-i", lst, "-i", audio, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", out)
