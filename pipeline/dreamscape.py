"""The Dreamstate - procedural cinematic dreamscape renderer.

Pure numpy + Pillow, no network. Produces layered night landscapes in the official palette:
gradient sky with a glowing horizon, soft stars, drifting cloud banks, a crescent moon with halo and rays,
atmospheric mountain ranges with mist, and a still lake that mirrors the sky.
focal: "moon" | "door" | "horizon"
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PAL = {"navy": (16, 21, 47), "blue": (32, 44, 91), "indigo": (53, 54, 111), "purple": (98, 81, 143),
       "lavender": (184, 166, 223), "lilac": (217, 200, 238), "pink": (233, 187, 216),
       "ivory": (255, 240, 217), "gold": (233, 206, 154)}
C = {k: np.array(v, np.float32) / 255 for k, v in PAL.items()}


def _fbm(w, h, rng, base=6, octaves=5, persist=0.55):
    out = np.zeros((h, w), np.float32); amp, tot = 1.0, 0.0
    for o in range(octaves):
        gw, gh = base * 2 ** o + 2, max(2, int(base * 2 ** o * h / w) + 2)
        g = Image.fromarray((rng.random((gh, gw)) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
        out += np.asarray(g, np.float32) / 255 * amp; tot += amp; amp *= persist
    out /= tot
    return (out - out.min()) / max(out.max() - out.min(), 1e-6)


def _ridge(w, rng, base_y, amp, rough=0.5, seed_off=0):
    xs = np.linspace(0, 1, w, dtype=np.float32)
    y = np.zeros(w, np.float32); a = 1.0
    for k in range(7):
        f = 1.6 * 2 ** k; ph = rng.random() * 6.28
        y += np.sin(xs * f * 6.283 + ph) * a * (rng.random() * 0.5 + 0.75); a *= rough
    y = (y - y.min()) / max(y.max() - y.min(), 1e-6)
    return base_y - y * amp


def _glow(w, h, cx, cy, r, color, strength, power=2.2):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    g = np.clip(1 - d, 0, 1) ** power * strength
    return g[..., None] * color[None, None, :]


def render(w, h, seed=1, focal="moon", sleep=False):
    rng = np.random.default_rng(seed)
    hz = int(h * (0.60 + rng.random() * 0.04))          # horizon line
    yy = np.linspace(0, 1, hz, dtype=np.float32)[:, None, None]

    # ---- sky gradient: navy top -> blue -> indigo -> purple glow at the horizon ----
    stops = [(0.0, C["navy"]), (0.35, C["blue"]), (0.68, C["indigo"]), (0.88, C["purple"]), (1.0, C["purple"] * 0.8 + C["pink"] * 0.2)]
    sky = np.zeros((hz, w, 3), np.float32)
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        m = ((yy >= a) & (yy <= b)).astype(np.float32)
        t = np.clip((yy - a) / (b - a), 0, 1)
        sky += m * (ca + (cb - ca) * t)
    sky = np.broadcast_to(sky, (hz, w, 3)).copy()
    if sleep: sky *= 0.78

    # ---- moon placement ----
    mx = w * (0.28 + rng.random() * 0.44) if focal != "door" else w * (0.5 + (rng.random() - 0.5) * 0.12)
    my = hz * (0.30 + rng.random() * 0.10); mr = h * (0.074 if focal != 'horizon' else 0.052)

    # ---- nebula / cloud banks (fbm), lavender + blush, concentrated toward horizon ----
    n1 = _fbm(w, hz, rng, base=4, octaves=6); n2 = _fbm(w, hz, rng, base=7, octaves=5)
    ys = np.linspace(0, 1, hz, dtype=np.float32)[:, None]
    band = np.clip(1.15 - np.abs(ys - 0.78) * 2.6, 0, 1)
    cl = np.clip((n1 - 0.46) * 2.4, 0, 1) * band
    sky += cl[..., None] * (C["lavender"] * 0.55 + C["pink"] * 0.20) * 0.38
    high = np.clip((n2 - 0.55) * 2.2, 0, 1) * np.clip(1 - ys * 1.6, 0, 1) * 0.5
    sky += high[..., None] * C["purple"] * 0.35

    # ---- horizon glow ----
    gx = mx if focal != "horizon" else w * 0.5
    sky += _glow(w, hz, gx, hz, w * 0.55, C["pink"] * 0.6 + C["lavender"] * 0.4, 0.35, 1.8)
    sky += _glow(w, hz, gx, hz, w * 0.22, C["gold"], 0.30 if focal != "horizon" else 0.62, 1.8)

    # ---- stars (varied, with soft glow and a few sparkles) ----
    st = np.zeros((hz, w), np.float32)
    count = int(w * hz / 1700)
    sx = rng.random(count) * w; sy = (rng.random(count) ** 1.5) * hz * 0.85
    br = rng.random(count) ** 3 * 0.9 + 0.12
    for x, y, b in zip(sx, sy, br):
        xi, yi = int(x), int(y)
        if 0 <= yi < hz and 0 <= xi < w: st[yi, xi] = max(st[yi, xi], b)
    st_img = Image.fromarray((st * 255).astype(np.uint8))
    stars = np.asarray(st_img.filter(ImageFilter.GaussianBlur(0.7)), np.float32) / 255 * 2.2
    halo = np.asarray(st_img.filter(ImageFilter.GaussianBlur(3.2)), np.float32) / 255 * 5.0
    fade = np.clip(1 - ys * 1.05, 0.05, 1)
    sky += (np.clip(stars + halo * 0.35, 0, 1) * fade)[..., None] * C["ivory"] * (0.55 if sleep else 0.9)

    # ---- moon: crescent + halo + soft rays ----
    sky += _glow(w, hz, mx, my, mr * 9, C["lavender"], 0.28, 2.4)
    sky += _glow(w, hz, mx, my, mr * 4, C["gold"] * 0.7 + C["ivory"] * 0.3, 0.38, 2.0)
    if focal in ("moon", "horizon"):
        gy, gx2 = np.mgrid[0:hz, 0:w].astype(np.float32)
        ang = np.arctan2(gy - my, gx2 - mx)
        rays = (np.sin(ang * 6 + rng.random() * 6) * 0.5 + 0.5) ** 2
        dist = np.sqrt((gx2 - mx) ** 2 + (gy - my) ** 2) / (w * 0.5)
        sky += (rays * np.clip(1 - dist, 0, 1) ** 2.6 * (0.05 if focal == 'moon' else 0.0))[..., None] * C["ivory"]
        big = Image.new("L", (w, hz), 0); d = ImageDraw.Draw(big)
        d.ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=255)
        d.ellipse([mx - mr * 0.52, my - mr * 1.12, mx + mr * 1.5, my + mr * 0.9], fill=0)
        moon = np.asarray(big.filter(ImageFilter.GaussianBlur(1.1)), np.float32) / 255
        glowm = np.asarray(big.filter(ImageFilter.GaussianBlur(mr * 0.35)), np.float32) / 255
        sky = sky * (1 - moon[..., None]) + C["ivory"] * moon[..., None]
        sky += glowm[..., None] * C["gold"] * 0.55
    elif focal == "door":
        sky += _glow(w, hz, mx, hz - h * 0.10, h * 0.36, C["gold"], 0.32, 2.2)

    sky = np.clip(sky, 0, 1)

    # ---- mountains: far -> near, atmospheric perspective + mist between ranges ----
    layers = [(0.980, 0.17, C["purple"] * 0.82 + C["lavender"] * 0.18, 0.62, 0.52),
              (0.995, 0.14, C["indigo"] * 0.95 + C["purple"] * 0.05, 0.84, 0.55),
              (1.015, 0.10, C["navy"] * 0.9 + C["blue"] * 0.1, 1.00, 0.60)]
    sky_full = np.zeros((h, w, 3), np.float32); sky_full[:hz] = sky
    canvas = sky_full.copy()
    canvas[hz:] = sky[-1][None, :, :]                   # placeholder below horizon (lake drawn later)
    X = np.arange(w)
    mist_masks = []
    for base, amp, col, opa, rough in layers:
        top = _ridge(w, rng, hz * base, hz * amp * 1.45 * (0.55 if sleep else 1.0), rough)
        yidx = np.arange(h)[:, None]
        mask = (yidx >= top[None, :]).astype(np.float32)
        mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.9)), np.float32) / 255
        mask[int(hz * (base + 0.0)) + int(hz * 0.2):] = np.maximum(mask[int(hz * (base + 0.0)) + int(hz * 0.2):], 0)
        shade = col[None, None, :] * np.ones((h, w, 1), np.float32)
        # darken toward the foot of each range, lighten at ridge line (rim light)
        depth = np.clip((yidx - top[None, :]) / (h * 0.16), 0, 1)[..., None]
        shade = shade * (1.0 - 0.22 * depth) + C["lavender"] * 0.06 * (1 - depth)
        m = mask[..., None] * opa
        canvas = canvas * (1 - m) + shade * m
        # mist hugging the foot of this range
        mist = np.clip(1 - np.abs(yidx - (top[None, :] + hz * 0.075)) / (hz * 0.12), 0, 1) ** 2
        mist = mist * (0.5 + 0.5 * _fbm(w, h, rng, base=3, octaves=4)) * mask
        canvas += mist[..., None] * (C["lilac"] * 0.5 + C["pink"] * 0.2) * 0.22

    if focal == "door":
        dw, dh = h * 0.062, h * 0.25
        big = Image.new("L", (w, hz), 0); d = ImageDraw.Draw(big)
        d.rounded_rectangle([mx - dw / 2, hz - dh, mx + dw / 2, hz + 40], radius=dw / 2, fill=255)
        door = np.asarray(big.filter(ImageFilter.GaussianBlur(0.9)), np.float32) / 255
        dg = np.asarray(big.filter(ImageFilter.GaussianBlur(h * 0.018)), np.float32) / 255
        dgl = np.asarray(big.filter(ImageFilter.GaussianBlur(h * 0.07)), np.float32) / 255
        top_c = canvas[:hz]
        top_c += dgl[..., None] * C["gold"] * 0.42 + dg[..., None] * C["gold"] * 0.32
        top_c = top_c * (1 - door[..., None]) + (C["ivory"] * 0.97 + C["gold"] * 0.03) * door[..., None]
        canvas[:hz] = np.clip(top_c, 0, 1.2)

    # ---- lake: mirrored sky/mountains, rippled, darkened ----
    lh = h - hz
    top_part = canvas[:hz]
    refl = top_part[::-1][:lh] if lh <= hz else np.concatenate([top_part[::-1], np.repeat(top_part[:1][::-1], lh - hz, 0)])[:lh]
    refl_img = Image.fromarray((np.clip(refl, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.2))
    refl = np.asarray(refl_img, np.float32) / 255
    # horizontal ripple displacement increasing toward the viewer
    rip = _fbm(w, lh, rng, base=3, octaves=3)
    shift = ((rip - 0.5) * 14 * (np.linspace(0.2, 1, lh, dtype=np.float32)[:, None])).astype(np.int32)
    cols = (np.arange(w)[None, :] + shift) % w
    refl = refl[np.arange(lh)[:, None], cols]
    streak = _fbm(w, lh, rng, base=2, octaves=2)
    lake = refl * (0.62 - 0.30 * np.linspace(0, 1, lh, dtype=np.float32)[:, None, None])
    lake = lake * 0.82 + C["navy"] * 0.18 + C["indigo"] * 0.05
    lake += (np.clip(streak - 0.55, 0, 1) * 0.35)[..., None] * C["lavender"] * np.linspace(0.6, 0.1, lh, dtype=np.float32)[:, None, None]
    # moon / door light column on the water
    colx = mx
    xs = np.arange(w, dtype=np.float32)[None, :]
    ysl = np.linspace(0, 1, lh, dtype=np.float32)[:, None]
    wob = (rip - 0.5) * 38 * (0.3 + ysl)
    colw = (w * (0.012 + 0.035 * ysl))
    column = np.exp(-(((xs - colx - wob) / colw) ** 2)) * np.clip(1 - ysl * 1.15, 0, 1) ** 1.3
    lake += (column * (0.78 if focal != "horizon" else 0.5))[..., None] * (C["gold"] * 0.85 + C["ivory"] * 0.15)
    canvas[hz:] = lake
    # soft shoreline highlight
    sh = np.clip(1 - np.abs(np.arange(h)[:, None] - hz) / 3, 0, 1) * 0.10
    canvas += sh[..., None] * C["lilac"]
    # ---- global grade: soft vignette, gentle glow lift, film grain ----
    img = np.clip(canvas, 0, 1)
    yy2, xx2 = np.mgrid[0:h, 0:w].astype(np.float32)
    vig = np.clip(((xx2 - w / 2) / (w * 0.72)) ** 2 + ((yy2 - h * 0.5) / (h * 0.80)) ** 2, 0, 1)
    img = img * (1 - 0.42 * vig[..., None]) + C["navy"] * 0.42 * vig[..., None] * 0.6
    bloom = np.asarray(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(h * 0.012)), np.float32) / 255
    img = np.clip(img + np.clip(bloom - 0.5, 0, 1) * 0.35, 0, 1)
    img += rng.normal(0, 0.012, (h, w, 1)).astype(np.float32)
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


if __name__ == "__main__":
    import sys
    f = sys.argv[1] if len(sys.argv) > 1 else "moon"
    render(1280, 720, 7, f).save(f"/tmp/t/scene_{f}.jpg", quality=92)
