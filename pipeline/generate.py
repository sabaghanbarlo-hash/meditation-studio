import asyncio, json, os, random, re, subprocess, sys, urllib.parse, urllib.request, io
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SR, W, H, FPS = 44100, 1920, 1080, 24
OUT, TMP = Path("out"), Path("tmp")
FONTS = Path(__file__).parent / "fonts"

# ---------- The Dreamstate brand ----------
PAL = {"navy": (16, 21, 47), "blue": (32, 44, 91), "indigo": (53, 54, 111), "purple": (98, 81, 143),
       "lavender": (184, 166, 223), "lilac": (217, 200, 238), "pink": (233, 187, 216),
       "ivory": (255, 240, 217), "gold": (233, 206, 154)}
BRAND_STYLE = ("premium cinematic fantasy landscape, celestial midnight-blue sky, deep indigo mountains, luminous lavender clouds, "
               "subtle blush-pink atmospheric haze, delicate champagne-gold crescent moon, tiny realistic stars, volumetric moonlight, "
               "soft ethereal mist, elegant composition, rich atmospheric depth, subtle film grain, dreamy emotionally immersive atmosphere, "
               "sophisticated color grading, high detail, 16:9, no people, no text, no logos")

# Visual journey: opening -> settling in -> main -> deepest moment -> closing (all inside the same dream universe)
JOURNEY = {
    "dreamstate": [
        "crescent moon above a vast lavender cloud sea, midnight blue sky full of stars",
        "reflective celestial lake between misty indigo mountains, soft moonlight on the water",
        "quiet moonlit garden path with floating lights, dim lavender mist",
        "dreamy waterfall falling into a still pool, misty mountain silhouettes, gentle blush clouds",
        "glowing doorway standing in a calm moonlit lake, golden light spilling over the water",
        "distant palace among clouds and stars, luminous lavender horizon",
        "glowing horizon between indigo mountains, soft pink and lavender clouds drifting",
        "calm midnight ocean under a crescent moon and delicate celestial halo",
        "peaceful night sky over still reflective water, drifting celestial dust, very dark and quiet",
    ],
}
THEMES = {
    "dreamstate": JOURNEY["dreamstate"],
    "cosmos": ["soft nebula in deep navy and purple", "starry night sky over a quiet lake", "gentle aurora over indigo hills", "glowing galaxy swirl in lavender", "crescent moon above calm clouds", "milky way over a still moonlit sea"],
    "ocean": ["slow gentle waves under moonlight", "quiet moonlit shore at dusk", "calm midnight sea with lavender haze", "moonlit calm sea", "soft surf on pale sand under stars", "sea mist at twilight"],
    "nature": ["misty mountain lake at twilight, indigo peaks", "moonlit meadow with soft glowing flowers", "calm river through dark hills under stars", "still pond with glowing lotus flowers at night", "moonlit forest clearing in blue mist", "twilight hills under lavender clouds"],
    "zen": ["moonlit zen garden with raked sand and stones", "bamboo grove in soft blue moonlight", "candlelit stone altar in night mist", "misty japanese temple garden at twilight", "smooth stacked stones by moonlit water", "paper lanterns glowing softly at night"],
}
# per-theme background tint used by the procedural fallback scene
TINTS = {k: "dreamstate" for k in THEMES}

VOICES = {"ava": "en-US-AvaMultilingualNeural", "emma": "en-US-EmmaMultilingualNeural", "jenny": "en-US-JennyNeural",
          "aria": "en-US-AriaNeural", "libby": "en-GB-LibbyNeural", "sonia": "en-GB-SoniaNeural",
          "natasha": "en-AU-NatashaNeural", "guy": "en-US-GuyNeural", "ryan": "en-GB-RyanNeural"}


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def ff(*args):
    run(["ffmpeg", "-y", "-loglevel", "error", *args])


def pcm(path):
    r = run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768


def write_wav(path, x):
    import wave
    st = (np.clip(np.stack([x, x], 1), -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(st.tobytes())


def silence(sec):
    return np.zeros(int(sec * SR), dtype=np.float32)


# ---------- 1. voiceover (more human) ----------
# Whole phrases are spoken in one go so the neural voice keeps natural rhythm and intonation,
# gaps are slightly varied instead of metronomic, and the voice gets a warm, gentle studio polish.
def plan(script):
    script = re.sub(r"\[pause\s*(\d+)\s*s?\]", r"\n\n@@\1@@\n\n", script)
    items = []
    for para in re.split(r"\n\s*\n", script.strip()):
        para = " ".join(para.split())
        m = re.fullmatch(r"@@(\d+)@@", para)
        if m:
            items.append(("gap", float(m.group(1)))); continue
        sents = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", para) if s.strip()]
        chunk = ""
        for s in sents:  # group 2-3 sentences (~260 chars) per synthesis call
            if chunk and len(chunk) + len(s) > 260:
                items += [("say", chunk), ("gap", "short")]; chunk = s
            else:
                chunk = (chunk + " " + s).strip()
        if chunk:
            items.append(("say", chunk))
        items.append(("gap", "long"))
    return items


def gap_len(kind):
    if kind == "short":
        return random.uniform(0.7, 1.3)
    if kind == "long":
        return random.uniform(2.2, 3.4)
    return float(kind)


def soften(text):
    # commas/ellipses give the voice natural breathing points
    text = text.replace("...", "…")
    return re.sub(r"\s*…\s*", "… ", text).strip()


async def synth(text, voice, path):
    import edge_tts
    for attempt in range(4):
        try:
            await edge_tts.Communicate(soften(text), voice, rate="-12%", pitch="-1Hz").save(str(path)); return
        except Exception as e:
            print("tts retry", attempt, e); await asyncio.sleep(2 + attempt * 3)
    raise RuntimeError("edge-tts failed repeatedly")


def trim_and_fade(x, thresh=0.004, fade=0.02):
    idx = np.where(np.abs(x) > thresh)[0]
    if len(idx):
        x = x[max(idx[0] - int(0.05 * SR), 0): idx[-1] + int(0.12 * SR)]
    f = int(fade * SR)
    if len(x) > 2 * f:
        x = x.copy(); x[:f] *= np.linspace(0, 1, f); x[-f:] *= np.linspace(1, 0, f)
    return x


def polish_voice(v):
    # warm, intimate, slightly "in a quiet room": low rumble cut, soft de-harshing, gentle compression, tiny room reflections
    write_wav(TMP / "voice_raw.wav", v / max(np.abs(v).max(), 1e-6) * 0.9)
    ff("-i", TMP / "voice_raw.wav", "-af",
       "highpass=f=70,equalizer=f=180:t=q:w=1:g=2,equalizer=f=4500:t=q:w=1.5:g=-3,equalizer=f=8000:t=q:w=1:g=-3,"
       "acompressor=threshold=-22dB:ratio=2.2:attack=20:release=250:makeup=2,aecho=0.85:0.7:45|90:0.10|0.05,loudnorm=I=-18:TP=-2:LRA=7",
       "-ar", SR, "-ac", 1, TMP / "voice_pol.wav")
    p = pcm(TMP / "voice_pol.wav")
    return p / max(np.abs(p).max(), 1e-6) * 0.9


async def voiceover(job):
    voice = VOICES.get(job.get("voice", "ava"), VOICES["ava"])
    parts, n = [silence(3)], 0
    for kind, val in plan(job["script"]):
        if kind == "gap":
            parts.append(silence(gap_len(val))); continue
        p = TMP / f"s{n}.mp3"; n += 1
        await synth(val, voice, p)
        parts.append(trim_and_fade(pcm(p)))
    parts.append(silence(6))
    return polish_voice(np.concatenate(parts))


# ---------- 2. music (seamless 60s loop, generated) ----------
def music(kind, total):
    L = 60; n = L * SR; t = np.arange(n) / SR
    if kind == "ocean":
        f = np.fft.rfftfreq(n, 1 / SR)
        spec = (np.random.randn(len(f)) + 1j * np.random.randn(len(f))) / np.maximum(f, 30) * (f < 1500)
        x = np.fft.irfft(spec, n); x /= np.abs(x).max()
        x *= 0.5 + 0.5 * np.sin(2 * np.pi * t / 12) ** 2; gain = 0.30
    else:
        notes = {"calm": [110, 164.8, 220, 277.2], "deep": [55, 82.4, 110, 165], "bright": [196, 293.7, 392, 493.9]}.get(kind, [110, 164.8, 220, 277.2])
        x = np.zeros(n)
        for k, f0 in enumerate(notes):
            for d in (0, 3):
                f = round(f0 * 60) / 60 + d / 60
                x += np.sin(2 * np.pi * f * t) * (0.65 + 0.35 * np.sin(2 * np.pi * t * (k + 2) / 60 + k))
        x /= np.abs(x).max(); gain = 0.16
    m = np.tile(x.astype(np.float32), int(total / L) + 2)[: int(total * SR)] * gain
    fi, fo = 6 * SR, 8 * SR
    m[:fi] *= np.linspace(0, 1, fi); m[-fo:] *= np.linspace(1, 0, fo)
    return m


def duck(m, voice, depth=0.35):
    # music dips softly while the voice speaks so the narration always sits on top
    win = SR // 2
    env = np.convolve(np.abs(voice), np.ones(win) / win, "same")
    env = np.convolve(env, np.ones(win) / win, "same")
    env = np.clip(env / max(env.max(), 1e-6) * 1.6, 0, 1)
    return m * (1 - depth * env[: len(m)])


# ---------- 3. images ----------
def font(name_candidates, size):
    for p in name_candidates:
        try: return ImageFont.truetype(str(p), size)
        except Exception: pass
    return ImageFont.load_default()


def dream_scene(w, h, seed, focal="moon"):
    """On-brand procedural dreamscape (used when the image service fails, and as a base for thumbnails)."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, h)[:, None, None]
    top, mid, low = np.array(PAL["navy"]), np.array(PAL["blue"]), np.array(PAL["indigo"])
    sky = np.where(y < 0.6, top + (mid - top) * (y / 0.6), mid + (low - mid) * ((y - 0.6) / 0.4))
    img = Image.fromarray(np.broadcast_to(sky, (h, w, 3)).astype(np.uint8))
    layer = lambda: Image.new("RGBA", (w, h), (0, 0, 0, 0))
    # stars
    st = layer(); sd = ImageDraw.Draw(st)
    for _ in range(int(w * h / 4500)):
        x, yy = rng.random() * w, rng.random() ** 1.6 * h * 0.7
        a = int(60 + rng.random() * 160); r = rng.choice([1, 1, 1, 2])
        sd.ellipse([x - r, yy - r, x + r, yy + r], fill=(*PAL["ivory"], a))
    img = Image.alpha_composite(img.convert("RGBA"), st.filter(ImageFilter.GaussianBlur(0.6)))
    # clouds: soft lavender / blush blobs
    cl = layer(); cd = ImageDraw.Draw(cl)
    for i in range(22):
        cx, cy = rng.random() * w, h * (0.55 + rng.random() * 0.3)
        rx, ry = w * (0.10 + rng.random() * 0.18), h * (0.03 + rng.random() * 0.06)
        col = PAL[["lavender", "pink", "purple", "lilac"][i % 4]]
        cd.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(*col, int(26 + rng.random() * 36)))
    img = Image.alpha_composite(img, cl.filter(ImageFilter.GaussianBlur(h * 0.03)))
    # moon + halo
    mx, my = w * (0.60 + rng.random() * 0.14), h * (0.17 + rng.random() * 0.04); mr = h * 0.085
    halo = layer(); hd = ImageDraw.Draw(halo)
    for k, a in ((6.0, 20), (3.6, 36), (2.0, 60)):
        hd.ellipse([mx - mr * k, my - mr * k, mx + mr * k, my + mr * k], fill=(*PAL["gold"], a))
    img = Image.alpha_composite(img, halo.filter(ImageFilter.GaussianBlur(h * 0.05)))
    moon = layer(); md = ImageDraw.Draw(moon)
    md.ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=(*PAL["ivory"], 255))
    md.ellipse([mx - mr * 0.55, my - mr * 1.15, mx + mr * 1.45, my + mr * 0.85], fill=(0, 0, 0, 0))
    mask = moon.split()[3]
    cut = Image.new("L", (w, h), 0); ImageDraw.Draw(cut).ellipse([mx - mr * 0.55, my - mr * 1.15, mx + mr * 1.45, my + mr * 0.85], fill=255)
    moon.putalpha(Image.fromarray(np.clip(np.array(mask).astype(int) - np.array(cut).astype(int), 0, 255).astype(np.uint8)))
    img = Image.alpha_composite(img, moon)
    # mountains (3 layers, far to near) + lake reflection glow
    def ridge(base, amp, col, a):
        xs = np.arange(w); ys = base * h
        for f, ph, am in ((1.3, rng.random() * 6, 1.0), (3.1, rng.random() * 6, 0.45), (7.0, rng.random() * 6, 0.15)):
            ys = ys - np.sin(xs / w * 2 * np.pi * f + ph) * amp * h * am
        pl = layer(); ImageDraw.Draw(pl).polygon([(0, h)] + list(zip(xs.tolist(), ys.tolist())) + [(w, h)], fill=(*col, a))
        return pl.filter(ImageFilter.GaussianBlur(1.2))
    for base, amp, col, a in ((0.66, 0.05, PAL["purple"], 150), (0.72, 0.06, PAL["indigo"], 210), (0.80, 0.05, PAL["navy"], 255)):
        img = Image.alpha_composite(img, ridge(base, amp, col, a))
    glow = layer(); gd = ImageDraw.Draw(glow)
    gd.ellipse([mx - w * 0.05, h * 0.82, mx + w * 0.05, h * 0.98], fill=(*PAL["gold"], 60))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(h * 0.03)))
    img = img.convert("RGB")
    grain = (rng.normal(0, 3.5, (h, w, 1))).astype(np.float32)
    return Image.fromarray(np.clip(np.array(img).astype(np.float32) + grain, 0, 255).astype(np.uint8))


def fetch(prompt, w, h, seed, path):
    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width={w}&height={h}&nologo=true&seed={seed}"
    for attempt in range(3):
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=90).read()
            img = Image.open(io.BytesIO(data)).convert("RGB").resize((w, h), Image.LANCZOS); img.save(path, quality=93); return img
        except Exception as e:
            print("image retry", attempt, e); import time; time.sleep(5 + attempt * 5)
    return None


def get_image(subject, seed, path):
    img = fetch(f"{subject}, {BRAND_STYLE}", W, H, seed, path)
    if img is None:
        print("using on-brand procedural fallback"); dream_scene(W, H, seed).save(path, quality=93)


def grade(img):
    # pull every image toward the Dreamstate palette (soft navy shadows, lavender highlights, no harsh contrast)
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255
    lum = a.mean(2, keepdims=True)
    shadow, high = np.array(PAL["navy"]) / 255, np.array(PAL["lilac"]) / 255
    toned = shadow + (high - shadow) * lum
    out = a * 0.62 + toned * 0.38
    out = np.clip((out - 0.5) * 0.94 + 0.5 + 0.01, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))


# ---------- 4. video ----------
def build_video(images, dur, audio, out):
    d = dur / len(images); frames = round(d * FPS); clips = []
    for i, img in enumerate(images):
        z = f"1+0.10*on/{frames}" if i % 2 == 0 else f"1.10-0.10*on/{frames}"  # slow, almost imperceptible push/pull
        vf = (f"scale=3840:-2,zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}:fps={FPS},"
              f"fade=t=in:st=0:d=3,fade=t=out:st={d-3:.2f}:d=3,format=yuv420p")
        c = TMP / f"clip{i}.mp4"
        ff("-i", img, "-vf", vf, "-frames:v", frames, "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", c)
        clips.append(c)
    lst = TMP / "list.txt"; lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    ff("-f", "concat", "-safe", "0", "-i", lst, "-i", audio, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out)


# ---------- 5. thumbnail (The Dreamstate design system, 1280x720) ----------
def ensure_fonts():
    FONTS.mkdir(exist_ok=True)
    src = {"CormorantGaramond-Bold.ttf": "https://github.com/google/fonts/raw/main/ofl/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf",
           "Jost.ttf": "https://github.com/google/fonts/raw/main/ofl/jost/Jost%5Bwght%5D.ttf"}
    for name, url in src.items():
        p = FONTS / name
        if not p.exists():
            try:
                p.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read())
            except Exception as e:
                print("font download failed (using fallback):", name, e)
    serif = [FONTS / "CormorantGaramond-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"]
    sans = [FONTS / "Jost.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    return serif, sans


def headline_for(job, title):
    """Short, emotional 2-6 word headline for the thumbnail."""
    if job.get("thumbnail_text"):
        return job["thumbnail_text"].strip().upper()
    t = re.sub(r"\b\d+[- ]?(minute|min|hour|hr)s?\b", "", title or "", flags=re.I)
    t = re.sub(r"\b(guided|meditation|for beginners|\| ?guided meditation)\b", "", t, flags=re.I)
    t = " ".join(t.replace("|", " ").replace(":", " ").split())
    words = t.split()
    if len(words) > 6:
        words = words[:6]
    if len(words) < 2:
        script = job.get("script", "").lower()
        for key, txt in (("sleep", "LET GO & SLEEP"), ("anx", "YOU ARE SAFE HERE"), ("future", "MEET YOUR FUTURE SELF"),
                         ("manifest", "STEP INTO YOUR DREAM LIFE"), ("affirm", "I AM ENOUGH")):
            if key in script:
                return txt
        return "A PEACEFUL MIND"
    return " ".join(words).upper()


def wrap(d, text, fnt, maxw):
    lines, cur = [], ""
    for w in text.split():
        test = (cur + " " + w).strip()
        if d.textlength(test, font=fnt) <= maxw or not cur: cur = test
        else: lines.append(cur); cur = w
    lines.append(cur)
    return lines


def thumbnail(base_img, headline, out, seed=1, regrade=True):
    TW, TH = 1280, 720
    serif, sans = ensure_fonts()
    img = base_img.convert("RGB").resize((TW, TH), Image.LANCZOS)
    img = (grade(img) if regrade else img).convert("RGBA")
    # soft vignette + lower-centre darkening so the headline always has strong contrast
    yy, xx = np.mgrid[0:TH, 0:TW]
    vig = np.clip(((xx - TW / 2) / (TW * 0.75)) ** 2 + ((yy - TH / 2) / (TH * 0.8)) ** 2, 0, 1)
    shade = np.clip(0.30 * vig + 0.30 * np.exp(-((yy - TH * 0.55) / (TH * 0.28)) ** 2), 0, 0.65)
    ov = np.zeros((TH, TW, 4), np.uint8); ov[..., :3] = PAL["navy"]; ov[..., 3] = (shade * 255).astype(np.uint8)
    img = Image.alpha_composite(img, Image.fromarray(ov))
    # headline (kept inside a safe margin, away from the edges)
    d = ImageDraw.Draw(img)
    size = 118
    while size > 60:
        f = font(serif, size); lines = wrap(d, headline, f, TW - 380)
        if len(lines) <= 3 and max(d.textlength(l, font=f) for l in lines) <= TW - 380: break
        size -= 6
    lh = int(size * 1.08); total = lh * len(lines)
    y0 = int(TH * 0.60 - total / 2)
    glow = Image.new("RGBA", (TW, TH), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
    shadow = Image.new("RGBA", (TW, TH), (0, 0, 0, 0)); sd = ImageDraw.Draw(shadow)
    y = y0
    for ln in lines:
        x = (TW - d.textlength(ln, font=f)) / 2
        gd.text((x, y), ln, font=f, fill=(*PAL["lavender"], 150))
        sd.text((x + 3, y + 5), ln, font=f, fill=(*PAL["navy"], 230))
        y += lh
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(7)))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(14)))
    d = ImageDraw.Draw(img); y = y0
    for ln in lines:
        x = (TW - d.textlength(ln, font=f)) / 2
        d.text((x, y), ln, font=f, fill=PAL["ivory"]); y += lh
    # delicate gold rule + brand line
    ry = y0 + total + 14
    d.line([(TW / 2 - 70, ry), (TW / 2 + 70, ry)], fill=(*PAL["gold"], 255), width=2)
    bf = font(sans, 30); brand = "T H E   D R E A M S T A T E"
    d.text(((TW - d.textlength(brand, font=bf)) / 2, ry + 18), brand, font=bf, fill=PAL["gold"])
    img.convert("RGB").save(out, quality=93, optimize=True)


def make_thumbnail(job, title, fallback_img, out, seed):
    """Dedicated thumbnail artwork (single clear focal point), not just a darkened video frame."""
    subject = job.get("thumbnail_prompt") or random.choice(
        ["a glowing crescent moon over a lavender cloud sea, single clear focal point, open dark space in the lower centre",
         "a golden doorway glowing in a calm reflective moonlit lake, single clear focal point, open dark space in the lower centre",
         "a luminous horizon between indigo mountains under a crescent moon, single clear focal point, open dark space in the lower centre"])
    path = TMP / "thumb_base.jpg"
    base = fetch(f"{subject}, {BRAND_STYLE}", 1280, 720, seed, path); real = base is not None
    if base is None:
        base = dream_scene(1280, 720, seed) if fallback_img is None else Image.open(fallback_img)
    headline = headline_for(job, title)
    thumbnail(base, headline, out, seed, regrade=real)
    print("Thumbnail headline:", headline)


# ---------- 6. metadata ----------
def metadata(job):
    script = " ".join(job["script"].split()); script = re.sub(r"\[pause[^\]]*\]", "", script)
    first = re.split(r"(?<=[.!?])\s+", script)[:2]
    title = (job.get("title") or first[0])[:70].strip(" .")
    tags = ["guided meditation", "meditation", "relaxation", "mindfulness", "stress relief", "calm", "sleep meditation", "inner peace", "the dreamstate"]
    meta = {"title": f"{title} | Guided Meditation"[:100],
            "description": " ".join(first) + "\n\nFind a comfortable position, press play, and let yourself relax.\n\nThe Dreamstate — Enter the life you imagine.\n\nVoice and visuals in this video are AI-generated.\n\n#meditation #mindfulness #relaxation #guidedmeditation",
            "tags": tags}
    key = os.getenv("GEMINI_API_KEY")
    if key:
        try:
            prompt = ("Write YouTube metadata for this guided meditation script for the channel 'The Dreamstate' (tagline: Enter the life you imagine). "
                      "Return JSON only with keys title (max 70 chars, calm, accurate and searchable), description (150-250 words, no unrealistic promises, ends with 4-6 hashtags), "
                      "tags (list of 10-15 short strings, no keyword stuffing).\n\nSCRIPT:\n" + script[:6000])
            body = json.dumps({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}).encode()
            model = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
            req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}", body, {"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=60).read())
            g = json.loads(r["candidates"][0]["content"]["parts"][0]["text"])
            meta = {"title": g["title"][:100], "description": g["description"][:4900], "tags": [str(t) for t in g["tags"]]}
        except Exception as e:
            print("gemini failed, using template:", e)
    out, total = [], 0
    for t in meta["tags"]:
        if total + len(t) + 1 > 450: break
        out.append(t); total += len(t) + 1
    meta["tags"] = out
    return meta


# ---------- 7. upload ----------
def upload(video, thumb, meta, privacy):
    need = ["YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"]
    if privacy == "none" or not all(os.getenv(k) for k in need):
        print("Skipping YouTube upload (build-only or secrets missing). Download the artifact instead."); return
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    creds = Credentials(None, refresh_token=os.environ["YT_REFRESH_TOKEN"], token_uri="https://oauth2.googleapis.com/token",
                        client_id=os.environ["YT_CLIENT_ID"], client_secret=os.environ["YT_CLIENT_SECRET"], scopes=["https://www.googleapis.com/auth/youtube.upload"])
    yt = build("youtube", "v3", credentials=creds)
    body = {"snippet": {"title": meta["title"], "description": meta["description"], "tags": meta["tags"], "categoryId": "26", "defaultLanguage": "en"},
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False}}
    req = yt.videos().insert(part="snippet,status", body=body, media_body=MediaFileUpload(str(video), chunksize=8 * 1024 * 1024, resumable=True))
    resp = None
    while resp is None:
        st, resp = req.next_chunk()
        if st: print(f"uploaded {int(st.progress() * 100)}%")
    vid = resp["id"]; print("Video:", f"https://youtu.be/{vid}")
    try:
        yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(thumb))).execute(); print("Thumbnail set")
    except Exception as e:
        print("Thumbnail failed (is your channel phone-verified?):", e)


def main(jobfile):
    job = json.load(open(jobfile)); OUT.mkdir(exist_ok=True); TMP.mkdir(exist_ok=True)
    voice = asyncio.run(voiceover(job)); dur = len(voice) / SR
    mix = voice + duck(music(job.get("music", "calm"), dur), voice); write_wav(TMP / "mix.wav", np.clip(mix, -1, 1))
    print(f"audio {dur/60:.1f} min")
    theme = job.get("theme") if job.get("theme") in THEMES else "dreamstate"
    n = max(3, min(24, round(dur / 75))); subjects = THEMES[theme]; imgs = []
    for i in range(n):
        p = TMP / f"img{i}.jpg"; seed = random.randint(1, 10**6)
        # spread the scenes across the emotional arc: opening ... deepest moment ... quiet close
        subj = subjects[min(int(i / max(n - 1, 1) * (len(subjects) - 1)), len(subjects) - 1)] if n <= len(subjects) else subjects[i % len(subjects)]
        get_image(subj, seed, p); grade(Image.open(p)).save(p, quality=93); imgs.append(p)
    build_video(imgs, dur, TMP / "mix.wav", OUT / "final.mp4")
    meta = metadata(job)
    make_thumbnail(job, job.get("title") or meta["title"].split(" | ")[0], imgs[0], OUT / "thumbnail.jpg", random.randint(1, 10**6))
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=1))
    upload(OUT / "final.mp4", OUT / "thumbnail.jpg", meta, job.get("privacy", "private"))


if __name__ == "__main__":
    main(sys.argv[1])
