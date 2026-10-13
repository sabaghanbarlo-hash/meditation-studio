import asyncio, json, os, random, re, subprocess, sys, urllib.parse, urllib.request, io
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SR, W, H, FPS = 44100, 1920, 1080, 24
OUT, TMP = Path("out"), Path("tmp")

VOICES = {"jenny": "en-US-JennyNeural", "aria": "en-US-AriaNeural", "sonia": "en-GB-SoniaNeural",
          "natasha": "en-AU-NatashaNeural", "guy": "en-US-GuyNeural", "ryan": "en-GB-RyanNeural"}
THEMES = {
    "nature": ["misty mountain lake at dawn", "sunlit forest path", "quiet meadow with wildflowers", "calm river through green hills", "golden hour over rolling hills", "still pond with lotus flowers"],
    "cosmos": ["soft nebula in deep blue and purple", "starry night sky over a quiet lake", "gentle aurora over snowy hills", "glowing galaxy swirl", "crescent moon above calm clouds", "milky way over desert dunes"],
    "ocean": ["slow gentle waves at sunrise", "quiet beach at dusk", "turquoise lagoon from above", "moonlit calm sea", "soft surf on pale sand", "sea mist at dawn"],
    "zen": ["zen garden with raked sand and stones", "bamboo grove with soft light", "candlelit stone altar with incense", "misty japanese temple garden", "smooth stacked stones by water", "paper lanterns glowing softly"],
}
TINTS = {"nature": (60, 120, 90), "cosmos": (60, 50, 130), "ocean": (40, 110, 140), "zen": (140, 120, 90)}


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


# ---------- 1. voiceover ----------
def plan(script):
    script = re.sub(r"\[pause\s*(\d+)\s*s?\]", r"\n\n@@\1@@\n\n", script)
    items = []
    for para in re.split(r"\n\s*\n", script.strip()):
        para = " ".join(para.split())
        m = re.fullmatch(r"@@(\d+)@@", para)
        if m:
            items.append(("gap", float(m.group(1)))); continue
        for s in re.split(r"(?<=[.!?\u2026])\s+", para):
            if s.strip():
                items += [("say", s.strip()), ("gap", 1.1)]
        items.append(("gap", 2.5))
    return items


async def synth(text, voice, path):
    import edge_tts
    for attempt in range(4):
        try:
            await edge_tts.Communicate(text, voice, rate="-18%", pitch="-4Hz").save(str(path)); return
        except Exception as e:
            print("tts retry", attempt, e); await asyncio.sleep(2 + attempt * 3)
    raise RuntimeError("edge-tts failed repeatedly")


async def voiceover(job):
    voice = VOICES.get(job.get("voice", "jenny"), VOICES["jenny"])
    parts, n = [silence(3)], 0
    for kind, val in plan(job["script"]):
        if kind == "gap":
            parts.append(silence(val)); continue
        p = TMP / f"s{n}.mp3"; n += 1
        await synth(val, voice, p)
        parts.append(pcm(p))
    parts.append(silence(6))
    v = np.concatenate(parts)
    return v / max(np.abs(v).max(), 1e-6) * 0.9


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


# ---------- 3. images ----------
def gradient(theme, seed):
    rng = np.random.default_rng(seed); base = np.array(TINTS[theme])
    small = (rng.random((5, 8, 3)) * 0.9 + 0.3) * base
    img = Image.fromarray(np.clip(small, 0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    return img.filter(ImageFilter.GaussianBlur(40))


def get_image(prompt, theme, seed, path):
    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width={W}&height={H}&nologo=true&seed={seed}"
    for attempt in range(3):
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=90).read()
            img = Image.open(io.BytesIO(data)).convert("RGB").resize((W, H)); img.save(path, quality=92); return
        except Exception as e:
            print("image retry", attempt, e); import time; time.sleep(5 + attempt * 5)
    print("using gradient fallback"); gradient(theme, seed).save(path, quality=92)


# ---------- 4. video ----------
def build_video(images, dur, audio, out):
    d = dur / len(images); frames = round(d * FPS); clips = []
    for i, img in enumerate(images):
        z = f"1+0.15*on/{frames}" if i % 2 == 0 else f"1.15-0.15*on/{frames}"
        vf = (f"scale=3840:-2,zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}:fps={FPS},"
              f"fade=t=in:st=0:d=2,fade=t=out:st={d-2:.2f}:d=2,format=yuv420p")
        c = TMP / f"clip{i}.mp4"
        ff("-i", img, "-vf", vf, "-frames:v", frames, "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", c)
        clips.append(c)
    lst = TMP / "list.txt"; lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    ff("-f", "concat", "-safe", "0", "-i", lst, "-i", audio, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out)


# ---------- 5. thumbnail ----------
def thumbnail(img_path, title, out):
    img = Image.open(img_path).convert("RGB").resize((1280, 720))
    img = Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), 0.42)
    d = ImageDraw.Draw(img); words, lines, cur = title.upper().split(), [], ""
    for w in words:
        if len(cur) + len(w) > 16 and cur: lines.append(cur); cur = w
        else: cur = (cur + " " + w).strip()
    lines.append(cur); lines = lines[:3]
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 110 if len(lines) < 3 else 90)
    except Exception: font = ImageFont.load_default()
    lh = 130 if len(lines) < 3 else 108; y = (720 - lh * len(lines)) // 2
    for ln in lines:
        x = (1280 - d.textlength(ln, font=font)) / 2
        d.text((x + 4, y + 4), ln, font=font, fill=(0, 0, 0)); d.text((x, y), ln, font=font, fill=(255, 255, 255)); y += lh
    img.save(out, quality=90)


# ---------- 6. metadata ----------
def metadata(job):
    script = " ".join(job["script"].split()); script = re.sub(r"\[pause[^\]]*\]", "", script)
    first = re.split(r"(?<=[.!?])\s+", script)[:2]
    title = (job.get("title") or first[0])[:70].strip(" .")
    tags = ["guided meditation", "meditation", "relaxation", "mindfulness", "stress relief", "calm", "sleep meditation", "inner peace"]
    meta = {"title": f"{title} | Guided Meditation"[:100],
            "description": " ".join(first) + "\n\nFind a comfortable position, press play, and let yourself relax.\n\nVoice and visuals in this video are AI-generated.\n\n#meditation #mindfulness #relaxation #guidedmeditation",
            "tags": tags}
    key = os.getenv("GEMINI_API_KEY")
    if key:
        try:
            prompt = ("Write YouTube metadata for this guided meditation script. Return JSON only with keys title (max 70 chars, calm and searchable), "
                      "description (150-250 words, ends with 4-6 hashtags), tags (list of 10-15 short strings).\n\nSCRIPT:\n" + script[:6000])
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
    mix = voice + music(job.get("music", "calm"), dur); write_wav(TMP / "mix.wav", np.clip(mix, -1, 1))
    print(f"audio {dur/60:.1f} min")
    theme = job.get("theme", "nature") if job.get("theme") in THEMES else "nature"
    n = max(3, min(24, round(dur / 75))); subjects = THEMES[theme]; imgs = []
    for i in range(n):
        p = TMP / f"img{i}.jpg"; seed = random.randint(1, 10**6)
        get_image(f"{subjects[i % len(subjects)]}, peaceful, soft light, no people, no text, cinematic, highly detailed", theme, seed, p); imgs.append(p)
    build_video(imgs, dur, TMP / "mix.wav", OUT / "final.mp4")
    meta = metadata(job); thumbnail(imgs[0], job.get("title") or meta["title"].split(" | ")[0], OUT / "thumbnail.jpg")
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=1))
    upload(OUT / "final.mp4", OUT / "thumbnail.jpg", meta, job.get("privacy", "private"))


if __name__ == "__main__":
    main(sys.argv[1])
