"""Meditation video, animated edition.

Same voiceover, music, thumbnail, metadata and YouTube upload as generate.py, but the visuals are procedurally
animated scenes (rain, river, forest, ocean, aurora, lanterns) drawn by motion.py instead of AI still images.
No image service is called, so there are no watermarks and nothing to run out of quota.

Optional job-file keys:  "scenes": ["rain", "river", "forest", "ocean", "aurora", "lanterns"]  (order to use)
                         "theme": dreamstate | cosmos | ocean | nature | zen   (picks a default scene order)
"""
import asyncio, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import generate as G
import motion


def main(jobfile):
    job = json.load(open(jobfile)); G.OUT.mkdir(exist_ok=True); G.TMP.mkdir(exist_ok=True)
    voice = asyncio.run(G.voiceover(job)); n = len(voice); dur = n / G.SR
    # music is built from float seconds and can come out 1 sample short of the voice: ask for a little extra, cut to exact length
    music = G.music(job.get("music", "calm"), dur + 1.0)[:n]
    if len(music) < n:
        music = np.pad(music, (0, n - len(music)))
    mix = voice + G.duck(music, voice)
    G.write_wav(G.TMP / "mix.wav", np.clip(mix, -1, 1))
    print(f"audio {dur/60:.1f} min")
    theme = job.get("theme") if job.get("theme") in G.THEMES else "dreamstate"
    n_scenes = max(3, min(24, round(dur / 75)))
    motion.build_video(theme, n_scenes, dur, G.TMP / "mix.wav", G.OUT / "final.mp4", G.TMP, job)
    meta = G.metadata(job)
    title = job.get("title") or (meta["title"].split(" | ")[0] if G.os.getenv("GEMINI_API_KEY") else "")
    G.make_thumbnail(job, title, None, G.OUT / "thumbnail.jpg", G.random.randint(1, 10 ** 6))
    (G.OUT / "metadata.json").write_text(json.dumps(meta, indent=1))
    G.upload(G.OUT / "final.mp4", G.OUT / "thumbnail.jpg", meta, job.get("privacy", "private"))


if __name__ == "__main__":
    main(sys.argv[1])
