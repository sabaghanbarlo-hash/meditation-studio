# Meditation Studio

Paste a meditation script on the website (`/docs`, served by GitHub Pages). It commits a job file to `jobs/`, which triggers the GitHub Actions workflow. The workflow builds the voiceover (edge-tts), ambient music (generated with numpy), images (Pollinations, with a generated-gradient fallback), a thumbnail (Pillow), and an MP4 (FFmpeg). It then uploads to YouTube with the title, description, hashtags, and tags.

Script tips: blank line = long pause, `[pause 10]` = 10 second pause.

Secrets (Settings > Secrets > Actions): `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN`; optional `GEMINI_API_KEY` for smarter titles/descriptions.

Every run also saves the MP4, thumbnail, and metadata as a downloadable artifact.
