# yapper

![yapper](assets/banner.png)

![version](https://img.shields.io/badge/version-2.0.1-b07b8d.svg?style=flat-square&labelColor=374151) ![runs local](https://img.shields.io/badge/runs-100%25_local-2e1a47?style=flat-square) ![writer: llama.cpp](https://img.shields.io/badge/writer-llama.cpp-9e365c?style=flat-square) ![voice](https://img.shields.io/badge/voice-Qwen3--TTS-c1573f?style=flat-square) ![vram](https://img.shields.io/badge/VRAM-~16_GB-e88a4a?style=flat-square)

She tells made-up stories forever: a man burping through a film, a kitten behind a coffee shop, bread that became a cat.

> **Oh my gosh, okay wait,** so after the ferret incident, I took the scenic route home. Big mistake. Huge.

Start her and she talks until you stop her: always mid-story, always delighted, always the same voice.

```bash
./yap
```

That's it. Ctrl-C when you've had enough.

---

## How she works

Two models, running at once on one GPU. One writes, one speaks.

```
   writes  ─────►  speaks  ─────►  you hear it
   (LLM)           (TTS)
     ▲                ▲                ▲
   story 6         story 5          story 4
```

While you hear one story, the next is recorded and another is being written. That buffer keeps her talking through recovery; at worst, she skips one story.

## Who is she

`voice.wav` is her. Back it up once you have one you like.

The TTS invents her once from a voice description, then copies that sample for every story. `./design-voice` generates the gitignored `voice.wav`; making a new one means meeting someone else.

## Changing her

**`config.toml`** defines her character and topics. **`.env`** points to the models.

Keep the prompt's `NEVER` rules: they stop her from drifting into melodrama, scene description, or present-tense narration. Change `tempo` for speaking speed, not the voice description.

For a new voice, aim for roughly 200–240 Hz for a woman in her twenties. `design-voice generate` prints pitch and speed so you can discard unsuitable candidates quickly.

## Commands

### Listening

```bash
./yap                  # forever
./yap --seconds 90     # just a taste
./yap --quiet          # don't print what she's saying
```

She starts `llama-server` if needed (about 30 seconds) and leaves it running for quick restarts. Run `pkill llama-server` to free the 10 GB afterward.

### Making a new voice

Only if you want a different woman. This replaces her permanently.

```bash
./design-voice generate     # invent 5 candidates, each a different person
./design-voice play 3       # listen to one
./design-voice choose 3     # keep her
```

`generate` prints pitch and speed for each candidate. It creates a new person every time, so choose one and she will reuse that recording.

### Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

61 tests, about three seconds, no models required.

### Setting it up from scratch

```bash
cp .env.example .env      # then point it at your models
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
scripts/build-llama.sh    # llama.cpp with CUDA
./yap
```

You also need `ffmpeg` and PipeWire. Use the virtualenv: `qwen-tts` requires its own Torch and Transformers versions.
