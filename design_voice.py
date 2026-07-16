#!/usr/bin/env python
"""Invent her, once.

VoiceDesign samples at temperature 0.9 and has no named speakers, so every call
invents a different woman. Calling it per chunk would make the narrator change
identity mid-story. So it runs exactly once, here: generate candidates, pick
one, save it as voice.wav. From then on the yapper clones that file and never
loads this model again.

    ./design-voice generate      # make candidates
    ./design-voice play 3        # hear one
    ./design-voice choose 3      # she's the one

Rerun to meet someone else.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from yapper.config import load

CANDIDATES = Path(__file__).resolve().parent / "candidates"

# Measured against real picks. Pitch decides whether she reads as a woman or a
# child, and it is the first thing to check -- asking for excitement drags it
# upward, so most candidates land too high.
PITCH_BANDS = [
    (0, 200, "older woman"),
    (200, 240, "woman in her twenties"),
    (240, 999, "reads as a child"),
]


def describe_pitch(hz):
    for lo, hi, label in PITCH_BANDS:
        if lo <= hz < hi:
            return label
    return "?"


def median_pitch(wav, sr):
    """Median F0 by autocorrelation, over voiced frames only.

    Rough, but it separates a woman from a child reliably, which is the only
    question being asked of it. Note this measures the *reference*; the clone
    lands near but not exactly on it.
    """
    import numpy as np

    if wav.ndim > 1:
        wav = wav.mean(1)
    frame, hop = int(0.04 * sr), int(0.02 * sr)
    lo, hi = int(sr / 350), int(sr / 70)  # search 70-350 Hz
    found = []
    for i in range(0, len(wav) - frame, hop):
        f = wav[i : i + frame]
        if np.sqrt(np.mean(f**2)) < 0.02:  # silence
            continue
        f = f - f.mean()
        ac = np.correlate(f, f, "full")[len(f) - 1 :]
        if ac[0] <= 0 or not len(ac[lo:hi]):
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] / ac[0] > 0.3:  # voiced
            found.append(sr / lag)
    return float(np.median(found)) if found else 0.0


def generate(cfg, count):
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    CANDIDATES.mkdir(exist_ok=True)
    for old in CANDIDATES.glob("voice_*.wav"):
        old.unlink()

    print(f"Loading VoiceDesign ({cfg.voice.design_model_dir})...", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        cfg.voice.design_model_dir,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # flash-attn skipped: we have VRAM to spare
    )

    print(f"Her description:\n  {cfg.voice.description}\n")
    print(f"Generating {count} candidates (each is a different woman)...", flush=True)

    # One batched call: same text N times, sampling makes each one different.
    wavs, sr = model.generate_voice_design(
        text=[cfg.voice.audition_text] * count,
        instruct=[cfg.voice.description] * count,
        language=cfg.voice.language,
    )

    words = len(cfg.voice.audition_text.split())
    print(f"\n  {'#':<3}{'pitch':>7}  {'speed':>8}   reads as")
    for i, wav in enumerate(wavs, 1):
        sf.write(CANDIDATES / f"voice_{i}.wav", wav, sr)
        hz = median_pitch(wav, sr)
        wpm = words / (len(wav) / sr) * 60
        print(f"  {i:<3}{hz:6.0f}Hz {wpm:6.0f}wpm   {describe_pitch(hz)}")

    print(f"\nListen:  ./design-voice play 1   ...through {count}")
    print("Keep:    ./design-voice choose 2")
    print("\nSkip the ones outside 200-240Hz; that's the difference between a")
    print("woman and a child. Regenerate if none land there.")


def play(index):
    path = CANDIDATES / f"voice_{index}.wav"
    if not path.exists():
        sys.exit(f"No candidate {index}. Run: ./design-voice generate")
    subprocess.run(["pw-play", str(path)], check=False)


def choose(cfg, index):
    path = CANDIDATES / f"voice_{index}.wav"
    if not path.exists():
        sys.exit(f"No candidate {index}. Run: ./design-voice generate")

    shutil.copy(path, cfg.voice.reference)
    # The clone is much better when it knows what the reference says.
    Path(cfg.voice.reference).with_suffix(".txt").write_text(cfg.voice.audition_text)

    print(f"Saved her to {cfg.voice.reference}")
    print("She will now sound the same in every chunk, and after every restart.")
    print("\nStart the yapper:  ./yap")


def main():
    ap = argparse.ArgumentParser(description="Design the narrator's voice, once.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate candidate voices")
    g.add_argument("-n", "--count", type=int, default=5)

    p = sub.add_parser("play", help="play a candidate")
    p.add_argument("index", type=int)

    c = sub.add_parser("choose", help="keep a candidate as voice.wav")
    c.add_argument("index", type=int)

    args = ap.parse_args()
    cfg = load()

    if args.cmd == "generate":
        generate(cfg, args.count)
    elif args.cmd == "play":
        play(args.index)
    elif args.cmd == "choose":
        choose(cfg, args.index)


if __name__ == "__main__":
    main()
