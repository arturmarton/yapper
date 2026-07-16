"""Chunk splitting and PCM assembly.

Everything here is pure numpy except `stretch`, which shells out to ffmpeg. That
keeps the splitting and assembly testable without a sound card or a model.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def split_for_tts(text, max_words):
    """Group text into TTS-sized chunks, breaking only at sentence ends.

    A single sentence longer than max_words is returned whole rather than cut
    mid-clause: an over-long chunk is a quality risk, a severed one is a defect.
    """
    text = text.strip()
    if not text:
        return []

    chunks, current, count = [], [], 0
    for sentence in _SENTENCE.split(text):
        n = len(sentence.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def join_with_pauses(segments, pause, sr, tail=0.0):
    """Concatenate segments with silence between them, plus optional tail silence.

    The gaps are where she breathes between sentences, and the tail is the beat
    between one story and the next.
    """
    if not segments:
        return np.zeros(0, dtype=np.float32)

    gap = np.zeros(int(pause * sr), dtype=np.float32)
    out = []
    for i, seg in enumerate(segments):
        if i:
            out.append(gap)
        out.append(np.asarray(seg, dtype=np.float32))
    if tail:
        out.append(np.zeros(int(tail * sr), dtype=np.float32))
    return np.concatenate(out)


def apply_gain(pcm, gain):
    return np.clip(np.asarray(pcm, dtype=np.float32) * gain, -1.0, 1.0)


def fade_edges(pcm, ms, sr):
    """Fade each segment in and out to kill clicks at the seams."""
    pcm = np.asarray(pcm, dtype=np.float32).copy()
    n = min(int(ms / 1000 * sr), len(pcm) // 2)
    if n < 1:
        return pcm
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    pcm[:n] *= ramp
    pcm[-n:] *= ramp[::-1]
    return pcm


def stretch(pcm, sr, tempo):
    """Change speech rate without changing pitch. tempo < 1 slows, > 1 speeds up.

    Rubberband, not atempo: the ratios worth using are far enough from 1.0 that
    atempo's phase-vocoder artifacts are audible.

    Currently a no-op -- config ships tempo = 1.0, her natural rate -- but it is
    the only reliable speed control, since the voice reference only partly
    carries pacing. Kept for that.
    """
    if abs(tempo - 1.0) < 1e-3 or len(pcm) == 0:
        return np.asarray(pcm, dtype=np.float32)
    if not shutil.which("ffmpeg"):
        return np.asarray(pcm, dtype=np.float32)

    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "in.wav", Path(d) / "out.wav"
        sf.write(src, pcm, sr)
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-filter:a", f"rubberband=tempo={tempo}:pitchq=quality", str(dst)],
            capture_output=True,
        )
        if proc.returncode != 0 or not dst.exists():
            # Never let a filter failure break the loop; unstretched speech is
            # worse than intended but still speech.
            return np.asarray(pcm, dtype=np.float32)
        out, _ = sf.read(dst, dtype="float32")
    return out
