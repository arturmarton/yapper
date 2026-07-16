"""Playback.

Writes in small blocks rather than sd.play()/sd.wait() so that Ctrl-C lands
immediately. A blocking whole-paragraph write would leave the process ignoring
you for up to a minute and a half.
"""

import numpy as np

from .audio import apply_gain

BLOCK = 2048


class Player:
    def __init__(self, cfg, stop_event=None):
        import sounddevice as sd

        self.sd = sd
        self.gain = cfg.pacing.gain
        self.stop_event = stop_event
        self.stream = None
        self.sr = None

    def _ensure_stream(self, sr):
        if self.stream is not None and self.sr == sr:
            return
        if self.stream is not None:
            self.stream.close()
        self.stream = self.sd.OutputStream(
            samplerate=sr, channels=1, dtype="float32", blocksize=BLOCK
        )
        self.stream.start()
        self.sr = sr

    def play(self, chunk):
        self._ensure_stream(chunk.sr)
        pcm = apply_gain(chunk.pcm, self.gain).astype(np.float32)

        for i in range(0, len(pcm), BLOCK):
            if self.stop_event is not None and self.stop_event.is_set():
                return
            self.stream.write(pcm[i : i + BLOCK])

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
