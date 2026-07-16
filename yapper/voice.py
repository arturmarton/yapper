"""Text -> her voice.

Loads the Base model once and clones voice.wav for every chunk. VoiceDesign is
never loaded here: it samples a new woman on every call, which would make the
narrator change identity mid-story. See design_voice.py.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import fade_edges, join_with_pauses, split_for_tts, stretch


@dataclass
class Chunk:
    """Rendered speech, still carrying the words it came from.

    The text travels with the audio so the console can print a paragraph when
    she says it rather than when she writes it -- she is several paragraphs
    ahead by then, and printing at write time made the text race the voice.
    """

    pcm: np.ndarray
    sr: int
    text: str


class Voice:
    def __init__(self, cfg):
        import torch
        from qwen_tts import Qwen3TTSModel

        self.cfg = cfg
        self.ref = cfg.voice.reference
        if not Path(self.ref).exists():
            raise FileNotFoundError(
                f"No voice at {self.ref}. She hasn't been created yet:\n"
                f"    ./design-voice generate\n"
                f"    ./design-voice choose <n>"
            )

        # The clone is noticeably better when it knows what the reference says.
        ref_txt = Path(self.ref).with_suffix(".txt")
        self.ref_text = ref_txt.read_text().strip() if ref_txt.exists() else None

        print(f"Loading voice from {cfg.voice.model_dir}...")
        self.model = Qwen3TTSModel.from_pretrained(
            cfg.voice.model_dir,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    def render(self, text):
        """Render one story to a Chunk. None if there's nothing to say.

        Rendered a few sentences at a time rather than all at once: long inputs
        degrade, and every call clones the same reference so the joins are
        seamless.
        """
        groups = split_for_tts(text, self.cfg.voice.max_words_per_call)
        if not groups:
            return None

        wavs, sr = self.model.generate_voice_clone(
            text=groups,
            ref_audio=[self.ref] * len(groups),
            ref_text=[self.ref_text] * len(groups) if self.ref_text else None,
            language=self.cfg.voice.language,
        )

        p = self.cfg.pacing
        segments = [
            fade_edges(stretch(w, sr, p.tempo), p.fade_ms, sr) for w in wavs
        ]
        pcm = join_with_pauses(
            segments, pause=p.sentence_pause, sr=sr, tail=p.paragraph_pause
        )
        return Chunk(pcm=pcm, sr=sr, text=text)
