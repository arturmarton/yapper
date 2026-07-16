"""Pure audio logic: chunk splitting and PCM assembly.

Measured: the clone speaks ~200 wpm regardless of the reference, so pauses do
the real work of making her unobtrusive. That makes this module load-bearing,
not cosmetic.
"""

import numpy as np
import pytest

from yapper.audio import apply_gain, fade_edges, join_with_pauses, split_for_tts

SR = 24000


class TestSplitForTTS:
    def test_keeps_short_paragraph_whole(self):
        text = "The kettle boiled. The room was quiet."
        assert split_for_tts(text, max_words=40) == [text]

    def test_splits_on_sentence_boundary_when_over_budget(self):
        text = " ".join([f"word{i}" for i in range(30)]) + ". " + \
               " ".join([f"other{i}" for i in range(30)]) + "."
        parts = split_for_tts(text, max_words=40)
        assert len(parts) == 2
        assert parts[0].endswith(".")
        assert parts[1].startswith("other0")

    def test_groups_several_short_sentences_together(self):
        text = "One. Two. Three. Four."
        assert split_for_tts(text, max_words=40) == ["One. Two. Three. Four."]

    def test_never_exceeds_budget_except_for_one_long_sentence(self):
        text = " ".join(f"w{i}" for i in range(100)) + "."
        parts = split_for_tts(text, max_words=40)
        # A single 100-word sentence can't be split on sentence boundaries;
        # it must still come back rather than vanish.
        assert len(parts) == 1
        assert parts[0].startswith("w0")

    def test_empty_returns_no_chunks(self):
        assert split_for_tts("", max_words=40) == []
        assert split_for_tts("   ", max_words=40) == []


class TestJoinWithPauses:
    def test_inserts_silence_between_segments(self):
        a = np.ones(SR, dtype=np.float32)
        b = np.ones(SR, dtype=np.float32)
        out = join_with_pauses([a, b], pause=0.5, sr=SR, tail=0.0)
        assert len(out) == SR + int(0.5 * SR) + SR

    def test_silence_is_actually_silent(self):
        a = np.ones(100, dtype=np.float32)
        out = join_with_pauses([a, a], pause=1.0, sr=100, tail=0.0)
        assert np.all(out[100:200] == 0.0)

    def test_appends_tail_silence(self):
        a = np.ones(100, dtype=np.float32)
        out = join_with_pauses([a], pause=0.0, sr=100, tail=1.0)
        assert len(out) == 200
        assert np.all(out[100:] == 0.0)

    def test_single_segment_gets_no_leading_pause(self):
        a = np.ones(100, dtype=np.float32)
        out = join_with_pauses([a], pause=1.0, sr=100, tail=0.0)
        assert len(out) == 100

    def test_empty_list_returns_empty(self):
        assert len(join_with_pauses([], pause=1.0, sr=100, tail=0.0)) == 0


class TestGain:
    def test_scales_amplitude(self):
        a = np.ones(10, dtype=np.float32)
        assert np.allclose(apply_gain(a, 0.5), 0.5)

    def test_clips_to_valid_range(self):
        a = np.full(10, 0.9, dtype=np.float32)
        assert np.max(apply_gain(a, 4.0)) <= 1.0


class TestFade:
    def test_starts_and_ends_at_silence(self):
        a = np.ones(SR, dtype=np.float32)
        out = fade_edges(a, ms=10, sr=SR)
        assert out[0] == pytest.approx(0.0, abs=1e-6)
        assert out[-1] == pytest.approx(0.0, abs=1e-6)

    def test_leaves_the_middle_alone(self):
        a = np.ones(SR, dtype=np.float32)
        out = fade_edges(a, ms=10, sr=SR)
        assert out[SR // 2] == pytest.approx(1.0)

    def test_handles_segment_shorter_than_fade(self):
        a = np.ones(5, dtype=np.float32)
        out = fade_edges(a, ms=100, sr=SR)  # fade longer than the audio
        assert len(out) == 5
        assert np.all(np.isfinite(out))
