"""The pipeline's job is that the voice never stops.

These tests use fake stages, so they prove the concurrency and recovery
behaviour with no models loaded and no audio device.
"""

import threading
import time

import pytest

from yapper.pipeline import Pipeline

TIMEOUT = 5.0


class FakeWriter:
    """Emits paragraph 0, 1, 2, ... and counts calls."""

    def __init__(self, texts=None, fail_times=0):
        self.texts = texts
        self.calls = 0
        self.fail_times = fail_times
        self.lock = threading.Lock()

    def next_paragraph(self):
        with self.lock:
            n = self.calls
            self.calls += 1
        if n < self.fail_times:
            raise RuntimeError(f"writer boom {n}")
        if self.texts is not None:
            return self.texts[n % len(self.texts)]
        return f"paragraph {n}"


class FakeVoice:
    """Renders text to a fake 'audio' string, optionally exploding on some."""

    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.rendered = []
        self.lock = threading.Lock()

    def render(self, text):
        if text in self.fail_on:
            raise RuntimeError(f"tts boom on {text!r}")
        with self.lock:
            self.rendered.append(text)
        return f"audio<{text}>"


class FakePlayer:
    """Records what it played. Can be held to simulate slow playback."""

    def __init__(self, gate=None):
        self.played = []
        self.gate = gate
        self.lock = threading.Lock()
        self.first_play = threading.Event()

    def play(self, audio):
        self.first_play.set()
        if self.gate is not None:
            self.gate.wait(TIMEOUT)
        with self.lock:
            self.played.append(audio)


def wait_until(predicate, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def stop_pipelines():
    started = []
    yield started
    for p in started:
        p.stop()


class TestOrder:
    def test_paragraphs_reach_the_player_in_order(self, stop_pipelines):
        writer, voice, player = FakeWriter(), FakeVoice(), FakePlayer()
        p = Pipeline(writer, voice, player)
        stop_pipelines.append(p)
        p.start()

        assert wait_until(lambda: len(player.played) >= 3)
        p.stop()

        assert player.played[:3] == [
            "audio<paragraph 0>",
            "audio<paragraph 1>",
            "audio<paragraph 2>",
        ]


class TestRunsAhead:
    """The whole point: generation must run ahead of playback."""

    def test_writer_runs_ahead_while_player_is_busy(self, stop_pipelines):
        gate = threading.Event()  # player is stuck on the first paragraph
        writer, voice, player = FakeWriter(), FakeVoice(), FakePlayer(gate=gate)
        p = Pipeline(writer, voice, player, text_depth=2, audio_depth=2)
        stop_pipelines.append(p)
        p.start()

        assert player.first_play.wait(TIMEOUT), "player never started"

        # While playback is blocked on paragraph 0, the writer and voice must
        # keep working ahead and fill both buffers.
        assert wait_until(lambda: len(voice.rendered) >= 3), (
            f"voice only rendered {len(voice.rendered)} while player was busy; "
            "generation is not running ahead of playback"
        )

        gate.set()
        assert wait_until(lambda: len(player.played) >= 3)

    def test_backpressure_stops_the_writer_running_away(self, stop_pipelines):
        gate = threading.Event()
        writer, voice, player = FakeWriter(), FakeVoice(), FakePlayer(gate=gate)
        p = Pipeline(writer, voice, player, text_depth=2, audio_depth=2)
        stop_pipelines.append(p)
        p.start()

        assert player.first_play.wait(TIMEOUT)
        time.sleep(0.5)  # let it fill and settle

        # Bounded queues must cap how far ahead it runs, or it would generate
        # forever and eat all RAM during a long pause.
        assert writer.calls < 12, (
            f"writer made {writer.calls} calls while playback was blocked; "
            "backpressure is not holding"
        )
        gate.set()


class TestRecovery:
    def test_writer_exception_does_not_kill_the_loop(self, stop_pipelines):
        writer = FakeWriter(fail_times=2)  # first two attempts explode
        voice, player = FakeVoice(), FakePlayer()
        p = Pipeline(writer, voice, player, retry_base=0.01)
        stop_pipelines.append(p)
        p.start()

        assert wait_until(lambda: len(player.played) >= 2), (
            "loop died after writer exceptions"
        )

    def test_voice_exception_skips_chunk_and_continues(self, stop_pipelines):
        writer = FakeWriter(texts=["good one", "poison", "good two"])
        voice = FakeVoice(fail_on=["poison"])
        player = FakePlayer()
        p = Pipeline(writer, voice, player, retry_base=0.01)
        stop_pipelines.append(p)
        p.start()

        assert wait_until(lambda: len(player.played) >= 2)
        p.stop()

        assert "audio<poison>" not in player.played
        assert "audio<good one>" in player.played
        assert "audio<good two>" in player.played

    def test_empty_paragraph_is_regenerated_not_played(self, stop_pipelines):
        writer = FakeWriter(texts=["", "   ", "real text"])
        voice, player = FakeVoice(), FakePlayer()
        p = Pipeline(writer, voice, player, retry_base=0.01)
        stop_pipelines.append(p)
        p.start()

        assert wait_until(lambda: len(player.played) >= 1)
        p.stop()

        assert all("real text" in a for a in player.played)
        assert voice.rendered  # never asked to render whitespace
        assert all(t.strip() for t in voice.rendered)


class TestShutdown:
    def test_stop_terminates_all_threads(self, stop_pipelines):
        writer, voice, player = FakeWriter(), FakeVoice(), FakePlayer()
        p = Pipeline(writer, voice, player)
        stop_pipelines.append(p)
        p.start()
        assert wait_until(lambda: len(player.played) >= 1)

        p.stop()

        assert wait_until(lambda: not any(t.is_alive() for t in p.threads)), (
            "threads still alive after stop; a blocked queue put/get is hanging them"
        )

    def test_stop_is_safe_when_player_is_blocked(self, stop_pipelines):
        """Buffers are full and every stage is blocked on a queue. Stop must
        still return rather than deadlock."""
        gate = threading.Event()
        writer, voice, player = FakeWriter(), FakeVoice(), FakePlayer(gate=gate)
        p = Pipeline(writer, voice, player, text_depth=1, audio_depth=1)
        stop_pipelines.append(p)
        p.start()
        assert player.first_play.wait(TIMEOUT)
        time.sleep(0.3)  # everything jams up behind the blocked player

        p.stop()
        gate.set()

        assert wait_until(lambda: not any(t.is_alive() for t in p.threads))
