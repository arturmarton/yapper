"""Three stages, two bounded queues, and the promise that she never stops.

    writer --text_q--> voice --audio_q--> player

The queues are the whole trick. The writer blocks once text_depth stories are
buffered and the voice blocks once audio_depth are rendered, so generation always
runs ahead of playback. A stall upstream -- a slow generation, a retry, an
exception -- is inaudible until it outlasts the rendered audio in hand (~90s at
the shipped depth of 1).

Bounded is as important as buffered: unbounded, the writer would run away and
render hours of speech nobody will hear.

Every stage swallows its own exceptions. No single failure may kill the loop --
that is the whole requirement, and it is why nothing here raises.
"""

import queue
import threading


class Pipeline:
    def __init__(self, writer, voice, player, text_depth=2, audio_depth=2,
                 retry_base=1.0, retry_max=30.0, log=lambda *a: None):
        self.writer, self.voice, self.player = writer, voice, player
        self.text_q = queue.Queue(maxsize=text_depth)
        self.audio_q = queue.Queue(maxsize=audio_depth)
        self.retry_base, self.retry_max = retry_base, retry_max
        self.log = log
        self._stop = threading.Event()
        self.threads = []

    # Queue ops that stay responsive to stop. A bare put()/get() on a full or
    # empty queue blocks forever and would hang shutdown.
    def _put(self, q, item):
        while not self._stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _get(self, q):
        while not self._stop.is_set():
            try:
                return q.get(timeout=0.1)
            except queue.Empty:
                continue
        return None

    def _sleep(self, seconds):
        """Interruptible backoff."""
        self._stop.wait(seconds)

    def _run(self, name, step):
        """Run one stage forever, absorbing its failures."""
        backoff = self.retry_base
        while not self._stop.is_set():
            try:
                if step():
                    backoff = self.retry_base
            except Exception as e:
                self.log(f"[{name}] {type(e).__name__}: {e} (retry in {backoff:.0f}s)")
                self._sleep(backoff)
                backoff = min(backoff * 2, self.retry_max)

    def _write_step(self):
        text = self.writer.next_paragraph()
        if not text or not text.strip():
            self.log("[writer] nothing usable, regenerating")
            return False  # don't render silence; ask again
        return self._put(self.text_q, text)

    def _voice_step(self):
        text = self._get(self.text_q)
        if text is None:
            return False
        try:
            audio = self.voice.render(text)
        except Exception as e:
            # Skip the chunk rather than stall the voice. One lost paragraph is
            # inaudible; a dead stage is the end of the session.
            self.log(f"[voice] dropped a chunk: {type(e).__name__}: {e}")
            return False
        if audio is None:
            return False
        return self._put(self.audio_q, audio)

    def _play_step(self):
        audio = self._get(self.audio_q)
        if audio is None:
            return False
        self.player.play(audio)
        return True

    def start(self):
        stages = [
            ("writer", self._write_step),
            ("voice", self._voice_step),
            ("player", self._play_step),
        ]
        for name, step in stages:
            t = threading.Thread(target=self._run, args=(name, step),
                                 name=name, daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self):
        self._stop.set()
        for t in self.threads:
            t.join(timeout=2.0)
