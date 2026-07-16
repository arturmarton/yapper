"""The endless supply of stories.

Sliding window only: the last N paragraphs go back verbatim and everything older
is dropped, so context stays flat no matter how long she runs. She forgets names
after ~10 minutes and quietly contradicts herself, which is the accepted trade --
each story stands alone anyway, and the forgetting is what stops her retelling
the same one.
"""

from collections import deque

import requests

from .sanitize import sanitize, trim_to_last_sentence


class Writer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.window = deque(maxlen=cfg.writer.window_paragraphs)
        self.nudges = list(cfg.writer.drift_nudges)
        self.told = 0

    def _messages(self):
        """Replay the window as a conversation.

        It has to be a conversation: this model's template rejects consecutive
        same-role turns outright ("roles must alternate user/assistant/..."), so
        posting the window as a block of assistant turns 400s every request.
        """
        msgs = [
            {"role": "system", "content": self.cfg.writer.system_prompt},
            {"role": "user", "content": self.cfg.writer.opening},
        ]
        if not self.window:
            return msgs

        for i, para in enumerate(self.window):
            msgs.append({"role": "assistant", "content": para})
            if i < len(self.window) - 1:
                msgs.append({"role": "user", "content": "Continue."})

        # Rotate a nudge in. With only a 2-paragraph window she otherwise keeps
        # telling variations of the story she can still see.
        nudge = self.nudges[self.told % len(self.nudges)]
        msgs.append({"role": "user", "content": f"Continue. {nudge}"})
        return msgs

    def next_paragraph(self):
        """Fetch, clean and trim one story. Returns "" if nothing usable came back."""
        c = self.cfg.llm
        r = requests.post(
            f"{c.server_url}/v1/chat/completions",
            json={
                "messages": self._messages(),
                "temperature": c.temperature,
                "top_p": c.top_p,
                "repeat_penalty": c.repeat_penalty,
                "repeat_last_n": c.repeat_last_n,
                "max_tokens": c.max_tokens,
            },
            timeout=c.request_timeout,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]

        # Both are load-bearing, not defensive: the model reliably runs out of
        # tokens mid-clause, and reliably emits asterisks the prompt forbids.
        text = trim_to_last_sentence(sanitize(raw))
        if not text:
            return ""

        self.window.append(text)
        self.told += 1
        return text
