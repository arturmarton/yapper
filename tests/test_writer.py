"""Message construction for the sliding window.

Regression: the first paragraph worked and every continuation returned HTTP 400.
This model's chat template raises "After the optional system message,
conversation roles must alternate user/assistant/user/assistant" -- feeding the
window back as consecutive assistant turns is rejected outright.
"""

from yapper.config import load
from yapper.writer import Writer


def make_writer():
    return Writer(load())  # __init__ makes no network calls


class TestMessageRoles:
    def test_opening_has_no_history(self):
        msgs = make_writer()._messages()
        assert [m["role"] for m in msgs] == ["system", "user"]

    def test_roles_alternate_strictly_after_system(self):
        w = make_writer()
        w.window.append("first paragraph")
        w.window.append("second paragraph")

        roles = [m["role"] for m in w._messages()]

        assert roles[0] == "system"
        rest = roles[1:]
        assert rest[0] == "user", "history must not start with an assistant turn"
        assert rest[-1] == "user", "the model must be left to speak next"
        for a, b in zip(rest, rest[1:]):
            assert a != b, f"consecutive {a} turns are rejected by the template"

    def test_window_paragraphs_are_all_present(self):
        w = make_writer()
        w.window.append("alpha paragraph")
        w.window.append("beta paragraph")

        content = [m["content"] for m in w._messages()]

        assert "alpha paragraph" in content
        assert "beta paragraph" in content

    def test_single_paragraph_window_alternates(self):
        w = make_writer()
        w.window.append("only paragraph")

        roles = [m["role"] for m in w._messages()]

        for a, b in zip(roles[1:], roles[2:]):
            assert a != b


class TestDrift:
    def test_nudge_is_included_and_rotates(self):
        w = make_writer()
        w.window.append("a paragraph")

        w.told = 0
        first = w._messages()[-1]["content"]
        w.told = 1
        second = w._messages()[-1]["content"]

        assert first != second, "the same nudge every turn defeats the purpose"
        assert w.cfg.writer.drift_nudges[0] in first
        assert w.cfg.writer.drift_nudges[1] in second


class TestSlidingWindow:
    def test_window_never_grows_past_its_limit(self):
        w = make_writer()
        for i in range(20):
            w.window.append(f"paragraph {i}")

        assert len(w.window) == w.cfg.writer.window_paragraphs
        assert "paragraph 19" in list(w.window)[-1]
