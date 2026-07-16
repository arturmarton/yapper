"""What the writer emits is not what the TTS can read aloud.

Forgotten-Safeword is a roleplay tune: it emits asterisk actions, markdown and
headings no matter what the system prompt says. The TTS pronounces every one of
those characters as a word. These tests pin the cleanup.
"""

from yapper.sanitize import sanitize


class TestRoleplayActions:
    def test_removes_asterisk_action(self):
        assert sanitize("*she sighs softly* The tide came in.") == "The tide came in."

    def test_removes_asterisk_action_at_end(self):
        assert sanitize("The tide came in. *she sighs*") == "The tide came in."

    def test_removes_multiple_asterisk_actions(self):
        text = "*pauses* The light moved. *smiles quietly* Then it was gone."
        assert sanitize(text) == "The light moved. Then it was gone."

    def test_keeps_bold_text_without_markers(self):
        assert sanitize("It was **very** quiet.") == "It was very quiet."

    def test_keeps_underscore_italic_text_without_markers(self):
        assert sanitize("It was _almost_ warm.") == "It was almost warm."

    def test_unpaired_asterisk_is_dropped(self):
        assert sanitize("The room was still*") == "The room was still"


class TestMarkdown:
    def test_drops_heading_line(self):
        assert sanitize("## Chapter Two\nThe rain stopped.") == "The rain stopped."

    def test_drops_bare_chapter_line(self):
        assert sanitize("Chapter Three\nThe rain stopped.") == "The rain stopped."

    def test_drops_list_markers(self):
        assert sanitize("- the kettle\n- the window") == "the kettle the window"

    def test_drops_blockquote_marker(self):
        assert sanitize("> she thought about it") == "she thought about it"


class TestTypography:
    def test_normalises_smart_quotes(self):
        assert sanitize("“Hello,” she said.") == '"Hello," she said.'

    def test_normalises_apostrophe(self):
        assert sanitize("It wasn’t cold.") == "It wasn't cold."

    def test_em_dash_becomes_comma_pause(self):
        assert sanitize("The door—it was open.") == "The door, it was open."

    def test_ellipsis_character_becomes_dots(self):
        assert sanitize("She waited…") == "She waited..."


class TestNoise:
    def test_removes_emoji(self):
        assert sanitize("The garden was quiet \U0001f331") == "The garden was quiet"

    def test_collapses_whitespace(self):
        assert sanitize("The   room\n\n\nwas  still.") == "The room was still."

    def test_strips_leading_and_trailing_space(self):
        assert sanitize("   The room was still.   ") == "The room was still."


class TestEmptyResults:
    """An all-action paragraph must sanitize to empty, not to garbage.

    Empty is the pipeline's signal to regenerate, so it has to be reachable.
    """

    def test_action_only_paragraph_becomes_empty(self):
        assert sanitize("*she sighs*") == ""

    def test_heading_only_becomes_empty(self):
        assert sanitize("## Chapter One") == ""

    def test_empty_input_stays_empty(self):
        assert sanitize("") == ""

    def test_whitespace_only_becomes_empty(self):
        assert sanitize("   \n\n  ") == ""


class TestPreservesProse:
    """The sanitizer must not damage the ordinary case."""

    def test_leaves_plain_prose_untouched(self):
        text = "The afternoon light came through the window and lay across the floor."
        assert sanitize(text) == text

    def test_keeps_sentence_punctuation(self):
        text = "It was quiet, mostly. The kettle ticked; nothing else happened."
        assert sanitize(text) == text

    def test_keeps_parentheses(self):
        text = "The house (the old one) was empty."
        assert sanitize(text) == text
