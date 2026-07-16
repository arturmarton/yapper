"""The writer hits max_tokens mid-clause and the TTS reads the fragment aloud.

Both real generations during design ended this way: "In the distance, a car
passes on" and "while inside your". These cases are transcribed from actual
model output, not invented.
"""

from yapper.sanitize import trim_to_last_sentence


class TestTrimming:
    def test_drops_trailing_fragment(self):
        text = "The clock ticked. In the distance, a car passes on"
        assert trim_to_last_sentence(text) == "The clock ticked."

    def test_keeps_complete_text_untouched(self):
        text = "The clock ticked. The light moved across the floor."
        assert trim_to_last_sentence(text) == text

    def test_keeps_question_and_exclamation_endings(self):
        text = "It was quiet! Then quieter."
        assert trim_to_last_sentence(text) == text

    def test_keeps_sentence_ending_in_quote(self):
        text = 'She said "later." Then nothing happened at all'
        assert trim_to_last_sentence(text) == 'She said "later."'

    def test_does_not_split_on_ellipsis_mid_sentence(self):
        """'She waited... and the light moved' is one sentence, not a fragment."""
        text = "She waited... and the light moved on."
        assert trim_to_last_sentence(text) == text

    def test_strips_trailing_space_after_trim(self):
        assert trim_to_last_sentence("Done. and then") == "Done."


class TestNothingUsable:
    """No complete sentence means regenerate, not speak a fragment."""

    def test_fragment_only_returns_empty(self):
        assert trim_to_last_sentence("the car passes on") == ""

    def test_empty_returns_empty(self):
        assert trim_to_last_sentence("") == ""
