"""Turn writer output into something a TTS can read aloud.

The writer is a roleplay tune. Observed in practice, with the system prompt
explicitly forbidding all of it: asterisk actions, em-dashes, and a literal
"*150 words*" trailing the paragraph. The TTS pronounces all of that. The prompt
is a suggestion; this module is the guarantee.
"""

import re

# *she sighs softly* -> gone entirely. Roleplay actions are not narration.
# Non-greedy, single-line, so it can't eat a whole paragraph between two
# unrelated asterisks on different lines.
_ACTION = re.compile(r"\*[^*\n]*\*")
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")
_ITALIC_UNDERSCORE = re.compile(r"(?<!\w)_([^_\n]+)_(?!\w)")
_STRAY_ASTERISK = re.compile(r"\*")

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s.*$", re.MULTILINE)
_BARE_CHAPTER = re.compile(
    r"^\s*(chapter|part|scene|section)\b[^\n.!?]{0,40}$",
    re.MULTILINE | re.IGNORECASE,
)
_LIST_MARKER = re.compile(r"^\s{0,3}(?:[-*+]|\d{1,2}[.)])\s+", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)

# Anything outside basic latin text/punctuation: emoji, dingbats, arrows.
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"
    "\U00002190-\U000021ff"
    "\U00002300-\U000027bf"
    "\U0000fe00-\U0000fe0f"
    "\U00002b00-\U00002bff"
    "]+"
)

_WHITESPACE = re.compile(r"\s+")

_TYPOGRAPHY = {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'",
    "…": "...",
    " ": " ",
}

# "The door—it was open." -> "The door, it was open."  The TTS either says
# nothing for these or stumbles; a comma gives it the pause that was meant.
_DASH_PAUSE = re.compile(r"\s*[—–]\s*|\s+-\s+")

# A sentence ends at .!? plus optional closing quote, followed by space or end.
# The negative lookbehind/ahead keeps "waited... and" as one sentence: an
# ellipsis is a pause the writer uses constantly, not a sentence boundary.
_SENTENCE_END = re.compile(r'(?<!\.)[.!?]["\')\]]?(?=\s|$)(?!\.)')


def sanitize(text: str) -> str:
    """Clean writer output for speech.

    Returns "" when nothing speakable survives, which is the pipeline's signal
    to regenerate rather than to render silence.
    """
    if not text:
        return ""

    text = _HEADING.sub("", text)
    text = _BARE_CHAPTER.sub("", text)

    # Bold before actions: **very** is emphasis to keep, *sighs* is an action to
    # drop. Reversing these two lines silently eats every bold word.
    text = _BOLD.sub(r"\1", text)
    text = _ACTION.sub("", text)
    text = _STRAY_ASTERISK.sub("", text)
    text = _ITALIC_UNDERSCORE.sub(r"\1", text)

    text = _BLOCKQUOTE.sub("", text)
    text = _LIST_MARKER.sub("", text)

    for bad, good in _TYPOGRAPHY.items():
        text = text.replace(bad, good)
    text = _DASH_PAUSE.sub(", ", text)

    text = _EMOJI.sub("", text)
    text = _WHITESPACE.sub(" ", text)

    return text.strip()


def trim_to_last_sentence(text: str) -> str:
    """Drop a trailing incomplete sentence.

    The writer hits max_tokens mid-clause and the TTS reads the fragment aloud
    and stops dead. Both sample generations during design ended this way, so
    this is the common case, not an edge case.

    Returns "" when no complete sentence exists, so the caller regenerates
    rather than speaking a fragment.
    """
    if not text:
        return ""
    matches = list(_SENTENCE_END.finditer(text))
    if not matches:
        return ""
    return text[: matches[-1].end()].strip()


