"""Text normalization shared by preprocessing, training and the deployed app.

Whatever is done to a sentence here at training time must be done identically
at inference time, so this module is the single source of truth for both.
"""
import re
import unicodedata

# Amharic has several families of homophone letters that writers use
# interchangeably (e.g. ሀ/ሐ/ኀ are all pronounced "ha"). Mapping each family to
# one canonical letter shrinks the vocabulary and removes spurious mismatches
# between the model output and the reference.
_HOMOPHONE_BASES = {
    "ሐ": "ሀ", "ኀ": "ሀ",   # ha
    "ሠ": "ሰ",             # se
    "ዐ": "አ",             # a
    "ፀ": "ጸ",             # tse
}


def _build_homophone_table():
    table = {}
    for src, dst in _HOMOPHONE_BASES.items():
        s, d = ord(src), ord(dst)
        # each Ethiopic consonant has 7 contiguous vowel orders (ä u i a e ə o)
        for order in range(7):
            table[s + order] = chr(d + order)
    # labialised forms that collapse to the plain consonant family
    table.update({ord("ሗ"): "ሏ", ord("ሧ"): "ሷ", ord("ፇ"): "ጿ"})
    table.update({ord("ቈ"): "ቆ", ord("ኰ"): "ኮ", ord("ጐ"): "ጎ"})
    return table


_HOMOPHONES = str.maketrans(_build_homophone_table())

_QUOTES = str.maketrans({
    "«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "‹": "'", "›": "'",
    "‘": "'", "’": "'", "`": "'", "ʹ": "", "·": "",   # ʹ and · are pronunciation marks in NWT names
    " ": " ", "﻿": "", "​": "", "‌": "", "‍": "",
})

# Ethiopic punctuation → a spaced canonical form. "፡" (word space) is dropped,
# "፤"/"፣" are clause separators, "።" is the full stop.
_ETH_PUNCT = str.maketrans({"፡": " ", "፥": "፣", "፦": ":", "፧": "?", "፨": "።"})

_SPACES = re.compile(r"\s+")
_PRIVATE_USE = re.compile(r"[-]")
_BRACKET_REF = re.compile(r"\(\s*\d+[\s:,\d]*\)")    # verse references like "(13 36)"


def _common(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_QUOTES)
    text = _PRIVATE_USE.sub("", text)
    text = _BRACKET_REF.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def normalize_en(text: str) -> str:
    """English: unify quotes/whitespace, lowercase (the corpus mixes cases
    arbitrarily — half the Bible sentences are lower-cased)."""
    return _common(text).lower()


def normalize_am(text: str) -> str:
    """Amharic: unify quotes/whitespace, Ethiopic punctuation and homophones."""
    text = _common(text).translate(_ETH_PUNCT).translate(_HOMOPHONES)
    text = text.replace("::", "።")                     # ASCII stand-in for "።"
    text = re.sub(r"\s*([።፣፤?!])", r"\1", text)       # no space before punctuation
    return _SPACES.sub(" ", text).strip()
