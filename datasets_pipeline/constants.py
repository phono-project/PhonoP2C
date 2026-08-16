"""Shared constants for the datasets / preprocessing pipeline."""

# Segment labels, shared between preprocessor.py and dataset.py.
PAUSE_LABEL = 0
CHINESE_LABEL = 1
NON_CHINESE_LABEL = 2

# Pause-meaning punctuation (half-width, after NFKC).
PAUSE_PUNCT = ["!", "?", "。", "\n", ":", ";", ","]
_PAUSE_SET = set(PAUSE_PUNCT)
