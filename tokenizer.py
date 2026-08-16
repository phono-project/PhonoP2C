"""
PostfixLM Tokenizer

Three-vocabulary tokenizer with Chinese/pinyin/context separation.
"""

from __future__ import annotations

import json
import logging
import os
import random
from types import SimpleNamespace
from typing import Optional, Union

import torch
import yaml

# ── Initials (简拼) ───────────────────────────────────────────────────────────
_SIMPLE_INITIALS = {
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x",
    "zh", "ch", "sh", "r", "z", "c", "s",
    "y", "w",
}

_COMP_CONSONANTS = {"zh", "ch", "sh"}
_SINGLE_INITIALS = {i for i in _SIMPLE_INITIALS if len(i) == 1}


def get_initial_and_final(syllable: str) -> tuple[str, str]:
    """Split a pinyin syllable into (initial, final) parts."""
    if len(syllable) >= 2 and syllable[:2] in _COMP_CONSONANTS:
        return syllable[:2], syllable[2:]
    if syllable and syllable[0] in _SINGLE_INITIALS:
        return syllable[0], syllable[1:]
    return "", syllable


def simple_initial_candidates(syllable: str) -> list[str]:
    """Return the valid 简拼 (simple-initial) forms of a pinyin syllable.

    Syllables *without* an initial (e.g. "ai", "ao", "an") previously yielded
    nothing from prefix splitting; now their first letter is used as the
    shorthand form.
    """
    initial, _ = get_initial_and_final(syllable)
    if initial:
        candidates = [initial]
        if len(initial) == 2:
            candidates.append(initial[0])
        return candidates
    if syllable:
        return [syllable[0]]
    return []


def _read_vocab_tokens(path: str) -> list[str]:
    """Read a unigram vocab file, one token per line.

    A line consisting of exactly the two literal characters ``\\n`` (backslash
    followed by 'n') is converted to a real newline character.  Empty lines are
    skipped (a genuine newline token must be written as the literal ``\\n``).
    """
    tokens: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.rstrip("\n").rstrip("\r")
            if len(tok) == 2 and tok == "\\n":
                tok = "\n"
            if not tok:
                continue
            tokens.append(tok)
    return tokens


class P2CTokenizer:
    """Three-vocabulary tokenizer for PostfixLM.

    Args:
        chinese_vocab_path : str
            Path to the Chinese character vocabulary file (one char per line).
            Used as the post-model output/target space.
        context_vocab_path : str
            Path to the context vocabulary file (one token per line).  Used as the
            pre-model input space; special tokens are appended at its end.
        pinyin_vocab_path : str
            Path to the pinyin vocabulary file (one syllable per line).
        special_tokens_def : list or None
            Ordered list of special-token names, e.g. ``["bos_token"]``.  They are
            appended to the end of context_vocab.  If None, no special tokens.
    """

    def __init__(
        self,
        chinese_vocab_path: str,
        context_vocab_path: str,
        pinyin_vocab_path: str,
        special_tokens_def: Optional[list] = None,
    ):
        # uild chinese_vocab
        chinese_chars = _read_vocab_tokens(chinese_vocab_path)

        # Deduplicate while preserving order (first occurrence wins lowest ID)
        seen = set()
        unique_chars = []
        for ch in chinese_chars:
            if ch not in seen:
                seen.add(ch)
                unique_chars.append(ch)

        self._chinese_vocab: dict[str, int] = {}
        for i, ch in enumerate(unique_chars):
            self._chinese_vocab[ch] = i
        self._chinese_size = len(self._chinese_vocab)
        self._id_to_chinese = {i: ch for ch, i in self._chinese_vocab.items()}

        # Build pinyin_vocab
        pinyin_tokens = _read_vocab_tokens(pinyin_vocab_path)

        self._pinyin_vocab: dict[str, int] = {}
        self._pinyin_list: list[str] = []
        seen_py = set()
        py_idx = 0
        for py in pinyin_tokens:
            if py not in seen_py:
                seen_py.add(py)
                self._pinyin_vocab[py] = py_idx
                self._pinyin_list.append(py)
                py_idx += 1
        self._pinyin_size = len(self._pinyin_vocab)
        
        self._id_to_pinyin = {i: py for py, i in self._pinyin_vocab.items()}

        # Build context_vocab (own file + special tokens at end)
        context_tokens = _read_vocab_tokens(context_vocab_path)
        self._context_vocab: dict[str, int] = {}
        for ctok in context_tokens:
            if ctok not in self._context_vocab:
                self._context_vocab[ctok] = len(self._context_vocab)

        # Number of "real" context tokens, before special tokens are appended.
        self._context_base_size = len(self._context_vocab)
        self._context_size = self._context_base_size

        self._special_tokens_list = special_tokens_def or []
        self._num_special = len(self._special_tokens_list)
        self.spec_tokens = SimpleNamespace()

        if self._special_tokens_list:
            # Special tokens are appended at the end of context_vocab.
            # Order in the list determines local_id (0, 1, 2, ...)
            for local_idx, name in enumerate(self._special_tokens_list):
                global_id = self._context_base_size + local_idx
                self._context_vocab[name] = global_id
                # Inject as attribute on spec_tokens
                setattr(self.spec_tokens, name, global_id)

        self._context_size += self._num_special
        self._id_to_context = {i: ch for ch, i in self._context_vocab.items()}

        # Per-character pronunciation frequencies (part of the tokenizer).
        # {char: {pinyin: probability}} normalized to sum to 1.
        self._char_pinyin_freq: dict[str, dict[str, float]] = {}
        self._char_pinyin_freq_path: Optional[str] = None

    # Class methods
    @classmethod
    def from_config(cls, config_path: str) -> "P2CTokenizer":
        """Build tokenizer from a YAML config file.

        The config is resolved relative to the config file's directory.
        """
        config_dir = os.path.dirname(os.path.abspath(config_path))
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        return cls.from_config_dict(config_dir, cfg)

    @classmethod
    def from_config_dict(cls, config_dir: str, config_dict: dict) -> "P2CTokenizer":
        """Build tokenizer from a resolved config dict.

        config_dir is used to resolve relative vocab paths.
        config_dict should have the structure: {"vocabs": {...}}
        """
        vocabs = config_dict["vocabs"]
        chinese_path = os.path.join(config_dir, vocabs["chinese_vocab"])
        context_path = os.path.join(config_dir, vocabs["context_vocab"])
        pinyin_path = os.path.join(config_dir, vocabs["pinyin_vocab"])
        special_tokens_def = vocabs.get("context_special_tokens", None)

        tokenizer = cls(chinese_path, context_path, pinyin_path, special_tokens_def)

        # Optional per-character pronunciation frequencies.
        freq_rel = vocabs.get("characters_pronounce_frequency", None)
        if freq_rel:
            freq_path = os.path.join(config_dir, freq_rel)
            if os.path.exists(freq_path):
                tokenizer.load_character_pinyin_frequency(freq_path)
            else:
                logging.warning(
                    "characters_pronounce_frequency file not found: %s", freq_path
                )
        return tokenizer

    # Vocabulary sizes
    @property
    def chinese_vocab_size(self) -> int:
        return self._chinese_size

    @property
    def pinyin_vocab_size(self) -> int:
        return self._pinyin_size

    @property
    def context_vocab_size(self) -> int:
        return self._context_size

    @property
    def vocab_size(self) -> int:
        """Alias for context_vocab_size (used by pre model)."""
        return self._context_size

    @property
    def proj_size(self) -> int:
        """Alias for chinese_vocab_size (used by post model projection)."""
        return self._chinese_size

    @property
    def num_special_tokens(self) -> int:
        return self._num_special

    # Heteronym sampling
    def load_character_pinyin_frequency(self, path: str) -> None:
        """Load the per-character pronunciation frequency JSON.

        Expected format: ``{char: {pinyin: probability, ...}, ...}`` with
        probabilities normalized to sum to 1 per character.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._char_pinyin_freq = {
            str(ch): {str(py): float(p) for py, p in readings.items()}
            for ch, readings in raw.items()
        }
        self._char_pinyin_freq_path = path

    @property
    def character_pinyin_frequency(self) -> dict[str, dict[str, float]]:
        return self._char_pinyin_freq

    def sample_heteronym(self, ch: str, rng: Optional[random.Random] = None) -> Optional[str]:
        """Sample a pronunciation for Chinese char *ch* by corpus frequency.

        Uses the probabilities in characters_pronounce_frequency (a part of
        the tokenizer).  Characters without recorded frequencies fall back to
        pypinyin's most probable reading.  Returns None when no reading is
        obtainable.
        """
        freqs = self._char_pinyin_freq.get(ch)
        if freqs:
            rng = rng or random
            choices = list(freqs.keys())
            weights = [freqs[c] for c in choices]
            return rng.choices(choices, weights=weights, k=1)[0]

        from pypinyin import pinyin, Style

        matrix = pinyin(ch, heteronym=False, strict=False,
                        errors="ignore", style=Style.NORMAL, v_to_u=False)
        if not matrix or not matrix[0]:
            return None
        sounds = [s for s in matrix[0] if s != "ê"]
        return sounds[0] if sounds else None

    # Encoding methods
    def encode_context(self, text: str) -> list[int]:
        """Encode text using context_vocab.  Returns IDs in context_vocab space.

        Used for the *pre* model input (Chinese context + punctuation/English).
        """
        ids = []
        for ch in text:
            tid = self._context_vocab.get(ch)
            if tid is not None:
                ids.append(tid)
        return ids

    def encode_chinese(self, text: str) -> list[int]:
        """Encode text using chinese_vocab only. Used for post-model targets."""
        ids = []
        for ch in text:
            tid = self._chinese_vocab.get(ch)
            if tid is not None:
                ids.append(tid)
        return ids

    def encode_chinese_with_ignore(self, text: str, ignore_index: int = -100) -> list[int]:
        """Encode text using chinese_vocab, mapping non-Chinese chars to ``ignore_index``.

        Unlike :meth:`encode_chinese`, the length is preserved.
        """
        return [self._chinese_vocab.get(ch, ignore_index) for ch in text]

    def encode_pinyin(self, pinyin_list: list[str]) -> list[int]:
        """Encode a list of pinyin syllables using pinyin_vocab.

        Unknown syllables are mapped to the closest match via edit distance.
        """
        ids = []
        for py in pinyin_list:
            tid = self._pinyin_vocab.get(py)
            if tid is not None:
                ids.append(tid)
            else:
                # Fallback: nearest edit-distance match
                best_idx = min(
                    range(len(self._pinyin_list)),
                    key=lambda i: self._edit_distance(py, self._pinyin_list[i]),
                )
                ids.append(best_idx)
        return ids

    def check_pinyin(self, pinyin_list: list[str]) -> bool:
        """Check whether all pinyin syllables are in-vocabulary."""
        for py in pinyin_list:
            if py not in self._pinyin_vocab:
                return False
        return True

    # Filtering
    def filter_text(
        self,
        text: str,
        allow_characters: Optional[list[str]] = None,
        vocab: str = "chinese",
    ) -> str:
        """Keep only characters present in the selected vocabulary.

        Args:
            allow_characters : list[str] or None
                Extra characters to keep even though they are not in the selected
                vocab (e.g. whitespace / boundary punctuation a caller wants to
                preserve temporarily for segmentation purposes).
            vocab : {"chinese", "context"}
                Which vocabulary to filter against.  "chinese" keeps only pure
                Chinese chars; "context" keeps the wider pre-model vocabulary
                (Chinese + punctuation + English).
        """
        if vocab == "context":
            keep_vocab = self._context_vocab
        elif vocab == "chinese":
            keep_vocab = self._chinese_vocab
        else:
            raise ValueError(f"filter_text: unknown vocab '{vocab}' (expected 'context' or 'chinese')")

        if allow_characters:
            allow_set = set(allow_characters)
            return "".join(
                ch for ch in text if ch in keep_vocab or ch in allow_set
            )
        return "".join(ch for ch in text if ch in keep_vocab)

    def is_chinese(self, ch: str) -> bool:
        """Return True if *ch* is a single character present in chinese_vocab."""
        return ch in self._chinese_vocab

    def check(self, text: str, vocab: str = "chinese") -> bool:
        """Check whether every character in *text* is in the selected vocab."""
        keep_vocab = self._context_vocab if vocab == "context" else self._chinese_vocab
        for ch in text:
            if ch not in keep_vocab:
                return False
        return True

    # Decoding
    def ids_to_text(self, ids: list[int]) -> str:
        """Convert chinese_vocab IDs back to characters."""
        chars = []
        for tid in ids:
            ch = self._id_to_chinese.get(tid)
            if ch is not None:
                chars.append(ch)
        return "".join(chars)

    def ids_to_context_text(self, ids: list[int]) -> str:
        """Convert context_vocab IDs back to characters (skipping special tokens)."""
        chars = []
        for tid in ids:
            if tid >= self._context_base_size:
                continue  # skip special tokens
            ch = self._id_to_context.get(tid)
            if ch is not None:
                chars.append(ch)
        return "".join(chars)

    def decode_greedy(self, ids: list[int]) -> str:
        """Decode chinese_vocab IDs to characters (greedy)."""
        return self.ids_to_text(ids)

    # Edit distance
    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        m, n = len(a), len(b)
        if m == 0:
            return n
        if n == 0:
            return m
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    # Possibility map
    def create_possibility_map(self) -> torch.Tensor:
        """Build a (pinyin_vocab_size, chinese_vocab_size) bool mask.

        Entry (p, c) is True if pinyin *p* can possibly map to Chinese char *c*,
        otherwise False. Used by the post model to mask impossible logits.

            If pinyin exactly matches one of the char's possible pinyin readings -> True
            If pinyin is a 简拼 (simple initial) and the char's pinyin *starts with*
        that 简拼 -> True
        """
        from pypinyin import pinyin, Style

        V_pinyin = self._pinyin_size
        V_chinese = self._chinese_size
        
        # Initialize the mask with all False values
        mask = torch.zeros(V_pinyin, V_chinese, dtype=torch.bool)

        # Iterate through the Chinese vocabulary to map characters to pinyins
        for ch, cid in self._chinese_vocab.items():
            all_sounds_matrix = pinyin(ch, heteronym=True, strict=False,
                                       errors="ignore", style=Style.NORMAL, v_to_u=False)
            if not all_sounds_matrix or not all_sounds_matrix[0]:
                logging.warning(f"Warning: No pinyin found for character '{ch}' (ID {cid})")
                continue
                
            possible_pinyins = [s for s in all_sounds_matrix[0] if s != 'ê'] # This is a very unusual Pinyin that cannot usually be typed.
            if not possible_pinyins:
                continue
            
            for pos_py in possible_pinyins:
                # Exact match for full pinyin
                if pos_py in self._pinyin_vocab:
                    pid = self._pinyin_vocab[pos_py]
                    mask[pid, cid] = True

                # Match simple initials (简拼): the syllable's initial letter(s);
                # syllables without an initial use their first letter instead.
                for candidate in simple_initial_candidates(pos_py):
                    if candidate in self._pinyin_vocab:
                        pid = self._pinyin_vocab[candidate]
                        mask[pid, cid] = True

        return mask