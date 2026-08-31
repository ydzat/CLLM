"""Character-level tokenizer: one Chinese character = one token.

Special ids are fixed: ``[PAD]`` = 0, ``[MASK]`` = 1, ``[UNK]`` = 2.
Character ids start at 3. Unknown characters encode to ``[UNK]``, never
``[PAD]`` (``[PAD]`` is reserved for grid padding).
"""
from __future__ import annotations

SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[MASK]", "[UNK]")


class CharTokenizer:
    """Bidirectional character <-> id map with fixed special tokens."""

    def __init__(self, chars: list[str] | None = None) -> None:
        self._c2i: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self._i2c: dict[int, str] = {i: tok for tok, i in self._c2i.items()}
        if chars:
            self.build(chars)

    def build(self, chars: list[str]) -> None:
        """Register characters; new chars get sequential ids (idempotent)."""
        for c in chars:
            if c not in self._c2i:
                idx = len(self._c2i)
                self._c2i[c] = idx
                self._i2c[idx] = c

    @property
    def vocab_size(self) -> int:
        return len(self._c2i)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def mask_id(self) -> int:
        return 1

    @property
    def unk_id(self) -> int:
        return 2

    def encode(self, text: str) -> list[int]:
        return [self._c2i.get(c, self.unk_id) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._i2c.get(i, "[UNK]") for i in ids)

    def save(self, path: str) -> None:
        """Persist the char->id map as JSON (creates parent dirs)."""
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._c2i, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Rebuild a tokenizer from a JSON written by :meth:`save`."""
        import json

        with open(path, encoding="utf-8") as f:
            c2i = json.load(f)
        tok = cls()
        tok._c2i = c2i
        tok._i2c = {i: c for c, i in c2i.items()}
        return tok
