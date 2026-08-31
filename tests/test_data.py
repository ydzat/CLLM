"""Smoke tests for the data layer (tasks T2, T3)."""
import random

from src.data.packing import grid_coords, mask_whole_blocks, num_blocks
from src.data.tokenizer import CharTokenizer


def test_tokenizer_roundtrip():
    tok = CharTokenizer(chars=list("中文阅读测试"))
    assert tok.decode(tok.encode("中文阅读")) == "中文阅读"


def test_tokenizer_unknown_maps_to_unk_not_pad():
    tok = CharTokenizer(chars=["中"])
    ids = tok.encode("中x")
    assert ids == [3, tok.unk_id]
    assert tok.unk_id != tok.pad_id


def test_num_blocks_1024_t4_is_64():
    assert num_blocks(1024, 4) == 64


def test_grid_coords_width():
    rows, cols = grid_coords(10, 4)
    assert rows == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    assert cols == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]


def test_mask_whole_blocks():
    rng = random.Random(0)
    m = mask_whole_blocks(1024, 4, 0.15, rng=rng)
    assert len(m) == round(64 * 0.15)
    assert min(m) >= 0 and max(m) < 64
