"""Smoke tests for the streaming corpus loader + vocab builder (zero download)."""
from src.data.build_vocab import LOCAL_SAMPLES, build_vocab
from src.data.corpus import iter_text_from_samples


def test_iter_text_from_samples():
    chars = list(iter_text_from_samples(["你好", "世界"]))
    assert chars == ["你", "好", "世", "界"]


def test_build_vocab_has_specials():
    tok = build_vocab(iter_text_from_samples(LOCAL_SAMPLES), vocab_size=100)
    assert tok.pad_id == 0
    assert tok.mask_id == 1
    assert tok.unk_id == 2
    assert tok.vocab_size <= 100
    assert tok.decode(tok.encode("中")) == "中"


def test_vocab_save_load(tmp_path):
    tok = build_vocab(iter_text_from_samples(LOCAL_SAMPLES), vocab_size=100)
    p = tmp_path / "vocab.json"
    tok.save(str(p))
    tok2 = type(tok).load(str(p))
    assert tok2._c2i == tok._c2i
    assert tok2.decode(tok.encode("阅读")) == "阅读"
