"""Smoke tests for batchify."""
from src.data.dataloader import batchify


def test_batchify_exact():
    batches = list(batchify(iter(range(20)), batch_size=2, seq_len=5, pad_id=0))
    assert len(batches) == 2
    assert batches[0].tolist() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert batches[1].tolist() == [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]


def test_batchify_pads_partial():
    batches = list(batchify(iter(range(12)), batch_size=2, seq_len=5, pad_id=0))
    assert len(batches) == 2
    assert batches[0].tolist() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    # trailing partial chunk padded, then padded to batch_size
    assert batches[1].tolist() == [[10, 11, 0, 0, 0], [0, 0, 0, 0, 0]]
