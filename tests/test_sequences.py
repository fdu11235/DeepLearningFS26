import numpy as np

from fraud.models.lstm.sequences import build_sequences


def _toy(n_per_card: list[int], seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    cards = []
    times = []
    labels = []
    base_t = 0
    for cid, n in enumerate(n_per_card):
        for k in range(n):
            rows.append(rng.standard_normal(4))
            cards.append(cid)
            times.append(base_t + k)
            labels.append(int(rng.random() < 0.1))
        base_t += 1000
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(cards),
        np.asarray(times, dtype=np.int64),
    )


def test_label_equals_last_tx_label():
    X, y, c, t = _toy([5, 3])
    seq = build_sequences(X, y, c, t, seq_len=4)
    # The library returns sequences in original row order.
    assert np.array_equal(seq.y.astype(int), y)


def test_lengths_capped_and_short_history_left_padded():
    X, y, c, t = _toy([3])  # only 3 transactions for one card
    seq = build_sequences(X, y, c, t, seq_len=5)
    # First tx has length 1, second 2, third 3 (sorted in time).
    sorted_lengths = seq.lengths[np.argsort(t)]
    assert sorted_lengths.tolist() == [1, 2, 3]
    # First-tx window must have its valid value at the LAST position (left-padded).
    sorted_X = seq.X[np.argsort(t)]
    assert np.allclose(sorted_X[0, :-1], 0.0)
    assert not np.allclose(sorted_X[0, -1], 0.0)


def test_no_cross_card_leakage():
    """A window for card B must not see card A's transactions even if A's
    timestamps are earlier."""
    X = np.array(
        [
            [1.0, 0.0],  # card 0 (label later)
            [2.0, 0.0],  # card 0
            [99.0, 0.0],  # card 1, earliest of card 1 — its window must NOT contain the rows above
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1], dtype=np.int64)
    c = np.array([0, 0, 1])
    t = np.array([1, 2, 3], dtype=np.int64)
    seq = build_sequences(X, y, c, t, seq_len=5)
    # Row index 2 corresponds to card 1's only transaction.
    card1_window = seq.X[2]
    # Length should be 1 (card 1 has only 1 tx) and only the last position should be non-zero.
    assert seq.lengths[2] == 1
    assert np.allclose(card1_window[:-1], 0.0)
    assert np.allclose(card1_window[-1], np.array([99.0, 0.0]))
