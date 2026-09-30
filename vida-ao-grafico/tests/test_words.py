import numpy as np

from vag.words import Vocabulary, learn_bpe


def test_bpe_learns_repeated_pattern_and_is_causal():
    rng = np.random.default_rng(0)
    k = 8
    # letras aleatórias com o motivo 3-5-7 inserido muitas vezes
    parts = []
    for _ in range(3000):
        parts.append(rng.integers(0, k, rng.integers(1, 4)))
        parts.append(np.array([3, 5, 7]))
    letters = np.concatenate(parts).astype(np.int16)
    vocab = learn_bpe(letters, k, merges=20, max_word_len=6, min_pair_count=50, log=lambda *_: None)
    assert (3, 5, 7) in vocab.words

    wid, wlen = vocab.word_ending_at(letters)
    # causalidade: calcular só com o prefixo dá o mesmo resultado em cada t
    for t in [10, 500, 4000, len(letters) - 1]:
        w2, l2 = vocab.word_ending_at(letters[: t + 1])
        assert w2[-1] == wid[t] and l2[-1] == wlen[t]
    # a palavra encontrada realmente termina em t
    for t in range(0, len(letters), 97):
        assert vocab.words[wid[t]] == tuple(letters[t - wlen[t] + 1: t + 1])


def test_words_do_not_cross_breaks():
    vocab = Vocabulary(4, [(0,), (1,), (2,), (3,), (1, 2)])
    letters = np.array([1, 2, 1, 2], dtype=np.int16)
    breaks = np.array([False, True, False, False])
    wid, wlen = vocab.word_ending_at(letters, breaks)
    assert list(wlen) == [1, 1, 1, 2]


def test_phrases_walk_backwards():
    vocab = Vocabulary(4, [(0,), (1,), (2,), (3,), (1, 2)])
    letters = np.array([3, 1, 2, 0], dtype=np.int16)
    ph = vocab.phrases(letters, 3)
    assert list(ph[3]) == [0, 4, 3]     # "0" <- "1 2" <- "3"
    assert list(ph[0]) == [3, -1, -1]
