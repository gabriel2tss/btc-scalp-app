"""Fase 4a — palavras: BPE sobre a sequência de letras.

Aprendizado (só no treino): pares de letras/palavras que aparecem juntos muito
mais que o resto viram uma palavra nova, repetidamente.

Uso SEM olhar o futuro: a segmentação clássica de BPE decide onde uma palavra
termina olhando letras que vêm depois. Aqui, em cada minuto t, a "palavra que
acabou de ser dita" é a maior palavra do vocabulário que termina exatamente em t
(sufixo), e a frase é formada andando para trás. Só usa letras <= t, então
serve igual no backtest e ao vivo.
"""

from __future__ import annotations

import json

import numpy as np

_P = np.uint64(0x9E3779B97F4A7C15)  # multiplicador para hash polinomial (overflow proposital mod 2^64)


def _hash_words(words: list[tuple[int, ...]]) -> dict[int, dict[int, int]]:
    """{comprimento: {hash: word_id}}"""
    out: dict[int, dict[int, int]] = {}
    for wid, w in enumerate(words):
        h = np.uint64(0)
        with np.errstate(over="ignore"):
            for j, letter in enumerate(reversed(w)):   # j=0 é a última letra
                h = h + np.uint64(letter + 1) * (_P ** np.uint64(j + 1))
        out.setdefault(len(w), {})[int(h)] = wid
    return out


class Vocabulary:
    def __init__(self, k: int, words: list[tuple[int, ...]]):
        self.k = k
        self.words = words             # words[i] = tupla de letras; as K primeiras são as próprias letras
        self.max_len = max(len(w) for w in words)
        self._by_len = _hash_words(words)

    def __len__(self):
        return len(self.words)

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"k": self.k, "words": [list(w) for w in self.words]}, f)

    @classmethod
    def load(cls, path) -> "Vocabulary":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["k"], [tuple(w) for w in d["words"]])

    def word_ending_at(self, letters: np.ndarray, breaks: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Para cada t: (id da maior palavra que termina em t, comprimento dela em letras).

        breaks[i] = True se há um buraco nos dados logo antes da letra i
        (palavras não atravessam buracos).
        """
        x = letters.astype(np.uint64) + np.uint64(1)
        n = len(x)
        brk_cum = np.cumsum(breaks.astype(np.int64)) if breaks is not None else np.zeros(n, dtype=np.int64)
        best_id = letters.astype(np.int32).copy()         # comprimento 1 sempre existe (a própria letra)
        best_len = np.ones(n, dtype=np.int16)
        h = np.zeros(n, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for ln in range(1, self.max_len + 1):
                # h[t] = soma_{j<ln} x[t-j] * P^(j+1)
                shifted = np.zeros(n, dtype=np.uint64)
                shifted[ln - 1:] = x[: n - ln + 1]
                h = h + shifted * (_P ** np.uint64(ln))
                table = self._by_len.get(ln)
                if ln == 1 or not table:
                    continue
                keys = np.fromiter(table.keys(), dtype=np.uint64, count=len(table))
                ids = np.fromiter(table.values(), dtype=np.int32, count=len(table))
                order = np.argsort(keys)
                keys, ids = keys[order], ids[order]
                pos = np.searchsorted(keys, h)
                pos[pos >= len(keys)] = 0
                hit = keys[pos] == h
                hit[: ln - 1] = False
                if breaks is not None:
                    start = np.arange(n) - ln + 1
                    ok = np.zeros(n, dtype=bool)
                    ok[ln - 1:] = (brk_cum[ln - 1:] - brk_cum[start[ln - 1:]]) == 0
                    hit &= ok
                best_id[hit] = ids[pos[hit]]
                best_len[hit] = ln
        return best_id, best_len

    def phrases(self, letters: np.ndarray, n_words: int, breaks: np.ndarray | None = None) -> np.ndarray:
        """Matriz (len, n_words): palavra que termina em t, a anterior, a anterior à anterior...
        -1 quando não há palavras suficientes para trás."""
        wid, wlen = self.word_ending_at(letters, breaks)
        n = len(letters)
        out = np.full((n, n_words), -1, dtype=np.int32)
        cur = np.arange(n, dtype=np.int64)
        valid = np.ones(n, dtype=bool)
        for j in range(n_words):
            out[valid, j] = wid[cur[valid]]
            cur = cur - wlen[np.clip(cur, 0, n - 1)]
            valid &= cur >= 0
        return out


def learn_bpe(letters: np.ndarray, k: int, merges: int, max_word_len: int, min_pair_count: int,
              breaks: np.ndarray | None = None, log=print) -> Vocabulary:
    """BPE vetorizado. `letters` = sequência de treino (ids 0..k-1)."""
    seq = letters.astype(np.int64).copy()
    brk = breaks.copy() if breaks is not None else np.zeros(len(seq), dtype=bool)
    words: list[tuple[int, ...]] = [(i,) for i in range(k)]
    lens = np.ones(k + merges, dtype=np.int64)
    V = k + merges
    for m in range(merges):
        a, b = seq[:-1], seq[1:]
        ok = ~brk[1:] & ((lens[a] + lens[b]) <= max_word_len)
        codes = (a * V + b)[ok]
        counts = np.bincount(codes, minlength=V * V) if len(codes) else np.zeros(1)
        best = int(counts.argmax())
        if counts[best] < min_pair_count:
            log(f"    BPE parou em {m} fusões (par mais comum < {min_pair_count})")
            break
        pa, pb = divmod(best, V)
        match = np.zeros(len(seq) - 1, dtype=bool)
        match[ok] = codes == best
        if pa == pb:   # evita sobreposição em "aaa": mantém ocorrências alternadas de cada sequência
            idx = np.arange(len(match))
            last_false = np.maximum.accumulate(np.where(~match, idx, -1))
            match &= ((idx - last_false - 1) % 2) == 0
        new_id = k + m
        words.append(words[pa] + words[pb])
        lens[new_id] = lens[pa] + lens[pb]
        pos = np.flatnonzero(match)
        seq[pos] = new_id
        keep = np.ones(len(seq), dtype=bool)
        keep[pos + 1] = False
        seq, brk = seq[keep], brk[keep]
        if (m + 1) % 200 == 0:
            log(f"    BPE {m + 1}/{merges}: {len(seq):,} tokens (compressão {len(letters) / len(seq):.2f}x)")
    return Vocabulary(k, words)
