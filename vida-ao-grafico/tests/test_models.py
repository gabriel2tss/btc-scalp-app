"""Testes de algoritmo (não de mercado): verificam mecânica e causalidade dos modelos."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vag.alphabet import train_alphabet
from vag.evaluate import bh_select, build_dictionary, simulate
from vag.lm import MarketGPT, surprise_entropy, train_lm

LM_CFG = {"context": 32, "n_layer": 1, "n_head": 2, "d_model": 32, "dropout": 0.0, "batch_size": 8, "lr": 3e-3,
          "max_steps": 60, "eval_every": 30, "patience_evals": 5, "amp": False}


def test_lm_learns_deterministic_grammar_and_resumes(tmp_path: Path):
    seq = np.tile(np.arange(10), 400).astype(np.int16)          # gramática trivial: 0,1,2,...,9,0,1,...
    brk = np.zeros(len(seq), dtype=bool)
    cfg = dict(LM_CFG, max_steps=150)
    model, stats = train_lm(seq, brk, seq[:600], brk[:600], 10, cfg, "cpu", 0, tmp_path / "lm.pt", log=lambda *_: None)
    assert stats["best_val_loss"] < 0.5 * stats["unigram_loss"]
    again, stats2 = train_lm(seq, brk, seq[:600], brk[:600], 10, cfg, "cpu", 0, tmp_path / "lm.pt", log=lambda *_: None)
    assert stats2["best_val_loss"] == stats["best_val_loss"]    # checkpoint "done" é reaproveitado


def test_surprise_is_causal():
    torch.manual_seed(0)
    m = MarketGPT(12, 32, 1, 2, 32, 0.0)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 12, 300).astype(np.int16)
    brk = np.zeros(len(x), dtype=bool)
    s1, e1 = surprise_entropy(m, x, brk, "cpu", False)
    x2 = x.copy()
    x2[200:] = (x2[200:] + 1) % 12                              # muda só o "futuro"
    s2, e2 = surprise_entropy(m, x2, brk, "cpu", False)
    np.testing.assert_allclose(s1[:200], s2[:200], rtol=1e-5, equal_nan=True)
    np.testing.assert_allclose(e1[:200], e2[:200], rtol=1e-5, equal_nan=True)
    assert np.isfinite(s1[1:]).all() and np.isfinite(e1).all()


def test_alphabet_uses_codebook():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 3, (20, 6))
    x = (centers[rng.integers(0, 20, 20000)] + rng.normal(0, 0.1, (20000, 6))).astype(np.float32)
    cfg = {"codebook_size": 32, "latent_dim": 4, "hidden": 32, "commitment": 0.25, "ema_decay": 0.95,
           "batch_size": 512, "epochs": 3, "lr": 3e-3, "dead_code_restart_every": 20}
    a, stats = train_alphabet(x, cfg, "cpu", 0, log=lambda *_: None)
    assert stats["used_letters"] >= 20
    codes = a.encode(x)
    assert codes.min() >= 0 and codes.max() < 32


def test_dictionary_and_simulation_accounting():
    keys = np.array([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int64)
    fwd = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, np.nan])
    d = build_dictionary(keys, fwd, np.arange(8), horizon=1)
    assert d.loc[1, "count"] == 4 and d.loc[2, "count"] == 3
    assert d.loc[1, "mean"] == 0.01
    sig = np.array([1, 1, 1, 0, 0, 0, -1, 0], dtype=np.int8)
    tr = simulate(sig, fwd, np.arange(8), horizon=2, cost_rt=0.001)
    assert list(tr["t"]) == [0, 6]                              # t=1,2 ignorados: posição aberta até t=2
    assert np.isclose(tr["net"].iloc[0], 0.009) and np.isclose(tr["net"].iloc[1], -0.011)


def test_bh():
    p = pd.Series([0.001, 0.01, 0.04, 0.5])
    assert list(bh_select(p, 0.05)) == [True, True, False, False]
