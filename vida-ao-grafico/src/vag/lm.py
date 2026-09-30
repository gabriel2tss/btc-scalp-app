"""Fase 4b — modelo de linguagem do mercado: mini-GPT sobre as letras (1 token por minuto).

Treinar sobre letras (e não sobre as palavras BPE) mantém tudo causal e
alinhado minuto a minuto: a segmentação em palavras dependeria de letras
futuras. O contexto de 512 minutos (~8,5 h) cobre "frases" longas.

Saídas usadas depois, por minuto:
- surpresa  = -log p(letra que de fato apareceu | passado)
- entropia  = incerteza do modelo sobre a próxima letra
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(drop))

    def forward(self, x, mask):
        y = self.ln1(x)
        x = x + self.attn(y, y, y, attn_mask=mask, need_weights=False, is_causal=True)[0]
        return x + self.mlp(self.ln2(x))


class MarketGPT(nn.Module):
    def __init__(self, vocab: int, ctx: int, n_layer: int, n_head: int, d: int, drop: float):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList(Block(d, n_head, drop) for _ in range(n_layer))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.register_buffer("mask", torch.triu(torch.full((ctx, ctx), float("-inf")), 1), persistent=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        m = self.mask[:T, :T]
        for b in self.blocks:
            x = b(x, m)
        return self.head(self.ln(x))


def _batch(seq: torch.Tensor, starts: np.ndarray, ctx: int, device):
    ix = torch.from_numpy(starts)
    x = torch.stack([seq[i:i + ctx] for i in ix])
    y = torch.stack([seq[i + 1:i + 1 + ctx] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def _valid_starts(n: int, ctx: int, breaks: np.ndarray | None) -> np.ndarray:
    """Inícios de janela que não atravessam buracos nos dados."""
    starts = np.arange(0, n - ctx - 1)
    if breaks is None or not breaks.any():
        return starts
    cum = np.cumsum(breaks.astype(np.int64))
    return starts[(cum[starts + ctx] - cum[starts]) == 0]


@torch.no_grad()
def _eval_loss(model, seq, starts, ctx, device, amp, n_batches=40, bs=32, seed=0) -> float:
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(n_batches):
        x, y = _batch(seq, rng.choice(starts, bs), ctx, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
        losses.append(F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
    model.train()
    return float(np.mean(losses))


def train_lm(train_letters: np.ndarray, train_breaks: np.ndarray, val_letters: np.ndarray, val_breaks: np.ndarray,
             vocab: int, cfg: dict, device: str, seed: int, ckpt_path: Path, log=print) -> tuple[MarketGPT, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ctx = cfg["context"]
    amp = bool(cfg["amp"]) and device.startswith("cuda")
    model = MarketGPT(vocab, ctx, cfg["n_layer"], cfg["n_head"], cfg["d_model"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    tr = torch.from_numpy(train_letters.astype(np.int64))
    va = torch.from_numpy(val_letters.astype(np.int64))
    tr_starts = _valid_starts(len(tr), ctx, train_breaks)
    va_starts = _valid_starts(len(va), ctx, val_breaks)
    max_steps, warm = cfg["max_steps"], 200
    unigram = np.bincount(train_letters.astype(np.int64), minlength=vocab) + 1
    unigram_loss = float(-(unigram / unigram.sum() * np.log(unigram / unigram.sum())).sum())

    step, best, bad = 0, float("inf"), 0
    if ckpt_path.exists():   # retomar treino interrompido
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if not ck.get("done"):
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            step, best, bad = ck["step"], ck["best"], ck["bad"]
            log(f"    retomando LM do passo {step}")
        else:
            model.load_state_dict(ck["best_model"])
            return model, ck["stats"]
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t0 = time.time()
    history = []
    model.train()
    while step < max_steps:
        lr = cfg["lr"] * min(1.0, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max_steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = _batch(tr, rng.choice(tr_starts, cfg["batch_size"]), ctx, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        if step % cfg["eval_every"] == 0 or step == max_steps:
            vl = _eval_loss(model, va, va_starts, ctx, device, amp)
            history.append({"step": step, "train": float(loss.item()), "val": vl})
            rate = step / (time.time() - t0)
            log(f"    LM passo {step}/{max_steps}: treino {loss.item():.4f}  validação {vl:.4f}  "
                f"(unigrama {unigram_loss:.4f})  {rate:.1f} passos/s")
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                        "bad": bad, "done": False}, ckpt_path)
            if bad >= cfg["patience_evals"]:
                log("    early stopping")
                break
    model.load_state_dict(best_state)
    stats = {"best_val_loss": best, "unigram_loss": unigram_loss, "steps": step,
             "val_gain_over_unigram_nats": unigram_loss - best, "history": history,
             "params": sum(p.numel() for p in model.parameters())}
    torch.save({"best_model": best_state, "done": True, "stats": stats}, ckpt_path)
    return model, stats


@torch.no_grad()
def surprise_entropy(model: MarketGPT, letters: np.ndarray, breaks: np.ndarray, device: str, amp: bool,
                     batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Para cada t: surpresa da letra t dado o passado (<t) e entropia da previsão para t+1.

    Janelas deslizantes com passo ctx/2; cada posição usa pelo menos ctx/2 de
    contexto (exceto no começo). Tudo causal: a saída em t só vê letras <= t.
    """
    model.eval()
    ctx = model.ctx
    half = ctx // 2
    n = len(letters)
    seq = torch.from_numpy(letters.astype(np.int64))
    surprise = np.full(n, np.nan, dtype=np.float32)
    entropy = np.full(n, np.nan, dtype=np.float32)
    starts = list(range(0, max(1, n - ctx + 1), half))
    if starts[-1] + ctx < n:
        starts.append(n - ctx)
    for bi in range(0, len(starts), batch):
        ss = starts[bi:bi + batch]
        x = torch.stack([seq[s:s + ctx] for s in ss]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.startswith("cuda")):
            logits = model(x).float()
        logp = F.log_softmax(logits, -1)
        ent = -(logp.exp() * logp).sum(-1).cpu().numpy()                     # previsão feita em t (para t+1)
        nxt = torch.gather(logp[:, :-1], 2, x[:, 1:, None]).squeeze(-1).cpu().numpy()  # log p(x_{t+1})
        for j, s in enumerate(ss):
            lo = 0 if s == 0 else half                       # posições com contexto suficiente
            entropy[s + lo:s + ctx] = ent[j, lo:]
            surprise[s + 1 + max(lo - 1, 0):s + ctx] = -nxt[j, max(lo - 1, 0):]
    # surpresa logo após um buraco nos dados não tem significado
    surprise[breaks] = np.nan
    return surprise, entropy
