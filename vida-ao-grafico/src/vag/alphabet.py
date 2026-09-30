"""Fase 3 — alfabeto aprendido (VQ-VAE).

Cada minuto (vetor de features) vira uma de K letras. As letras são escolhidas
pelos dados: o codebook é o conjunto de K "estados típicos" que melhor
reconstroem o mercado. Codebook atualizado por EMA; letras mortas são
reiniciadas com estados reais do lote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAQuantizer(nn.Module):
    def __init__(self, k: int, dim: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.k, self.dim, self.decay, self.eps = k, dim, decay, eps
        embed = torch.randn(k, dim) * 0.1
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(k))
        self.register_buffer("embed_sum", embed.clone())
        self.register_buffer("usage", torch.zeros(k))   # uso recente, para reiniciar letras mortas

    def codes(self, z: torch.Tensor) -> torch.Tensor:
        d = (z.pow(2).sum(1, keepdim=True) - 2 * z @ self.embed.t() + self.embed.pow(2).sum(1)[None, :])
        return d.argmin(1)

    def forward(self, z: torch.Tensor):
        idx = self.codes(z)
        q = self.embed[idx]
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.k).type(z.dtype)
                n = onehot.sum(0)
                self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
                self.embed_sum.mul_(self.decay).add_(onehot.t() @ z, alpha=1 - self.decay)
                total = self.cluster_size.sum()
                size = (self.cluster_size + self.eps) / (total + self.k * self.eps) * total
                self.embed.copy_(self.embed_sum / size[:, None])
                self.usage.mul_(0.99).add_(n, alpha=0.01)
        commit = F.mse_loss(z, q.detach())
        q = z + (q - z).detach()   # straight-through
        return q, idx, commit

    @torch.no_grad()
    def restart_dead(self, z: torch.Tensor, threshold: float = 1e-3) -> int:
        dead = (self.usage < threshold).nonzero().flatten()
        if len(dead) == 0:
            return 0
        pick = z[torch.randint(0, len(z), (len(dead),), device=z.device)]
        self.embed[dead] = pick
        self.embed_sum[dead] = pick
        self.cluster_size[dead] = 1.0
        self.usage[dead] = 1.0
        return int(len(dead))


class VQVAE(nn.Module):
    def __init__(self, n_in: int, k: int, latent: int, hidden: int, decay: float):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_in))
        self.vq = EMAQuantizer(k, latent, decay)

    def forward(self, x):
        z = self.enc(x)
        q, idx, commit = self.vq(z)
        return self.dec(q), idx, commit, z


@dataclass
class Alphabet:
    model: VQVAE
    mean: np.ndarray
    std: np.ndarray
    k: int

    def standardize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype("float32")

    @torch.no_grad()
    def encode(self, x: np.ndarray, device: str = "cpu", batch: int = 65536) -> np.ndarray:
        self.model.eval().to(device)
        xs = self.standardize(x)
        out = np.empty(len(xs), dtype=np.int16)
        for i in range(0, len(xs), batch):
            z = self.model.enc(torch.from_numpy(xs[i:i + batch]).to(device))
            out[i:i + batch] = self.model.vq.codes(z).cpu().numpy()
        return out

    def save(self, path) -> None:
        torch.save({"state": self.model.state_dict(), "mean": self.mean, "std": self.std, "k": self.k,
                    "arch": {"n_in": self.model.enc[0].in_features, "latent": self.model.vq.dim,
                             "hidden": self.model.enc[0].out_features}}, path)

    @classmethod
    def load(cls, path) -> "Alphabet":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck["arch"]
        m = VQVAE(a["n_in"], ck["k"], a["latent"], a["hidden"], 0.99)
        m.load_state_dict(ck["state"])
        return cls(m, ck["mean"], ck["std"], ck["k"])


def code_stats(codes: np.ndarray, k: int) -> dict:
    counts = np.bincount(codes.astype(np.int64), minlength=k)
    p = counts / max(counts.sum(), 1)
    nz = p[p > 0]
    return {
        "used_letters": int((counts > 0).sum()),
        "letters_used_1pct_of_uniform": int((p > 0.01 / k).sum()),
        "perplexity": float(math.exp(-(nz * np.log(nz)).sum())),   # nº "efetivo" de letras
        "top10_share": float(np.sort(p)[-10:].sum()),
    }


def train_alphabet(x_train: np.ndarray, cfg: dict, device: str, seed: int, log=print) -> tuple[Alphabet, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean = x_train.mean(0)
    std = x_train.std(0) + 1e-6
    a = Alphabet(None, mean, std, cfg["codebook_size"])
    xs = torch.from_numpy(a.standardize(x_train))
    model = VQVAE(xs.shape[1], cfg["codebook_size"], cfg["latent_dim"], cfg["hidden"], cfg["ema_decay"]).to(device)
    a.model = model
    opt = torch.optim.AdamW(list(model.enc.parameters()) + list(model.dec.parameters()), lr=cfg["lr"])
    bs, n = cfg["batch_size"], len(xs)
    steps_per_epoch = max(1, n // bs)
    total = steps_per_epoch * cfg["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=total)

    # inicializa o codebook com estados reais (evita começar com letras mortas)
    with torch.no_grad():
        init = xs[torch.from_numpy(rng.choice(n, size=min(n, 20 * cfg["codebook_size"]), replace=False))].to(device)
        z0 = model.enc(init)
        model.vq.embed.copy_(z0[torch.randperm(len(z0))[: cfg["codebook_size"]]])
        model.vq.embed_sum.copy_(model.vq.embed)
        model.vq.cluster_size.fill_(1.0)
        model.vq.usage.fill_(1.0)

    step, restarts = 0, 0
    model.train()
    for ep in range(cfg["epochs"]):
        perm = torch.from_numpy(rng.permutation(n))
        tot_rec = 0.0
        for i in range(steps_per_epoch):
            xb = xs[perm[i * bs:(i + 1) * bs]].to(device, non_blocking=True)
            rec, idx, commit, z = model(xb)
            loss_rec = F.mse_loss(rec, xb)
            loss = loss_rec + cfg["commitment"] * commit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tot_rec += loss_rec.item()
            # não reinicia no fim, para o codebook final ser estável
            if step % cfg["dead_code_restart_every"] == 0 and step < 0.8 * total:
                restarts += model.vq.restart_dead(z.detach())
        log(f"    alfabeto época {ep + 1}/{cfg['epochs']}: reconstrução {tot_rec / steps_per_epoch:.4f}")

    codes = a.encode(x_train, device)
    with torch.no_grad():
        sample = torch.from_numpy(a.standardize(x_train[rng.choice(n, size=min(n, 200_000), replace=False)])).to(device)
        model.eval()
        rec, _, _, _ = model(sample)
        rec_mse = F.mse_loss(rec, sample).item()
    stats = {"reconstruction_mse": rec_mse, "dead_code_restarts": restarts, **code_stats(codes, a.k)}
    return a, stats
