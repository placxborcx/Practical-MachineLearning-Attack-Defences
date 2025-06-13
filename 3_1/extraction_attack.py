# extraction_attack_optimized.py
"""
Model Extraction Attack on MNIST – *Complete Edition*
=====================================================

‣ Added **robust checkpoint loading**: if an old `target_model.pth` was saved
  with a *different* architecture (hence key/shape mismatch), the script now
  **automatically retrains** the victim ConvNet and overwrites the checkpoint.
  You can also force fresh training with `--retrain`.

Other phases (query ‑ surrogate ‑ evaluation) remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

# --------------------------------------------------------------------------- #
# 0. ––––– Configuration & CLI
# --------------------------------------------------------------------------- #

CFG = {
    "data_root": "./data/MNIST",
    "batch_size": 128,
    # victim
    "victim_ckpt": "target_model.pth",
    "victim_epochs": 5,
    "victim_lr": 1e-3,
    # attack
    "n_queries": 8_000,
    "surrogate_epochs": 10,
    "surrogate_lr": 1e-3,
    # misc
    "seed": 2025,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

parser = argparse.ArgumentParser()
parser.add_argument("--quick",   action="store_true", help="⌚ 2‑epoch quick run (debug)")
parser.add_argument("--retrain", action="store_true", help="⚒ ignore existing checkpoint")
ARGS = parser.parse_args()

if ARGS.quick:
    CFG.update(victim_epochs=2, surrogate_epochs=3, n_queries=1_000)

# --------------------------------------------------------------------------- #
# 0‑b.  Reproducibility helpers
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG["seed"])

device = torch.device(CFG["device"])
Path(CFG["data_root"]).mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1. ––––– Model definition
# --------------------------------------------------------------------------- #

class ConvNet(nn.Module):
    """2 × Conv → FC.  (≈ My_MNIST)"""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.fc    = nn.Linear(32 * 7 * 7, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.flatten(1)
        return self.fc(x)

# --------------------------------------------------------------------------- #
# 2. ––––– Data loaders & metrics
# --------------------------------------------------------------------------- #

def get_loaders(bs: int = 128) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([transforms.ToTensor()])
    tr = datasets.MNIST(CFG["data_root"], train=True,  download=True, transform=tfm)
    te = datasets.MNIST(CFG["data_root"], train=False, download=True, transform=tfm)
    return (
        DataLoader(tr, bs, shuffle=True,  drop_last=True),
        DataLoader(te, bs, shuffle=False, drop_last=False),
    )

@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval(); ok = tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        ok  += (model(x).argmax(1) == y).sum().item()
        tot += y.size(0)
    return ok / tot

@torch.no_grad()
def agreement(a: nn.Module, b: nn.Module, loader: DataLoader) -> float:
    a.eval(); b.eval(); match = tot = 0
    for x, _ in loader:
        x = x.to(device)
        match += (a(x).argmax(1) == b(x).argmax(1)).sum().item()
        tot   += x.size(0)
    return match / tot

# --------------------------------------------------------------------------- #
# 3. ––––– Victim model (robust load / train)
# --------------------------------------------------------------------------- #

def train_victim() -> nn.Module:
    tr, te = get_loaders(CFG["batch_size"])
    m = ConvNet().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=CFG["victim_lr"])
    ce  = nn.CrossEntropyLoss()
    print("\n[Victim] training …")
    for ep in range(1, CFG["victim_epochs"] + 1):
        m.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); ce(m(x), y).backward(); opt.step()
        print(f"  epoch {ep:02d} | test acc = {accuracy(m, te):.4f}")
    torch.save(m.state_dict(), CFG["victim_ckpt"])
    print("[Victim] saved →", CFG["victim_ckpt"])
    return m


def load_victim() -> nn.Module:
    if ARGS.retrain:
        print("[Victim] --retrain flag set – ignoring existing checkpoint.")
        return train_victim().eval()

    m = ConvNet().to(device)
    if os.path.exists(CFG["victim_ckpt"]):
        try:
            m.load_state_dict(torch.load(CFG["victim_ckpt"], map_location=device))
            print("[Victim] checkpoint loaded.")
        except (RuntimeError, KeyError) as e:
            print("[Warning] checkpoint incompatible →", e.args[0].split("\n")[0])
            print("[Victim] retraining with current architecture …")
            m = train_victim()
    else:
        m = train_victim()
    m.eval()
    return m

# --------------------------------------------------------------------------- #
# 4. ––––– Black‑box API
# --------------------------------------------------------------------------- #

class BlackBoxAPI:
    def __init__(self, victim: nn.Module):
        self.victim = victim
    @torch.no_grad()
    def query(self, x: torch.Tensor, *, logits: bool = False) -> torch.Tensor:
        out = self.victim(x.to(device))
        return out if logits else F.softmax(out, 1)

# --------------------------------------------------------------------------- #
# 5. ––––– Query set builder
# --------------------------------------------------------------------------- #

def build_query_set(api: BlackBoxAPI, n: int) -> TensorDataset:
    tfm = transforms.Compose([transforms.ToTensor()])
    base = datasets.MNIST(CFG["data_root"], train=True, download=True, transform=tfm)
    idx  = np.random.choice(len(base), n, replace=False)
    ld   = DataLoader(Subset(base, idx), CFG["batch_size"], shuffle=False)
    xs: List[torch.Tensor] = []
    ps: List[torch.Tensor] = []
    for x, _ in ld:
        xs.append(x)
        ps.append(api.query(x))
    return TensorDataset(torch.cat(xs), torch.cat(ps))

# --------------------------------------------------------------------------- #
# 6. ––––– Surrogate training
# --------------------------------------------------------------------------- #

def train_surrogate(ds: TensorDataset) -> nn.Module:
    m   = ConvNet().to(device)
    ld  = DataLoader(ds, CFG["batch_size"], shuffle=True)
    opt = torch.optim.Adam(m.parameters(), lr=CFG["surrogate_lr"])
    kld = nn.KLDivLoss(reduction="batchmean")
    print("\n[Surrogate] training …")
    for ep in range(1, CFG["surrogate_epochs"] + 1):
        m.train(); tot = 0.0
        for x, p in ld:
            x, p = x.to(device), p.to(device)
            opt.zero_grad()
            loss = kld(F.log_softmax(m(x), 1), p)
            loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
        print(f"  epoch {ep:02d} | KL = {tot / len(ds):.4f}")
    return m.eval()

# --------------------------------------------------------------------------- #
# 7. ––––– Evaluation
# --------------------------------------------------------------------------- #

def evaluate(victim: nn.Module, surrogate: nn.Module) -> None:
    _, te = get_loaders(CFG["batch_size"])
    res = {
        "victim_acc":    accuracy(victim, te),
        "surrogate_acc": accuracy(surrogate, te),
        "agreement":     agreement(victim, surrogate, te),
    }
    print("\n[Evaluation]")
    for k, v in res.items():
        print(f"  {k:14s}: {v * 100:.2f}%")
    with open("results.json", "w", encoding="utf-8") as fp:
        json.dump(res, fp, indent=2)
    print("[Saved] results.json")

# --------------------------------------------------------------------------- #
# 8. ––––– Main
# --------------------------------------------------------------------------- #

def main() -> None:
    victim    = load_victim()
    api       = BlackBoxAPI(victim)
    queries   = build_query_set(api, CFG["n_queries"])
    surrogate = train_surrogate(queries)
    evaluate(victim, surrogate)

if __name__ == "__main__":
    main()
