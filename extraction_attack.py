# extraction_attack_optimized.py
"""
Model Extraction Attack on MNIST – *My_MNIST Victim Edition*
============================================================

*Victim network*  → `My_MNIST`   (imported from **a3_mnist.py**)  
*Surrogate network* → `SurrogateNet` (簡化 2‑conv CNN)。

此版本完全沿用你作業提供的 `My_MNIST` 權重檔 `target_model.pth`；若檔案
不存在或與網路形狀不符，腳本會自動重新**以 NLLLoss 訓練 My_MNIST** 並覆寫
checkpoint。

使用方式
---------
```bash
$ python extraction_attack_optimized.py          # 如已有 target_model.pth 直接載入
$ python extraction_attack_optimized.py --retrain  # 永遠重新訓練 Victim
```

CLI 旗標 `--quick` 仍可做 2‑epoch 測試 (for debug)。
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
# 0. ––––– Import Victim architecture (from assignment)                     #
# --------------------------------------------------------------------------- #

from a3_mnist import My_MNIST  # ← 你作業中的 LeNet 變體

# --------------------------------------------------------------------------- #
# 0‑a. Configuration & CLI                                                  #
# --------------------------------------------------------------------------- #

CFG = {
    "data_root": "./data/MNIST",
    "batch_size": 128,
    # victim
    "victim_ckpt": "target_model.pth",
    "victim_epochs": 5,
    "victim_lr": 1.0,           # 與原 a3_mnist 預設一致 (Adadelta lr)
    # attack
    "n_queries": 8_000,
    "surrogate_epochs": 10,
    "surrogate_lr": 1e-3,
    # misc
    "seed": 2025,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

parser = argparse.ArgumentParser()
parser.add_argument("--quick",   action="store_true", help="⌚ 2‑epoch quick run (debug)")
parser.add_argument("--retrain", action="store_true", help="⚒ ignore existing checkpoint")
ARGS = parser.parse_args()

if ARGS.quick:
    CFG.update(victim_epochs=2, surrogate_epochs=3, n_queries=1_000)

# --------------------------------------------------------------------------- #
# 0‑b.  Reproducibility helpers                                             #
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
# 1. ––––– Surrogate architecture                                           #
# --------------------------------------------------------------------------- #

class SurrogateNet(nn.Module):
    """2‑conv → FC：容量小於 My_MNIST，但足以模仿。"""
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.fc    = nn.Linear(32 * 7 * 7, n_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        return self.fc(x.flatten(1))

# --------------------------------------------------------------------------- #
# 2. ––––– Data loaders & metrics                                           #
# --------------------------------------------------------------------------- #

# 使用與 a3_mnist 相同的 Normalize 參數
T_MNIST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

def get_loaders(bs: int = 128) -> Tuple[DataLoader, DataLoader]:
    tr = datasets.MNIST(CFG["data_root"], train=True,  download=True, transform=T_MNIST)
    te = datasets.MNIST(CFG["data_root"], train=False, download=True, transform=T_MNIST)
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
# 3. ––––– Victim model utils                                               #
# --------------------------------------------------------------------------- #

def train_victim() -> nn.Module:
    """(Re)train My_MNIST with NLLLoss, mimic original assignment script."""
    tr, te = get_loaders(CFG["batch_size"])
    m   = My_MNIST().to(device)
    opt = torch.optim.Adadelta(m.parameters(), lr=CFG["victim_lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.7)
    nll = nn.NLLLoss()
    print("\n[Victim] training My_MNIST …")
    for ep in range(1, CFG["victim_epochs"] + 1):
        m.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = nll(m(x), y); loss.backward(); opt.step()
        scheduler.step()
        print(f"  epoch {ep:02d} | test acc = {accuracy(m, te):.4f}")
    torch.save(m.state_dict(), CFG["victim_ckpt"])
    print("[Victim] saved →", CFG["victim_ckpt"])
    return m


def load_victim() -> nn.Module:
    if ARGS.retrain:
        print("[Victim] --retrain flag set – ignoring existing checkpoint.")
        return train_victim().eval()

    m = My_MNIST().to(device)
    if os.path.exists(CFG["victim_ckpt"]):
        try:
            m.load_state_dict(torch.load(CFG["victim_ckpt"], map_location=device))
            print("[Victim] checkpoint loaded.")
        except (RuntimeError, KeyError) as e:
            print("[Warning] checkpoint incompatible →", e.args[0].split("\n")[0])
            m = train_victim()
    else:
        m = train_victim()
    return m.eval()

# --------------------------------------------------------------------------- #
# 4. ––––– Black‑box API                                                    #
# --------------------------------------------------------------------------- #

class BlackBoxAPI:
    """Wraps victim; returns probabilities by default."""
    def __init__(self, victim: nn.Module):
        self.victim = victim
    @torch.no_grad()
    def query(self, x: torch.Tensor, *, logits: bool = False) -> torch.Tensor:
        log_p = self.victim(x.to(device))       # My_MNIST → log‑probs
        return log_p if logits else torch.exp(log_p)

# --------------------------------------------------------------------------- #
# 5. ––––– Query set builder                                                #
# --------------------------------------------------------------------------- #

def build_query_set(api: BlackBoxAPI, n: int) -> TensorDataset:
    base = datasets.MNIST(CFG["data_root"], train=True, download=True, transform=T_MNIST)
    idx  = np.random.choice(len(base), n, replace=False)
    ld   = DataLoader(Subset(base, idx), CFG["batch_size"], shuffle=False)
    xs: List[torch.Tensor] = []
    ps: List[torch.Tensor] = []
    for x, _ in ld:
        xs.append(x)
        ps.append(api.query(x))
    return TensorDataset(torch.cat(xs), torch.cat(ps))

# --------------------------------------------------------------------------- #
# 6. ––––– Surrogate training (KL distillation)                             #
# --------------------------------------------------------------------------- #

def train_surrogate(ds: TensorDataset) -> nn.Module:
    m   = SurrogateNet().to(device)
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
# 7. ––––– Evaluation & dump                                                #
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
# 8. ––––– Main                                                             #
# --------------------------------------------------------------------------- #

def main() -> None:
    victim    = load_victim()
    api       = BlackBoxAPI(victim)
    queries   = build_query_set(api, CFG["n_queries"])
    surrogate = train_surrogate(queries)
    evaluate(victim, surrogate)

if __name__ == "__main__":
    main()
