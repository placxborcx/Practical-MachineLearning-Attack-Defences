# extraction_attack_optimized.py
"""
Model-Extraction Attack on MNIST (My_MNIST Victim)
==================================================
End-to-end pipeline  +  headless visualisations  +  phase timings.

Artefacts
---------
* confusion_matrices.png
* agreement_matrix.png
* performance_comparison.png
* results.json   (accuracy, agreement, timings)
"""
from __future__ import annotations

import argparse, json, os, random, time
from pathlib import Path
from typing import Dict, List, Tuple
import traceback
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # headless backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

# Import check for victim network
try:
    from a3_mnist import My_MNIST         # victim network
    print("[INFO] Successfully imported My_MNIST")
except ImportError as e:
    print(f"[ERROR] Cannot import My_MNIST: {e}")
    print("[ERROR] Please ensure a3_mnist.py is in the same directory")
    sys.exit(1)

# ─────────────────── config & CLI ───────────────────
CFG: Dict[str, object] = {
    "data_root":      "./data/MNIST",
    "batch_size":     128,
    # victim
    "victim_ckpt":    "target_model.pth",
    "victim_epochs":  5,
    "victim_lr":      1.0,
    # attack
    "n_queries":      8_000,
    "surrogate_epochs": 10,
    "surrogate_lr":   1e-3,
    # misc
    "seed":           2025,
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
}

P = argparse.ArgumentParser()
P.add_argument("--quick",   action="store_true", help="debug (2 epoch / 1 k queries)")
P.add_argument("--retrain", action="store_true", help="force victim retrain")
P.add_argument("--clean", action="store_true", help="clean previous outputs")
ARGS = P.parse_args()

if ARGS.clean:
    print("[INFO] Cleaning previous outputs...")
    files_to_clean = ["confusion_matrices.png", "agreement_matrix.png", 
                     "performance_comparison.png", "results.json", "target_model.pth"]
    for f in files_to_clean:
        if os.path.exists(f):
            os.remove(f)
            print(f"[INFO] Removed {f}")

if ARGS.quick:
    CFG.update(victim_epochs=2, surrogate_epochs=3, n_queries=1_000)  # type: ignore
    print("[INFO] Quick mode enabled")

print(f"[INFO] Using device: {CFG['device']}")

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device(CFG["device"])
Path(CFG["data_root"]).mkdir(parents=True, exist_ok=True)

TIMES: Dict[str, float] = {}
T0_GLOBAL = time.perf_counter()

# ─────────────────── surrogate network ───────────────────
class SurrogateNet(nn.Module):
    """Light 2-conv CNN (~55 k params)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.fc    = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x: torch.Tensor) -> torch.Tensor:          # type: ignore
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        return self.fc(x.flatten(1))

# ─────────────────── data helpers ───────────────────
TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

def get_loaders(bs: int) -> Tuple[DataLoader, DataLoader]:
    tr = datasets.MNIST(CFG["data_root"], train=True,  download=True, transform=TFM)
    te = datasets.MNIST(CFG["data_root"], train=False, download=True, transform=TFM)
    return (
        DataLoader(tr, bs, shuffle=True,  drop_last=True),
        DataLoader(te, bs, shuffle=False, drop_last=False),
    )

@torch.no_grad()
def accuracy(m: nn.Module, ld: DataLoader) -> float:
    m.eval(); ok=tot=0
    for x,y in ld:
        ok += (m(x.to(DEVICE)).argmax(1).cpu()==y).sum().item(); tot+=y.size(0)
    return ok/tot

@torch.no_grad()
def agreement(a: nn.Module, b: nn.Module, ld: DataLoader) -> float:
    a.eval(); b.eval(); mt=tt=0
    for x,_ in ld:
        x=x.to(DEVICE)
        mt += (a(x).argmax(1)==b(x).argmax(1)).sum().item(); tt+=x.size(0)
    return mt/tt

# ─────────────────── victim utils ───────────────────
def _train_victim() -> nn.Module:
    print("[INFO] Training victim model...")
    tr,te=get_loaders(CFG["batch_size"])
    m=My_MNIST().to(DEVICE)
    opt=torch.optim.Adadelta(m.parameters(), lr=CFG["victim_lr"])
    sch=torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.7)
    nll=nn.NLLLoss()
    for ep in range(1, CFG["victim_epochs"]+1):
        m.train()
        for x,y in tr:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad(); nll(m(x),y).backward(); opt.step()
        sch.step(); print(f"    Victim epoch {ep:02d} acc:{accuracy(m,te):.4f}")
    torch.save(m.state_dict(), CFG["victim_ckpt"])
    print(f"[INFO] Victim model saved to {CFG['victim_ckpt']}")
    return m

def load_victim() -> nn.Module:
    if ARGS.retrain:
        print("[Victim] retrain flag – training fresh model.")
        return _train_victim().eval()
    m=My_MNIST().to(DEVICE)
    if os.path.exists(CFG["victim_ckpt"]):
        try:
            m.load_state_dict(torch.load(CFG["victim_ckpt"], map_location=DEVICE))
            print("[Victim] checkpoint loaded.")
        except Exception as e:
            print(f"[Victim] checkpoint load error: {e} – retraining…"); m=_train_victim()
    else:
        print("[Victim] no checkpoint – training…"); m=_train_victim()
    return m.eval()

# ─────────────────── black-box wrapper ───────────────────
class BlackBoxAPI:
    def __init__(self, vict: nn.Module): 
        self.v=vict
        self.query_count = 0
    @torch.no_grad()
    def query(self, x: torch.Tensor, *, logits=False) -> torch.Tensor:
        self.query_count += x.size(0)
        logp=self.v(x.to(DEVICE))
        return logp if logits else torch.exp(logp)

# ─────────────────── build transfer set ───────────────────
def build_query_set(api: BlackBoxAPI, n: int) -> TensorDataset:
    print(f"[INFO] Building query set with {n} samples...")
    base=datasets.MNIST(CFG["data_root"],train=True,download=True,transform=TFM)
    idx=np.random.choice(len(base), n, replace=False)
    ld=DataLoader(Subset(base, idx), CFG["batch_size"], shuffle=False)
    xs:List[torch.Tensor]=[]; ps:List[torch.Tensor]=[]
    for i, (x,_) in enumerate(ld):
        xs.append(x); ps.append(api.query(x))
        if (i+1) % 10 == 0:
            print(f"    Processed {(i+1)*CFG['batch_size']} samples")
    print(f"[INFO] Query set built. Total queries made: {api.query_count}")
    return TensorDataset(torch.cat(xs), torch.cat(ps))

# ─────────────────── train surrogate ───────────────────
def train_surrogate(ds: TensorDataset) -> nn.Module:
    print(f"[INFO] Training surrogate model with {len(ds)} samples...")
    m=SurrogateNet().to(DEVICE)
    ld=DataLoader(ds, CFG["batch_size"], shuffle=True)
    opt=torch.optim.Adam(m.parameters(), lr=CFG["surrogate_lr"])
    kld=nn.KLDivLoss(reduction="batchmean")
    for ep in range(1, CFG["surrogate_epochs"]+1):
        m.train(); run=0.0
        for x,p in ld:
            x,p=x.to(DEVICE),p.to(DEVICE)
            opt.zero_grad(); loss=kld(F.log_softmax(m(x),1),p); loss.backward(); opt.step()
            run+=loss.item()*x.size(0)
        print(f"    Surrogate epoch {ep:02d} KL:{run/len(ds):.4f}")
    print("[INFO] Surrogate model training completed")
    return m.eval()

# ─────────────────── evaluation + plots ───────────────────
@torch.no_grad()
def eval_and_plot(v: nn.Module, s: nn.Module) -> Dict[str,float]:
    print("[INFO] Evaluating models and generating plots...")
    _,te=get_loaders(CFG["batch_size"])
    y_t,y_v,y_s=[],[],[]
    for x,y in te:
        x=x.to(DEVICE)
        y_t+=y.tolist()
        y_v+=v(x).argmax(1).cpu().tolist()
        y_s+=s(x).argmax(1).cpu().tolist()
    y_t=np.array(y_t); y_v=np.array(y_v); y_s=np.array(y_s)
    acc_v=(y_v==y_t).mean(); acc_s=(y_s==y_t).mean(); agree=(y_v==y_s).mean()

    try:
        # Confusion matrices
        cm_v=confusion_matrix(y_t,y_v); cm_s=confusion_matrix(y_t,y_s)
        fig,ax=plt.subplots(1,2,figsize=(14,6))
        for a,cm,t in zip(ax,[cm_v,cm_s],["Victim","Surrogate"]):
            im=a.imshow(cm,cmap="Blues")
            a.set_title(f"{t} Confusion"); a.set_xlabel("Pred"); a.set_ylabel("True")
            a.set_xticks(range(10)); a.set_yticks(range(10))
            for i in range(10):
                for j in range(10):
                    a.text(j,i,int(cm[i,j]),ha="center",va="center",fontsize=7)
        fig.colorbar(im,ax=ax.ravel().tolist(),shrink=0.6); fig.tight_layout()
        fig.savefig("confusion_matrices.png",dpi=150)
        plt.close(fig)
        print("[INFO] Confusion matrices saved")

        # Agreement matrix
        agree_mat=confusion_matrix(y_v,y_s,labels=range(10))
        plt.figure(figsize=(6,5))
        plt.imshow(agree_mat,cmap="Greens"); plt.title("Agreement (V vs S)")
        plt.xlabel("Surrogate"); plt.ylabel("Victim")
        plt.xticks(range(10)); plt.yticks(range(10))
        for i in range(10):
            for j in range(10):
                plt.text(j,i,int(agree_mat[i,j]),ha="center",va="center",fontsize=7)
        plt.tight_layout(); plt.savefig("agreement_matrix.png",dpi=150)
        plt.close()
        print("[INFO] Agreement matrix saved")

        # Performance comparison
        m=["Accuracy","Agreement"]; v_vals=[acc_v*100,100]; s_vals=[acc_s*100,agree*100]
        x=np.arange(len(m)); w=0.35
        plt.figure(figsize=(8,5))
        plt.bar(x-w/2,v_vals,w,label="Victim",alpha=0.7)
        plt.bar(x+w/2,s_vals,w,label="Surrogate",alpha=0.7)
        plt.xticks(x,m); plt.ylabel("Percent (%)")
        plt.title("Extraction Attack Comparison"); plt.legend(); plt.grid(axis="y",alpha=0.3)
        plt.tight_layout(); plt.savefig("performance_comparison.png",dpi=150)
        plt.close()
        print("[INFO] Performance comparison saved")

    except Exception as e:
        print(f"[ERROR] Failed to generate plots: {e}")
        traceback.print_exc()

    res={"victim_acc":acc_v,"surrogate_acc":acc_s,"agreement":agree}
    return res

# ─────────────────── main orchestration ───────────────────
def main() -> None:
    try:
        print("[INFO] Starting model extraction attack...")
        
        # victim prepare
        print("\n[Phase 1] Preparing victim model...")
        t=time.perf_counter(); vict=load_victim(); TIMES["victim_prepare_s"]=time.perf_counter()-t
        print(f"[INFO] Victim preparation completed in {TIMES['victim_prepare_s']:.1f}s")

        # query phase
        print("\n[Phase 2] Querying victim model...")
        api=BlackBoxAPI(vict)
        t=time.perf_counter(); qset=build_query_set(api,int(CFG["n_queries"])); TIMES["query_phase_s"]=time.perf_counter()-t
        print(f"[INFO] Query phase completed in {TIMES['query_phase_s']:.1f}s")

        # surrogate
        print("\n[Phase 3] Training surrogate model...")
        t=time.perf_counter(); surro=train_surrogate(qset); TIMES["surrogate_train_s"]=time.perf_counter()-t
        print(f"[INFO] Surrogate training completed in {TIMES['surrogate_train_s']:.1f}s")

        # evaluate + plots
        print("\n[Phase 4] Evaluation and visualization...")
        t=time.perf_counter(); res=eval_and_plot(vict,surro); TIMES["evaluation_s"]=time.perf_counter()-t
        print(f"[INFO] Evaluation completed in {TIMES['evaluation_s']:.1f}s")

        TIMES["total_runtime_s"]=time.perf_counter()-T0_GLOBAL

        print("\n" + "="*50)
        print("[EVALUATION RESULTS]")
        print("="*50)
        for k,v in res.items(): 
            print(f"  {k:20s}: {v*100:6.2f}%")

        print("\n" + "="*50)
        print("[TIMING SUMMARY]")
        print("="*50)
        for k,v in TIMES.items(): 
            print(f"  {k:20s}: {v:6.1f}s")

        # Save results
        final_results = {**res, "timings": TIMES}
        with open("results.json","w",encoding="utf-8") as fp:
            json.dump(final_results, fp, indent=2)
        print(f"\n[INFO] Results saved to results.json")
        
        # Check if files were created
        output_files = ["confusion_matrices.png", "agreement_matrix.png", 
                       "performance_comparison.png", "results.json"]
        print("\n[OUTPUT FILES]")
        for f in output_files:
            if os.path.exists(f):
                size = os.path.getsize(f)
                print(f"  ✓ {f} ({size} bytes)")
            else:
                print(f"  ✗ {f} (missing)")

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__=="__main__": 
    main()