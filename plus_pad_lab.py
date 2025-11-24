# plus_pad_lab_final_fixed.py
import os
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATA_ROOT = Path("./PLUS")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
K = 5
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------
# 1. LOAD IMAGES
# ------------------------------------------------------------
def load_plus_images():
    real  = [p for p in (DATA_ROOT / "real").rglob("*.png") if p.is_file()]
    spoof = [p for p in (DATA_ROOT / "spoof").rglob("*.png") if p.is_file()]
    print(f"Real: {len(real)}, Spoof: {len(spoof)}")
    return real, spoof

def get_subject_id(path: Path) -> str:
    return path.name.split("_")[3]

# ------------------------------------------------------------
# 2. BUILD FOLDS (LOSO if <5 subjects)
# ------------------------------------------------------------
def debug_and_build():
    real_all, spoof_all = load_plus_images()
    
    real_map = defaultdict(list)
    spoof_map = defaultdict(list)
    
    for p in real_all:
        real_map[get_subject_id(p)].append(p)
    for p in spoof_all:
        spoof_map[get_subject_id(p)].append(p)
    
    common = set(real_map.keys()) & set(spoof_map.keys())
    print(f"Subjects with both real and spoof: {len(common)}")
    
    if len(common) == 0:
        print("No overlapping subjects!")
        return None, None
    
    if len(common) < 5:
        print(f"Only {len(common)} subject(s) → using LOSO")
        folds = []
        for test_subj in common:
            train_subj = [s for s in common if s != test_subj]
            tr_r = [p for s in train_subj for p in real_map[s]]
            tr_s = [p for s in train_subj for p in spoof_map[s]]
            te_r = real_map[test_subj]
            te_s = spoof_map[test_subj]
            folds.append({'tr_real': tr_r, 'tr_spoof': tr_s, 'te_real': te_r, 'te_spoof': te_s})
        return folds, "LOSO"
    else:
        print("Using 5-fold CV")
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        subjects = list(common)
        folds = []
        for train_idx, test_idx in skf.split(subjects, [0]*len(subjects)):
            train_subj = [subjects[i] for i in train_idx]
            test_subj  = [subjects[i] for i in test_idx]
            tr_r = [p for s in train_subj for p in real_map[s]]
            tr_s = [p for s in train_subj for p in spoof_map[s]]
            te_r = [p for s in test_subj for p in real_map[s]]
            te_s = [p for s in test_subj for p in spoof_map[s]]
            folds.append({'tr_real': tr_r, 'tr_spoof': tr_s, 'te_real': te_r, 'te_spoof': te_s})
        return folds, "5-fold"

# ------------------------------------------------------------
# 3. EXTRACTOR – MODERN
# ------------------------------------------------------------
weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)
extractor = nn.Sequential(*list(model.children())[:-1])
extractor.eval().to(DEVICE)
print(f"ResNet-50 loaded with {weights}")

transform = T.Compose([
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------------------------------------------------
# 4. DATASET – HANDLES GRAYSCALE
# ------------------------------------------------------------
class Dataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = read_image(str(self.paths[i])).float() / 255.0
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3, :, :]
        if self.transform:
            img = self.transform(img)
        return img

def extract(paths):
    if not paths:
        return np.zeros((0, 2048))
    ds = Dataset(paths, transform)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=0, pin_memory=True)
    feats = []
    with torch.no_grad():
        for x in dl:
            f = extractor(x.to(DEVICE)).flatten(1)
            feats.append(f.cpu().numpy())
    return np.vstack(feats)

# ------------------------------------------------------------
# 5. RUN
# ------------------------------------------------------------
def run():
    folds, mode = debug_and_build()
    if not folds:
        return
    
    print(f"\nUsing {mode} validation\n")
    
    for name, ratio in [
        ("Baseline", 1.0),
        ("Step1", 0.5),
        ("Step2", 0.25),
    ]:
        print(f"--- {name} ({int(ratio*100)}% data) ---")
        acers = []
        for fold in folds:
            tr_r = fold['tr_real'][:int(len(fold['tr_real'])*ratio)]
            tr_s = fold['tr_spoof'][:int(len(fold['tr_spoof'])*ratio)]
            X_tr = extract(tr_r + tr_s)
            X_te = extract(fold['te_real'] + fold['te_spoof'])
            y_tr = [0]*len(tr_r) + [1]*len(tr_s)
            y_te = [0]*len(fold['te_real']) + [1]*len(fold['te_spoof'])
            
            if len(set(y_tr)) < 2:
                print("  Warning: Only one class in training → skipping")
                continue
                
            knn = KNeighborsClassifier(n_neighbors=K, metric='cosine')
            knn.fit(X_tr, y_tr)
            pred = knn.predict(X_te)
            tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
            apcer = fp/(fp+tn) if (fp+tn) else 0.0
            bpcer = fn/(fn+tp) if (fn+tp) else 0.0
            acer = (apcer + bpcer)/2
            acers.append(acer)
        if acers:
            print(f"ACER = {np.mean(acers):.4f} ± {np.std(acers):.4f}\n")
        else:
            print("No valid folds\n")

if __name__ == "__main__":
    run()