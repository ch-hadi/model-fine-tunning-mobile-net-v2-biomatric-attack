import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

# OpenMP runtime error fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
DATASET_NAME = "IDIAP"
DEVICE = torch.device('cpu') # Specifying CPU for laptop
BATCH_SIZE = 16 

PATHS = {
    "SCUT": {
        "real": r"D:\Study\image processing lab\ipl\SCUT\full\train",
        "syn": r"D:\Study\image processing lab\ipl\VAE-diff\SCUT"
    },
    "PLUS": {
        "real": r"D:\Study\image processing lab\ipl\PLUS",
        "syn": r"D:\Study\image processing lab\ipl\VAE-diff\PLUS_matched"
    },
    "IDIAP": {
        "real": r"D:\Study\image processing lab\ipl\IDIAP\full\train",
        "syn": r"D:\Study\image processing lab\ipl\VAE-diff\IDIAP"
    }
    
}
REAL_ROOT = PATHS[DATASET_NAME]["real"]
SYN_ROOT = PATHS[DATASET_NAME]["syn"]

# -----------------------------
# 2. DATA HANDLING
# -----------------------------
class PADDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(r["path"]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, r["label"]

def extract_id(filename):
    m = re.search(r'(\d+)', filename)
    return m.group(1) if m else "unknown"

def build_indices():
    real_records = []
    for label, folder in [(0, "real"), (1, "spoof")]:
        base = os.path.join(REAL_ROOT, folder)
        if not os.path.exists(base): continue
        for dp, _, fs in os.walk(base):
            for f in fs:
                if f.lower().endswith(('.png', '.jpg', '.bmp')):
                    subj = os.path.basename(dp) if os.path.basename(dp).isdigit() else extract_id(f)
                    real_records.append({"path": os.path.join(dp, f), "label": label, "subject": subj})
    
    syn_records = []
    syn_path = os.path.join(SYN_ROOT, "spoof", "samples")
    if os.path.exists(syn_path):
        for f in os.listdir(syn_path):
            if f.lower().endswith(('.png', '.jpg', '.bmp')):
                syn_records.append({"path": os.path.join(syn_path, f), "label": 1, "subject": extract_id(f.replace("sample", ""))})
    return real_records, syn_records
def print_loaded_images(real_records, syn_records):
    real_count = len(real_records)
    syn_count = len(syn_records)
    print(f"Loaded {DATASET_NAME}: {real_count} Real, {syn_count} Synthetic")
# -----------------------------
# 3. FEATURE EXTRACTION
# -----------------------------

def get_features(model, records):
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    loader = DataLoader(PADDataset(records, transform), batch_size=BATCH_SIZE)
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for img_batch, label_batch in tqdm(loader, desc="Extracting Features (CPU)", leave=False):
            f = model(img_batch.to(DEVICE))
            # MobileNetV2 gives 1280-dim vector after pooling
            feats.append(f.squeeze().numpy())
            labels.extend(label_batch.numpy())
    return np.vstack(feats), np.array(labels)

# -----------------------------
# 4. EXPERIMENT STEPS
# -----------------------------

def run_step(real_data, syn_data, step_num):
    print(f"\n--- RUNNING {DATASET_NAME} STEP {step_num} (KNN) ---")
    
    model = models.mobilenet_v2(weights='DEFAULT')
    model.classifier = nn.Identity() 
    
    kf = GroupKFold(n_splits=5)
    subjs = [r["subject"] for r in real_data]
    labels = [r["label"] for r in real_data]
    
    # Lists to store metrics for all folds
    fold_acers = []
    fold_apcers = []
    fold_bpcers = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(real_data, labels, subjs), 1):
        real_tr = [real_data[i] for i in train_idx]
        test_set = [real_data[i] for i in test_idx]
        tr_subs = sorted(list(set(r["subject"] for r in real_tr)))
        
        train_final = []
        if step_num == 3:
            s1 = set(tr_subs[:len(tr_subs)//4])
            s2 = set(tr_subs[len(tr_subs)//4 : 2*len(tr_subs)//4])
            train_final = [r for r in real_tr if r["subject"] in s1]
            train_final += [r for r in real_tr if (r["subject"] in s2 and r["label"] == 0)]
            train_final += [s for s in syn_data if s["subject"] in s2]
        
        elif step_num == 4:
            s1 = set(tr_subs[:len(tr_subs)//4])
            others = set(tr_subs[len(tr_subs)//4:])
            train_final = [r for r in real_tr if r["subject"] in s1]
            train_final += [r for r in real_tr if (r["subject"] in others and r["label"] == 0)]
            train_final += [s for s in syn_data if s["subject"] in others]
            
        elif step_num == 5:
            train_final = [r for r in real_tr if r["label"] == 0]
            train_final += [s for s in syn_data if s["subject"] in set(tr_subs)]

        X_tr, y_tr = get_features(model, train_final)
        X_te, y_te = get_features(model, test_set)
        
        knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
        knn.fit(X_tr, y_tr)
        preds = knn.predict(X_te)
        
        # --- Metrics Calculation ---
        real_mask, fake_mask = (y_te == 0), (y_te == 1)
        
        # BPCER (Bona Fide Presentation Classification Error Rate) - Real classified as Spoof
        bpcer = np.sum(preds[real_mask] == 1) / np.sum(real_mask) if np.sum(real_mask) > 0 else 0
        # APCER (Attack Presentation Classification Error Rate) - Spoof classified as Real
        apcer = np.sum(preds[fake_mask] == 0) / np.sum(fake_mask) if np.sum(fake_mask) > 0 else 0
        # ACER (Average Classification Error Rate)
        acer = (apcer + bpcer) / 2
        
        fold_apcers.append(apcer)
        fold_bpcers.append(bpcer)
        fold_acers.append(acer)
        
        print(f"Fold {fold} | APCER: {apcer:.4f} | BPCER: {bpcer:.4f} | ACER: {acer:.4f}")

    # Final Average Results
    print("-" * 30)
    print(f">> STEP {step_num} FINAL RESULTS <<")
    print(f"AVG APCER: {np.mean(fold_apcers):.4f}")
    print(f"AVG BPCER: {np.mean(fold_bpcers):.4f}")
    print(f"AVG ACER : {np.mean(fold_acers):.4f}")
    print("-" * 30)
if __name__ == "__main__":
    r_data, s_data = build_indices()
    print_loaded_images(r_data, s_data)
    if r_data and s_data:
        for s in [3, 4, 5]:
            run_step(r_data, s_data, s)