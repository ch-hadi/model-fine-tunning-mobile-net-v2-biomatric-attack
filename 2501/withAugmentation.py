import os, re, random, copy, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
DATASET_NAME = "PLUS"  # Options: "SCUT", "PLUS", "IDIAP"


PATHS = {
    "SCUT": {
        "real": r"D:\Study\image processing lab\ipl\SCUT\full\train",
        "syn": r"D:\Study\image processing lab\ipl\VAE-diff\SCUT"
    },
    "PLUS": {
        "real": r"D:\Study\image processing lab\ipl\PLUS",
        "syn": r"D:\Study\image processing lab\ipl\VAE-diff\PLUS-matched"
    },
    "IDIAP": {
        "real": r"D:\Study\image processing lab\ipl\IDIAP",
        "syn": r"D:\Study\image processing lab\ipl\VAE-diff\IDIAP"
    }
}

REAL_ROOT = PATHS[DATASET_NAME]["real"]
SYN_ROOT = PATHS[DATASET_NAME]["syn"]

BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# 2. DATA UTILS & ID EXTRACTION
# -----------------------------
def extract_id(filename, dataset_type):
    if dataset_type == "SCUT":
        m = re.match(r'(\d+)', filename)
        return m.group(1) if m else "unknown"
    elif dataset_type == "PLUS":
        # PLUS-FV3-Laser_PALMAR_001_01_02_01.png -> 001
        parts = filename.split('_')
        return parts[2] if len(parts) > 2 else "unknown"
    elif dataset_type == "IDIAP":
        m = re.match(r'(\d+)', filename)
        return m.group(1) if m else "unknown"
    return "unknown"

def build_real_index(root, d_type):
    records = []
    for label, folder in [(0, "real"), (1, "spoof")]:
        base = os.path.join(root, folder)
        if not os.path.exists(base): continue
        for dp, _, fs in os.walk(base):
            for f in fs:
                if f.lower().endswith(('.png', '.jpg', '.bmp')):
                    path = os.path.join(dp, f)
                    subj = os.path.basename(dp) if os.path.basename(dp).isdigit() else extract_id(f, d_type)
                    records.append({"path": path, "label": label, "subject": subj})
    return records

def build_syn_index(syn_root, d_type):
    records = []
    syn_path = os.path.join(syn_root, "spoof", "samples")
    if os.path.exists(syn_path):
        files = [f for f in os.listdir(syn_path) if f.lower().endswith(('.png', '.jpg', '.bmp'))]
        for f in files:
            path = os.path.join(syn_path, f)
            clean_name = f.replace("sample", "").replace("Sample", "")
            subj = extract_id(clean_name, d_type)
            records.append({"path": path, "label": 1, "subject": subj})
    return records

class PADDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        try:
            img = Image.open(r["path"]).convert("RGB")
            if self.transform: img = self.transform(img)
            return img, r["label"]
        except Exception as e:
            print(f"Error loading {r['path']}: {e}")
            return torch.zeros(3, 224, 224), r["label"]

# -----------------------------
# 3. METRICS & TRAINING
# -----------------------------
def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    real_mask, fake_mask = (y_true == 0), (y_true == 1)
    acc = np.mean(y_true == y_pred)
    bpcer = np.sum(y_pred[real_mask] == 1) / np.sum(real_mask) if np.sum(real_mask) > 0 else 0
    apcer = np.sum(y_pred[fake_mask] == 0) / np.sum(fake_mask) if np.sum(fake_mask) > 0 else 0
    return {'accuracy': acc, 'apcer': apcer, 'bpcer': bpcer, 'acer': (apcer + bpcer) / 2}

def train_and_eval(train_records, test_records):
    # Training Augmentation: To help Step 5 generalize
    train_transform = T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    test_transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_loader = DataLoader(PADDataset(train_records, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(PADDataset(test_records, test_transform), batch_size=BATCH_SIZE, shuffle=False)
    
    model = models.mobilenet_v2(weights='DEFAULT')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = model(imgs.to(DEVICE))
            preds = torch.max(outputs, 1)[1]
            all_y.extend(labels.numpy())
            all_p.extend(preds.cpu().numpy())
            
    return calculate_metrics(all_y, all_p)

# -----------------------------
# 4. EXPERIMENT ENGINE
# -----------------------------
def run_experiment_step(real_records, syn_records, step_num):
    print(f"\n{'='*15} RUNNING {DATASET_NAME} - STEP {step_num} {'='*15}")
    kf = GroupKFold(n_splits=5)
    labels = [r["label"] for r in real_records]
    subjs = [r["subject"] for r in real_records]
    
    step_results = []
    # Cross-validation starts
    for fold, (train_idx, test_idx) in enumerate(kf.split(real_records, labels, subjs), 1):
        real_train = [real_records[i] for i in train_idx]
        test_set = [real_records[i] for i in test_idx]
        tr_subjs = sorted(list(set(r["subject"] for r in real_train)))
        train_final = []

        if step_num == 3:
            s1_subs = set(tr_subjs[:len(tr_subjs)//4]) # Fold 1 (Real BF + Real PA)
            s2_subs = set(tr_subjs[len(tr_subjs)//4 : 2*len(tr_subjs)//4]) # Fold 2 (Real BF + Syn PA)
            for r in real_train:
                if r["subject"] in s1_subs: train_final.append(r)
                if r["subject"] in s2_subs and r["label"] == 0: train_final.append(r)
            for s in syn_records:
                if s["subject"] in s2_subs: train_final.append(s)

        elif step_num == 4:
            s1_subs = set(tr_subjs[:len(tr_subjs)//4]) # Fold 1 (Real BF + Real PA)
            others = set(tr_subjs[len(tr_subjs)//4:]) # Folds 2,3,4 (Real BF + Syn PA)
            for r in real_train:
                if r["subject"] in s1_subs: train_final.append(r)
                if r["subject"] in others and r["label"] == 0: train_final.append(r)
            for s in syn_records:
                if s["subject"] in others: train_final.append(s)

        elif step_num == 5:
            # All training folds: Real BF + Synthetic PA
            for r in real_train:
                if r["label"] == 0: train_final.append(r)
            for s in syn_records:
                if s["subject"] in set(tr_subjs): train_final.append(s)

        res = train_and_eval(train_final, test_set)
        print(f"Fold {fold}: ACC={res['accuracy']:.4f}, APCER={res['apcer']:.4f}, BPCER={res['bpcer']:.4f}, ACER={res['acer']:.4f}")
        step_results.append(res)
        
    print(f"\n>>> FINAL SUMMARY {DATASET_NAME} STEP {step_num}:")
    print(f"Avg Accuracy: {np.mean([r['accuracy'] for r in step_results]):.4f}")
    print(f"Avg APCER:    {np.mean([r['apcer'] for r in step_results]):.4f}")
    print(f"Avg BPCER:    {np.mean([r['bpcer'] for r in step_results]):.4f}")
    print(f"Avg ACER:     {np.mean([r['acer'] for r in step_results]):.4f}")

if __name__ == "__main__":
    real_data = build_real_index(REAL_ROOT, DATASET_NAME)
    syn_data = build_syn_index(SYN_ROOT, DATASET_NAME)
    
    if len(real_data) == 0:
        print(f"Error: Real images not found at {REAL_ROOT}")
    elif len(syn_data) == 0:
        print(f"Error: Synthetic images not found at {SYN_ROOT}")
    else:
        print(f"Loaded {DATASET_NAME}: {len(real_data)} Real, {len(syn_data)} Synthetic")
        for step in [3, 4, 5]:
            run_experiment_step(real_data, syn_data, step)