import os, re, random, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
from PIL import Image
from sklearn.model_selection import GroupKFold

# -----------------------------
# 1. CONFIGURATION (Optimized for SqueezeNet)
# -----------------------------
DATASET_NAME = "IDIAP"     
MODEL_NAME = "squeezenet"   # Ab ye SqueezeNet ke liye optimized hai
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 16
NUM_EPOCHS = 10     # SqueezeNet ke liye 10 epochs behtar hain
LEARNING_RATE = 0.0001 # LR kam kiya taake training stable ho

PATHS = {
    "SCUT":  {"real": r"D:\Study\image processing lab\ipl\SCUT\full\train", "syn": r"D:\Study\image processing lab\ipl\VAE-diff\SCUT"},
    "PLUS":  {"real": r"D:\Study\image processing lab\ipl\PLUS", "syn": r"D:\Study\image processing lab\ipl\VAE-diff\PLUS_matched"},
    "IDIAP": {"real": r"D:\Study\image processing lab\ipl\IDIAP\full\train", "syn": r"D:\Study\image processing lab\ipl\VAE-diff\IDIAP"}
}

# -----------------------------
# 2. FIXED SQUEEZENET FACTORY
# -----------------------------
def get_model(name):
    if name == "squeezenet":
        # SqueezeNet 1.1 use kar rahe hain jo 1.0 se fast aur accurate hai
        model = models.squeezenet1_1(weights='DEFAULT')
        # SqueezeNet ka classifier aik Conv2d layer hoti hai
        model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1,1))
        model.num_classes = 2
    elif name == "mobilenet":
        model = models.mobilenet_v2(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model.to(DEVICE)

# -----------------------------
# 3. DATA UTILS
# -----------------------------
def extract_id(filename, dataset_type):
    filename = filename.replace("sample", "").replace("Sample", "")
    m = re.search(r'(\d+)', filename)
    if m: return m.group(1)
    return "unknown"

def build_indices(d_type):
    real_records, syn_records = [], []
    # Real data
    for label, folder in [(0, "real"), (1, "spoof")]:
        base = os.path.join(PATHS[d_type]["real"], folder)
        if not os.path.exists(base): continue
        for dp, _, fs in os.walk(base):
            for f in fs:
                if f.lower().endswith(('.png', '.jpg', '.bmp')):
                    subj = os.path.basename(dp) if os.path.basename(dp).isdigit() else extract_id(f, d_type)
                    real_records.append({"path": os.path.join(dp, f), "label": label, "subject": subj})
    # Syn data
    syn_path = os.path.join(PATHS[d_type]["syn"], "spoof", "samples")
    if os.path.exists(syn_path):
        for f in os.listdir(syn_path):
            if f.lower().endswith(('.png', '.jpg', '.bmp')):
                syn_records.append({"path": os.path.join(syn_path, f), "label": 1, "subject": extract_id(f, d_type)})
    return real_records, syn_records

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

# -----------------------------
# 4. TRAINING & EVAL
# -----------------------------
def train_and_eval(train_records, test_records):
    transform = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train_loader = DataLoader(PADDataset(train_records, transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(PADDataset(test_records, transform), batch_size=BATCH_SIZE, shuffle=False)
    
    model = get_model(MODEL_NAME)
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
            # SqueezeNet specific flattening for outputs
            if MODEL_NAME == "squeezenet":
                outputs = torch.flatten(nn.functional.adaptive_avg_pool2d(outputs, (1, 1)), 1)
            preds = torch.max(outputs, 1)[1]
            all_y.extend(labels.numpy())
            all_p.extend(preds.cpu().numpy())
            
    y_true, y_pred = np.array(all_y), np.array(all_p)
    real_mask, fake_mask = (y_true == 0), (y_true == 1)
    acc = np.mean(y_true == y_pred)
    bpcer = np.sum(y_pred[real_mask] == 1) / np.sum(real_mask) if np.sum(real_mask) > 0 else 0
    apcer = np.sum(y_pred[fake_mask] == 0) / np.sum(fake_mask) if np.sum(fake_mask) > 0 else 0
    return {'accuracy': acc, 'apcer': apcer, 'bpcer': bpcer, 'acer': (apcer + bpcer) / 2}

# -----------------------------
# 5. MAIN LOGIC
# -----------------------------
def run_experiment(real_records, syn_records, step_num):
    print(f"\n--- {DATASET_NAME} | {MODEL_NAME.upper()} | STEP {step_num} ---")
    kf = GroupKFold(n_splits=5)
    labels = [r["label"] for r in real_records]
    subjs = [r["subject"] for r in real_records]
    
    results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(real_records, labels, subjs), 1):
        real_train = [real_records[i] for i in train_idx]
        test_set = [real_records[i] for i in test_idx]
        tr_subjs = sorted(list(set(r["subject"] for r in real_train)))
        
        train_final = []
        if step_num == 3:
            s1, s2 = set(tr_subjs[:len(tr_subjs)//4]), set(tr_subjs[len(tr_subjs)//4 : 2*len(tr_subjs)//4])
            train_final = [r for r in real_train if r["subject"] in s1]
            train_final += [r for r in real_train if (r["subject"] in s2 and r["label"] == 0)]
            train_final += [s for s in syn_records if s["subject"] in s2]
        elif step_num == 4:
            s1, others = set(tr_subjs[:len(tr_subjs)//4]), set(tr_subjs[len(tr_subjs)//4:])
            train_final = [r for r in real_train if r["subject"] in s1]
            train_final += [r for r in real_train if (r["subject"] in others and r["label"] == 0)]
            train_final += [s for s in syn_records if s["subject"] in others]
        elif step_num == 5:
            train_final = [r for r in real_train if r["label"] == 0]
            train_final += [s for s in syn_records if s["subject"] in set(tr_subjs)]

        res = train_and_eval(train_final, test_set)
        print(f"Fold {fold}: ACC={res['accuracy']:.4f}, APCER={res['apcer']:.4f}, BPCER={res['bpcer']:.4f}, ACER={res['acer']:.4f}")
        results.append(res)
    print(f">> AVG ACER: {np.mean([r['acer'] for r in results]):.4f}")

if __name__ == "__main__":
    r_data, s_data = build_indices(DATASET_NAME)
    for s in [3, 4, 5]:
        run_experiment(r_data, s_data, s)