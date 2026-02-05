import os, re, random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_ROOT = r"D:\Study\image processing lab\ipl\IDIAP\full\train" 
BF_FOLDER = "real"
PA_FOLDER = "spoof"
TRAIN_RATIO = 1.0
K_NEIGHBORS = 5
BATCH_SIZE = 32

# -----------------------------
# Subject ID extraction
# -----------------------------
def extract_subject_id(path: str) -> str:
    m = re.search(r'(\d{3})', path)
    return m.group(1) if m else os.path.basename(path)[:3]

# -----------------------------
# Dataset scanning
# -----------------------------
def list_images(root: str):
    exts = {'.png','.jpg','.jpeg','.bmp'}
    paths = []
    for dp,_,fs in os.walk(root):
        for f in fs:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(dp,f))
    return paths

def build_index(data_root, bf_folder="real", pa_folder="spoof"):
    index=[]
    for label,folder in [(0,bf_folder),(1,pa_folder)]:
        for p in list_images(os.path.join(data_root,folder)):
            index.append({"path":p,"label":label,"subject":extract_subject_id(p)})
    return index

# -----------------------------
# Dataset wrapper
# -----------------------------
class ImgDataset(Dataset):
    def __init__(self,records,transform):
        self.records=records
        self.transform=transform
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        img=Image.open(r["path"]).convert("RGB") 
        return self.transform(img), r["label"], r["subject"]

# -----------------------------
# SqueezeNet Feature extractor
# -----------------------------
class SqueezeNetFeatureExtractor(nn.Module):
    """
    Uses SqueezeNet1_1 and extracts features before the classifier.
    Feature dimension = 512 (from the last Fire module)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        print("--- Using SqueezeNet1_1 as backbone ---")
        
        # Load SqueezeNet
        base = models.squeezenet1_1(
            weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # Keep only the feature extraction part (up to the last Fire module)
        # SqueezeNet has 512 channels at the end of features
        self.backbone = base.features
        
        # Global average pooling → 512-dim vector
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # This is the output feature dimension for k-NN
        self.out_dim = 512  

    def forward(self, x):
        f = self.backbone(x)           # shape: (N, 512, ~13, ~13) or similar
        f = self.pool(f)               # shape: (N, 512, 1, 1)
        return f.view(f.size(0), -1)   # shape: (N, 512)

# -----------------------------
# Metrics
# -----------------------------
def acer_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    bf_row, att_row = cm[0], cm[1]
    
    bpcer = bf_row[1]/bf_row.sum() if bf_row.sum() > 0 else 0
    apcer = att_row[0]/att_row.sum() if att_row.sum() > 0 else 0
    
    return {"APCER":apcer, "BPCER":bpcer, "ACER":0.5*(apcer+bpcer)}

# -----------------------------
# Feature extraction
# -----------------------------
@torch.no_grad()
def extract_feats(model, loader, device):
    model.eval()
    X, y, subs = [], [], []
    for imgs, labels, subjects in tqdm(loader, desc="Extracting features"):
        imgs = imgs.to(device)
        f = model(imgs).cpu().numpy()
        X.append(f)
        y.append(labels.numpy())
        subs.extend(subjects)
        
    return np.vstack(X), np.concatenate(y), subs

# -----------------------------
# Main experiment
# -----------------------------
def run():
    records = build_index(DATA_ROOT, BF_FOLDER, PA_FOLDER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use SqueezeNet feature extractor
    model = SqueezeNetFeatureExtractor(pretrained=True).to(device) 
    
    # Standard ImageNet normalization (SqueezeNet was trained on ImageNet)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225])
    ])

    # GroupKFold — subject-independent protocol
    splits = GroupKFold(n_splits=5).split(
        np.zeros(len(records)),
        [r["label"] for r in records],
        [r["subject"] for r in records]
    )

    results = []
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        print(f"\n--- Fold {fold} ---")
        
        # Optional: reduce training subjects
        if TRAIN_RATIO < 1.0:
            rng = random.Random(42)
            train_subjects = list(set(records[i]["subject"] for i in train_idx))
            rng.shuffle(train_subjects)
            keep = int(len(train_subjects) * TRAIN_RATIO)
            keep_subs = set(train_subjects[:keep])
            train_idx = [i for i in train_idx if records[i]["subject"] in keep_subs]

        # Datasets & loaders
        train_ds = ImgDataset([records[i] for i in train_idx], transform)
        test_ds  = ImgDataset([records[i] for i in test_idx],  transform)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        print(f"Training set: {len(train_ds)} images")
        Xtr, ytr, _ = extract_feats(model, train_loader, device)
        
        print(f"Test set: {len(test_ds)} images")
        Xte, yte, _ = extract_feats(model, test_loader, device)

        # k-NN
        knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
        knn.fit(Xtr, ytr)
        ypred = knn.predict(Xte)

        # Evaluation
        m = acer_metrics(yte, ypred)
        acc = (ypred == yte).mean()
        
        print(f"Fold {fold}: acc={acc:.3f} | APCER={m['APCER']:.3f} | BPCER={m['BPCER']:.3f} | ACER={m['ACER']:.3f}")
        results.append((acc, m))

    # Final summary
    avg_acc   = np.mean([r[0] for r in results])
    avg_apcer = np.mean([r[1]['APCER'] for r in results])
    avg_bpcer = np.mean([r[1]['BPCER'] for r in results])
    avg_acer  = np.mean([r[1]['ACER'] for r in results])
    
    print("\n=== 5-Fold Cross-Validation Summary ===")
    print(f"Avg Accuracy : {avg_acc:.3f}")
    print(f"Avg APCER    : {avg_apcer:.3f}")
    print(f"Avg BPCER    : {avg_bpcer:.3f}")
    print(f"Avg ACER     : {avg_acer:.3f}")

# -----------------------------
if __name__ == "__main__":
    run()