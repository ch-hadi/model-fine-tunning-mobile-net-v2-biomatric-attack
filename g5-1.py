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
DATA_ROOT = r"D:\Study\image processing lab\ipl\PLUS"
BF_FOLDER = "real"
PA_FOLDER = "spoof"
SYNTHETIC_ROOT = None   # <-- agar synthetic dataset path ho to yahan set karo, warna None rehne do
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

def build_synthetic_index(syn_root):
    index=[]
    if syn_root and os.path.exists(syn_root):
        for p in list_images(syn_root):
            index.append({"path":p,"label":1,"subject":extract_subject_id(p)})
    return index

# -----------------------------
# Dataset wrapper
# -----------------------------
class ImgDataset(Dataset):
    def __init__(self,records,transform):
        self.records=records; self.transform=transform
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        img=Image.open(r["path"]).convert("RGB")
        return self.transform(img), r["label"], r["subject"]

# -----------------------------
# MobileNetV2 Feature extractor
# -----------------------------
class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone=base.features
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.out_dim=1280
    def forward(self,x):
        f=self.backbone(x)
        f=self.pool(f)
        return f.view(f.size(0),-1)

# -----------------------------
# Metrics
# -----------------------------
def acer_metrics(y_true,y_pred):
    cm=confusion_matrix(y_true,y_pred,labels=[0,1])
    bf_row,att_row=cm[0],cm[1]
    bpcer=bf_row[1]/bf_row.sum() if bf_row.sum()>0 else 0
    apcer=att_row[0]/att_row.sum() if att_row.sum()>0 else 0
    return {"APCER":apcer,"BPCER":bpcer,"ACER":0.5*(apcer+bpcer)}

# -----------------------------
# Feature extraction
# -----------------------------
@torch.no_grad()
def extract_feats(model,loader,device):
    model.eval(); X=[]; y=[]; subs=[]
    for imgs,labels,subjects in tqdm(loader):
        imgs=imgs.to(device)
        f=model(imgs).cpu().numpy()
        X.append(f); y.append(labels.numpy()); subs+=subjects
    return np.vstack(X), np.concatenate(y), subs

# -----------------------------
# One experiment run
# -----------------------------
def run_experiment(records,train_ratio,device,model,transform,desc=""):
    splits=GroupKFold(n_splits=5).split(
        np.zeros(len(records)),
        [r["label"] for r in records],
        [r["subject"] for r in records]
    )

    print(f"\n### {desc} (train_ratio={train_ratio}) ###")
    results=[]
    for fold,(train_idx,test_idx) in enumerate(splits,1):
        if train_ratio<1.0:
            rng=random.Random(42)
            train_subjects=list(set(records[i]["subject"] for i in train_idx))
            rng.shuffle(train_subjects)
            keep=int(len(train_subjects)*train_ratio)
            keep_subs=set(train_subjects[:keep])
            train_idx=[i for i in train_idx if records[i]["subject"] in keep_subs]

        train_ds=ImgDataset([records[i] for i in train_idx],transform)
        test_ds=ImgDataset([records[i] for i in test_idx],transform)
        train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=False)
        test_loader=DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False)

        Xtr,ytr,_=extract_feats(model,train_loader,device)
        Xte,yte,_=extract_feats(model,test_loader,device)

        knn=KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
        knn.fit(Xtr,ytr)
        ypred=knn.predict(Xte)

        m=acer_metrics(yte,ypred)
        acc=(ypred==yte).mean()
        print(f"Fold {fold}: acc={acc:.3f}, APCER={m['APCER']:.3f}, BPCER={m['BPCER']:.3f}, ACER={m['ACER']:.3f}")
        results.append((acc,m))

    avg_acc=np.mean([r[0] for r in results])
    avg_apcer=np.mean([r[1]['APCER'] for r in results])
    avg_bpcer=np.mean([r[1]['BPCER'] for r in results])
    avg_acer=np.mean([r[1]['ACER'] for r in results])
    print(f"Summary: Acc={avg_acc:.3f}, APCER={avg_apcer:.3f}, BPCER={avg_bpcer:.3f}, ACER={avg_acer:.3f}")

# -----------------------------
# Entry point
# -----------------------------
if __name__=="__main__":
    records=build_index(DATA_ROOT,BF_FOLDER,PA_FOLDER)
    syn_records=build_synthetic_index(SYNTHETIC_ROOT)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=MobileNetFeatureExtractor(pretrained=True).to(device)
    transform=T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],sytd=[0.229,0.224,0.225])
    ])

    # Steps 0–2 always
    run_experiment(records,1.0,device,model,transform,"Baseline Step 0")
    run_experiment(records,0.4,device,model,transform,"Step 1 (2/5 training)")
    run_experiment(records,0.2,device,model,transform,"Step 2 (1/5 training)")

    # Steps 3–5 only if synthetic dataset provided
    if syn_records:
        combined = records + syn_records
        run_experiment(combined,0.2,device,model,transform,"Step 3 (synthetic + removed real)")
        run_experiment(combined,0.2,device,model,transform,"Step 4 (synthetic + remaining folds)")
        run_experiment(combined,1.0,device,model,transform,"Step 5 (real + synthetic only)")
