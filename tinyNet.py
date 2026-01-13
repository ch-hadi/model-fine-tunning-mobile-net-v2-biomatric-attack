import os, re, random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models # Keep for standard models (used in placeholder)
from torch.utils.data import Dataset, DataLoader

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix

# -----------------------------
# CONFIGURATION (edit here if needed)
# -----------------------------
# NOTE: Update DATA_ROOT to your local dataset path
DATA_ROOT = r"D:\Study\image processing lab\ipl\SCUT\full\train" 
BF_FOLDER = "real"
PA_FOLDER = "spoof"
TRAIN_RATIO = 1.0 # 1.0 baseline, 0.4 step1, 0.2 step2
K_NEIGHBORS = 5
BATCH_SIZE = 32

# -----------------------------
# Subject ID extraction
# -----------------------------
def extract_subject_id(path: str) -> str:
    m = re.search(r'(\d{3})', path)
    # Uses the first 3 digits found, or the first 3 chars of the filename if no digits
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
        self.records=records; self.transform=transform
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        # Ensure image is loaded as RGB
        img=Image.open(r["path"]).convert("RGB") 
        return self.transform(img), r["label"], r["subject"]

# -----------------------------
# TinyNet Feature extractor (Placeholder - MUST BE MODIFIED)
# -----------------------------
class TinyNetFeatureExtractor(nn.Module):
    """
    NOTE: This class uses ResNet18 as a working placeholder since 
    the specific TinyNet model implementation is not provided 
    and is not standard in torchvision.
    
    You MUST replace the ResNet18 model loading with your actual 
    TinyNet implementation and set the correct output dimension (out_dim).
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        print("--- WARNING: Using ResNet18 as TinyNet ---")
        
        # --- START OF CUSTOM TINYNET/PLACEHOLDER DEFINITION ---
        # 1. Load the model (Replace with your actual TinyNet load function)
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # 2. Extract the feature backbone (e.g., everything before AvgPool and FC layer)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        
        # 3. Define the pooling layer
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # 4. Define the output dimension (IMPORTANT: Change 512 to your TinyNet's feature size)
        self.out_dim = 512 
        # --- END OF CUSTOM TINYNET/PLACEHOLDER DEFINITION ---
        
    def forward(self,x):
        f=self.backbone(x)
        f=self.pool(f)
        # Flatten the features for k-NN
        return f.view(f.size(0),-1)

# -----------------------------
# Metrics
# -----------------------------
def acer_metrics(y_true,y_pred):
    # Calculate confusion matrix: rows=true, cols=pred. Labels [0 (real), 1 (spoof)]
    cm=confusion_matrix(y_true,y_pred,labels=[0,1])
    bf_row,att_row=cm[0],cm[1] # BF (Bonafide/Real), ATT (Attack/Spoof)
    
    # BPCER (Real classified as Spoof / Total Real)
    bpcer=bf_row[1]/bf_row.sum() if bf_row.sum()>0 else 0
    # APCER (Spoof classified as Real / Total Spoof)
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
# Main experiment
# -----------------------------
def run():
    # 1. Setup
    records=build_index(DATA_ROOT,BF_FOLDER,PA_FOLDER)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the TinyNet feature extractor
    model=TinyNetFeatureExtractor(pretrained=True).to(device) 
    
    # Define standard ImageNet normalization transform
    transform=T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    # 2. GroupKFold Splitting (Subject-Independent)
    # Groups ensures that all images of a subject go into the same split (train or test)
    splits=GroupKFold(n_splits=5).split(
        np.zeros(len(records)),
        [r["label"] for r in records],
        [r["subject"] for r in records]
    )

    results=[]
    for fold,(train_idx,test_idx) in enumerate(splits,1):
        print(f"\n--- Starting Fold {fold} ---")
        
        # 3. Training Set Reduction (if TRAIN_RATIO < 1.0)
        if TRAIN_RATIO<1.0:
            rng=random.Random(42)
            train_subjects=list(set(records[i]["subject"] for i in train_idx))
            rng.shuffle(train_subjects)
            keep=int(len(train_subjects)*TRAIN_RATIO)
            keep_subs=set(train_subjects[:keep])
            train_idx=[i for i in train_idx if records[i]["subject"] in keep_subs]

        # 4. Data Loaders
        train_ds=ImgDataset([records[i] for i in train_idx],transform)
        test_ds=ImgDataset([records[i] for i in test_idx],transform)
        train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=False, num_workers=4)
        test_loader=DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False, num_workers=4)
        
        print(f"Extracting features for Training Set ({len(train_ds)} images)...")
        Xtr,ytr,_=extract_feats(model,train_loader,device)
        print(f"Extracting features for Test Set ({len(test_ds)} images)...")
        Xte,yte,_=extract_feats(model,test_loader,device)

        # 5. k-NN Classification
        knn=KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
        knn.fit(Xtr,ytr)
        ypred=knn.predict(Xte)

        # 6. Evaluation
        m=acer_metrics(yte,ypred)
        acc=(ypred==yte).mean()
        
        print(f"Fold {fold}: acc={acc:.3f}, APCER={m['APCER']:.3f}, BPCER={m['BPCER']:.3f}, ACER={m['ACER']:.3f}")
        results.append((acc,m))

    # 7. Summary
    avg_acc=np.mean([r[0] for r in results])
    avg_apcer=np.mean([r[1]['APCER'] for r in results])
    avg_bpcer=np.mean([r[1]['BPCER'] for r in results])
    avg_acer=np.mean([r[1]['ACER'] for r in results])
    
    print("\n=== Summary (5-Fold Cross-Validation) ===")
    print(f"Avg Acc={avg_acc:.3f}, Avg APCER={avg_apcer:.3f}, Avg BPCER={avg_bpcer:.3f}, Avg ACER={avg_acer:.3f}")

# -----------------------------
# Entry point
# -----------------------------
if __name__=="__main__":
    run()