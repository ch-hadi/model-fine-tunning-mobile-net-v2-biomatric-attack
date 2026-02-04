import os, re, random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_ROOT = r"D:\Study\image processing lab\ipl\SCUT\full\train"
BF_FOLDER = "real"
PA_FOLDER = "spoof"
K_NEIGHBORS = 5
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# -----------------------------
# ✅ PERFECT FIX - Use folder structure!
# -----------------------------
def extract_subject_id(path: str) -> str:
    """
    ✅ PERFECT FIX - Extract subject ID from FOLDER NAME
    
    Your actual structure:
    real/1/1_1_1_0_5.bmp   → Subject folder = "1"
    spoof/1/1_1_0_0_3.bmp  → Subject folder = "1"
    
    Problem: SAME folder name "1" in both real and spoof!
    
    Solution: Combine folder_name + parent_folder
    real/1/...  → "1_real"
    spoof/1/... → "1_spoof"
    """
    # Get full path in normalized form
    norm_path = os.path.normpath(path)
    
    # Split path into parts
    # Example: "D:\SCUT\train\real\1\1_1_1_0_5.bmp"
    # Parts: ["D:", "SCUT", "train", "real", "1", "1_1_1_0_5.bmp"]
    parts = norm_path.split(os.sep)
    
    # Find the subject folder (number folder like "1", "2", etc.)
    # It's the folder BEFORE the filename
    # parts[-1] = filename
    # parts[-2] = subject folder
    # parts[-3] = real/spoof folder
    
    if len(parts) >= 3:
        subject_folder = parts[-2]  # "1", "2", "3", etc.
        parent_folder = parts[-3].lower()  # "real" or "spoof"
        
        # Combine them to make unique subject ID
        if parent_folder == 'real':
            return f"{subject_folder}_real"
        elif parent_folder == 'spoof':
            return f"{subject_folder}_spoof"
        else:
            # Fallback
            return subject_folder
    else:
        # If path structure is unexpected, use filename
        filename = os.path.basename(path)
        return filename.split('_')[0] if '_' in filename else filename[:3]

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
        full_path = os.path.join(data_root, folder)
        if os.path.exists(full_path):
            for p in list_images(full_path):
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
        img=Image.open(r["path"]).convert("RGB")
        return self.transform(img), r["label"], r["subject"]

# -----------------------------
# MobileNetV2 with Fine-tuning capability
# -----------------------------
class MobileNetFineTuner(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 2)
        )
        
    def forward(self, x, return_feats=False):
        f = self.backbone(x)
        f = self.pool(f)
        f = f.view(f.size(0), -1)
        if return_feats:
            return f
        return self.classifier(f)

# -----------------------------
# Training Function with Validation
# -----------------------------
def fine_tune_model(model, train_loader, val_loader, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    print("\n🚀 Fine-tuning started...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Check for overfitting
        gap = train_acc - val_acc
        if gap > 15:
            print(f"  ⚠️  Gap = {gap:.2f}% (Possible overfitting)")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  🛑 Early stopping at epoch {epoch+1}")
                break

# -----------------------------
# Metrics & Feature extraction
# -----------------------------
def acer_metrics(y_true,y_pred):
    cm=confusion_matrix(y_true,y_pred,labels=[0,1])
    bf_row,att_row=cm[0],cm[1]
    bpcer=bf_row[1]/bf_row.sum() if bf_row.sum()>0 else 0
    apcer=att_row[0]/att_row.sum() if att_row.sum()>0 else 0
    return {"APCER":apcer,"BPCER":bpcer,"ACER":0.5*(apcer+bpcer)}

@torch.no_grad()
def extract_feats(model,loader,device):
    model.eval()
    X=[]; y=[]; subs=[]
    for imgs,labels,subjects in loader:
        imgs=imgs.to(device)
        f=model(imgs, return_feats=True).cpu().numpy()
        X.append(f); y.append(labels.numpy()); subs+=subjects
    return np.vstack(X), np.concatenate(y), subs

# -----------------------------
# One experiment run
# -----------------------------
def run_experiment(records, train_ratio, device, transform, desc=""):
    splits = GroupKFold(n_splits=5).split(
        np.zeros(len(records)),
        [r["label"] for r in records],
        [r["subject"] for r in records]
    )

    print(f"\n{'='*70}")
    print(f"{desc} (train_ratio={train_ratio})")
    print(f"{'='*70}")
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold}/5")
        print(f"{'='*70}")
        
        # Fresh model har fold ke liye
        model = MobileNetFineTuner(pretrained=True).to(device)
        
        # Train set ko further split karo: train aur validation
        if train_ratio < 1.0:
            rng = random.Random(42)
            train_subjects = list(set(records[i]["subject"] for i in train_idx))
            rng.shuffle(train_subjects)
            keep = int(len(train_subjects) * train_ratio)
            keep_subs = set(train_subjects[:keep])
            train_idx = [i for i in train_idx if records[i]["subject"] in keep_subs]
        
        # Validation set banao (20% of training data)
        rng = random.Random(42 + fold)
        train_subjects = list(set(records[i]["subject"] for i in train_idx))
        rng.shuffle(train_subjects)
        
        val_split = int(len(train_subjects) * 0.8)
        train_subs = set(train_subjects[:val_split])
        val_subs = set(train_subjects[val_split:])
        
        actual_train_idx = [i for i in train_idx if records[i]["subject"] in train_subs]
        val_idx = [i for i in train_idx if records[i]["subject"] in val_subs]
        
        # Datasets banao
        train_ds = ImgDataset([records[i] for i in actual_train_idx], transform)
        val_ds = ImgDataset([records[i] for i in val_idx], transform)
        test_ds = ImgDataset([records[i] for i in test_idx], transform)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_ds)} samples")
        print(f"  Val:   {len(val_ds)} samples")
        print(f"  Test:  {len(test_ds)} samples")

        # STEP 1: Fine-tune with validation
        fine_tune_model(model, train_loader, val_loader, device)

        # STEP 2: Extract features from UNSEEN test data
        Xtr, ytr, _ = extract_feats(model, train_loader, device)
        Xte, yte, _ = extract_feats(model, test_loader, device)

        # STEP 3: Train KNN and evaluate
        knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
        knn.fit(Xtr, ytr)
        ypred = knn.predict(Xte)

        m = acer_metrics(yte, ypred)
        acc = (ypred == yte).mean()
        
        print(f"\n{'='*70}")
        print(f"✓ FOLD {fold} RESULTS:")
        print(f"{'='*70}")
        print(f"  Accuracy: {acc:.3f} ({acc*100:.2f}%)")
        print(f"  APCER:    {m['APCER']:.3f}")
        print(f"  BPCER:    {m['BPCER']:.3f}")
        print(f"  ACER:     {m['ACER']:.3f}")
        
        results.append((acc, m))

    if not results:
        print("\n❌ No valid folds!")
        return
    
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    avg_acc = np.mean([r[0] for r in results])
    avg_acer = np.mean([r[1]['ACER'] for r in results])
    std_acc = np.std([r[0] for r in results])
    std_acer = np.std([r[1]['ACER'] for r in results])
    
    print(f"Average Accuracy: {avg_acc:.3f} ± {std_acc:.3f}")
    print(f"Average ACER:     {avg_acer:.3f} ± {std_acer:.3f}")
    print(f"{'='*70}")

# -----------------------------
# Entry point
# -----------------------------
if __name__=="__main__":
    print("\n" + "="*70)
    print("🎯 SCUT Face Anti-Spoofing - Fine-tuning (FOLDER-BASED FIX)")
    print("="*70)
    
    records = build_index(DATA_ROOT, BF_FOLDER, PA_FOLDER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    if not records:
        print("❌ Error: No images found.")
    else:
        print(f"✅ Total images: {len(records)}")
        
        # Show path structure to verify
        print("\n" + "="*70)
        print("📁 PATH STRUCTURE VERIFICATION")
        print("="*70)
        
        print("\nReal images (first 3):")
        real_samples = [r for r in records if r["label"] == 0][:3]
        for r in real_samples:
            # Show last 3 parts of path
            parts = os.path.normpath(r['path']).split(os.sep)
            short_path = os.sep.join(parts[-3:])
            print(f"  ...{os.sep}{short_path}")
            print(f"    → Subject ID: '{r['subject']}'")
        
        print("\nSpoof images (first 3):")
        spoof_samples = [r for r in records if r["label"] == 1][:3]
        for r in spoof_samples:
            parts = os.path.normpath(r['path']).split(os.sep)
            short_path = os.sep.join(parts[-3:])
            print(f"  ...{os.sep}{short_path}")
            print(f"    → Subject ID: '{r['subject']}'")
        
        # Check unique subjects
        all_subjects = [r['subject'] for r in records]
        unique_subjects = set(all_subjects)
        real_subjects = set([r['subject'] for r in records if r['label'] == 0])
        spoof_subjects = set([r['subject'] for r in records if r['label'] == 1])
        
        print(f"\n📊 Subject Statistics:")
        print(f"  Total unique subjects: {len(unique_subjects)}")
        print(f"  Real subjects:  {len(real_subjects)}")
        print(f"  Spoof subjects: {len(spoof_subjects)}")
        
        overlap = real_subjects & spoof_subjects
        print(f"  Overlap: {len(overlap)}")
        
        if len(overlap) == 0:
            print("\n✅ NO OVERLAP! Fix is working correctly!")
        else:
            print(f"\n⚠️  WARNING: {len(overlap)} subjects still overlap!")
            print(f"  Overlapping IDs: {sorted(list(overlap))[:10]}")
        
        print("\n" + "="*70)
        
        run_experiment(records, 1.0, device, transform, "Fine-tuned Baseline")