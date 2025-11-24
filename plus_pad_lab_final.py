# plus_pad_lab_fina.py
import os
import cv2
from pathlib import Path
import numpy as np
# import random
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

# CONFIG
# ------
BASE_PATH = r"C:\Users\Xeven\Desktop\ipl\PLUS"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 5
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. LOAD IMAGES + SUBJECT ID
def load_images_from_folder(folder, label):
    data = []
    for user in os.listdir(folder):
        user_folder = os.path.join(folder, user)
        if not os.path.isdir(user_folder):
            continue
        for filename in os.listdir(user_folder):
            img_path = os.path.join(user_folder, filename)
            try:
                img = read_image(img_path)
                if img is None:
                    continue
                data.append((img_path, label, user))  # (path, label, subject)
            except:
                continue
    return data

real_data  = load_images_from_folder(os.path.join(BASE_PATH, "real"), 0)
spoof_data = load_images_from_folder(os.path.join(BASE_PATH, "spoof"), 1)
all_data   = real_data + spoof_data

SYNTHETIC_FOLDER = os.path.join(BASE_PATH, "synthetic")
# Load synthetic as spoof (label=1)
synth_data = load_images_from_folder(SYNTHETIC_FOLDER, 1)
print(f"Loaded {len(synth_data)} synthetic spoof samples")

print(f"Loaded {len(real_data)} real, {len(spoof_data)} spoof → Total: {len(all_data)}")

# 2. SUBJECT-DISJOINT 5-FOLD CV
# -----------------------------
def get_subject_disjoint_folds():
    subject_to_samples = defaultdict(list)
    for path, label, subj in all_data:
        subject_to_samples[subj].append((path, label))

    subjects = list(subject_to_samples.keys())
    print(f"Found {len(subjects)} unique subjects")

    if len(subjects) < 5:
        print("Not enough subjects for 5-fold → using 80/20 split")
        return None

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for train_idx, test_idx in kf.split(subjects):
        train_subj = [subjects[i] for i in train_idx]
        test_subj  = [subjects[i] for i in test_idx]
        train_data = [(p, l) for s in train_subj for p, l in subject_to_samples[s]]
        test_data  = [(p, l) for s in test_subj  for p, l in subject_to_samples[s]]
        folds.append((train_data, test_data))
    return folds

folds = get_subject_disjoint_folds()

# 3. RESNET-50 EXTRACTOR
# ----------------------
weights = ResNet50_Weights.IMAGENET1K_V1
backbone = resnet50(weights=weights)
extractor = torch.nn.Sequential(*list(backbone.children())[:-1])
extractor.eval().to(DEVICE)

transform = T.Compose([
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImgDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = read_image(self.paths[i]).float() / 255.0
        if img.shape[0] == 1: img = img.repeat(3, 1, 1)
        if img.shape[0] == 4: img = img[:3]
        return transform(img)

def extract_features(paths):
    if not paths: return np.zeros((0, 2048))
    ds = ImgDataset(paths)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    feats = []
    with torch.no_grad():
        for x in dl:
            f = extractor(x.to(DEVICE)).flatten(1)
            feats.append(f.cpu().numpy())
    return np.vstack(feats)

# ------------------------------------------------------------
# 4. EVALUATE
# ------------------------------------------------------------
def evaluate(train_data, test_data, ratio, step_id, synth_data):
    """
    step_id:
        0 = Baseline
        1 = Step 1
        2 = Step 2
        3 = Step 3 (25% real + 25% spoof + synthetic)
        4 = Step 4 (25% real + large synthetic)
        5 = Step 5 (pure synthetic)
    """
    # Split train into real and spoof
    train_real = [p for p, l in train_data if l == 0]
    train_spoof = [p for p, l in train_data if l == 1]

    # Base ratio (for real and spoof)
    train_real = train_real[:int(len(train_real) * ratio)]
    train_spoof = train_spoof[:int(len(train_spoof) * ratio)]

    # --- Step 3: 25% real + 25% spoof + ALL synthetic ---
    if step_id == 3:
        synth_paths = [p for p, l, _ in synth_data]  # ← 3 values
        train_spoof += synth_paths

    # --- Step 4: 25% real + LARGE synthetic ---
    elif step_id == 4:
        train_real = train_real[:int(len(train_real) * 0.25)]
        synth_paths = [p for p, l, _ in synth_data]  # ← 3 values
        train_spoof = synth_paths

    # --- Step 5: Pure synthetic ---
    elif step_id == 5:
        train_real = []
        synth_paths = [p for p, l, _ in synth_data]  # ← 3 values
        train_spoof = synth_paths

    # Final training set
    train_paths = train_real + train_spoof
    train_y = [0] * len(train_real) + [1] * len(train_spoof)

    # Test set
    test_paths = [p for p, l in test_data]
    test_y = [l for p, l in test_data]

    # Safety check
    if len(train_paths) == 0 or len(set(train_y)) < 2:
        return None

    # Extract features
    X_train = extract_features(train_paths)
    X_test = extract_features(test_paths)

    # Train k-NN
    knn = KNeighborsClassifier(n_neighbors=K, metric='cosine')
    knn.fit(X_train, train_y)
    pred = knn.predict(X_test)

    # Metrics
    tn, fp, fn, tp = confusion_matrix(test_y, pred).ravel()
    apcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acer = (apcer + bpcer) / 2

    return {
        "acer": acer,
        "apcer": apcer,
        "bpcer": bpcer,
        "train_real": len(train_real),
        "train_spoof": len(train_spoof),
        "test_real": sum(1 for l in test_y if l == 0),
        "test_spoof": sum(1 for l in test_y if l == 1)
    }
# ------------------------------------------------------------
# 5. RUN
# ------------------------------------------------------------
# steps = [("Baseline", 1.0), ("Step1", 0.5), ("Step2", 0.25)]
steps = [
    ("Baseline", 1.0, 0),
    ("Step 1",   0.5, 1),
    ("Step 2",   0.25, 2),
    ("Step 3",   0.25, 3),
    ("Step 4",   0.25, 4),
    ("Step 5",   1.0, 5),
]

results = {name: [] for name, _, _ in steps}

if folds:
    print("\n=== 5-FOLD SUBJECT-DISJOINT CV ===\n")
    for fold_idx, (train_data, test_data) in enumerate(folds, 1):
        print(f"Fold {fold_idx}")
        for name, ratio, step_id in steps:
            res = evaluate(train_data, test_data, ratio, step_id, synth_data)
            if res:
                print(f"  {name:8} ACER={res['acer']:.4f} | "
                      f"Train R={res['train_real']} S={res['train_spoof']} | "
                      f"Test R={res['test_real']} S={res['test_spoof']}")
                results[name].append(res["acer"])
else:
    print("\n=== 80/20 FALLBACK ===")
    np.random.shuffle(all_data)
    split = int(0.8 * len(all_data))
    train_data = [(p, l) for p, l, s in all_data[:split]]
    test_data  = [(p, l) for p, l, s in all_data[split:]]
    
    for name, ratio, step_id in steps:
        print(f"\n--- {name} ---")
        res = evaluate(train_data, test_data, ratio, step_id, synth_data)
        if res:
            print(f"ACER={res['acer']:.4f} | Train R={res['train_real']} S={res['train_spoof']} | "
                  f"Test R={res['test_real']} S={res['test_spoof']}")
            results[name].append(res["acer"])

# ------------------------------------------------------------
# 6. FINAL RESULTS
# ------------------------------------------------------------
print("\n" + "="*80)
print("FINAL RESULTS (mean ± std)")
print("="*80)
for name in results:
    if results[name]:
        mean_acer = np.mean(results[name])
        std_acer = np.std(results[name])
        print(f"  {name:8} ACER={res['acer']:.4f} "
            f"APCER={res['apcer']:.4f} BPCER={res['bpcer']:.4f} | "
            f"Train R={res['train_real']} S={res['train_spoof']} | "
            f"Test R={res['test_real']} S={res['test_spoof']}")
    else:
        print(f"{name:8} → No result")