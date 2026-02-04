import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import KFold
import random
from tqdm import tqdm
from PIL import Image

# 1. Dataset Class
class SCUTDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths, self.labels, self.transform = image_paths, labels, transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        try: img = Image.open(self.image_paths[idx]).convert('RGB')
        except: img = Image.new('RGB', (224, 224), color=0)
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

def get_transforms(training=True):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if training:
        return transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)])

# 2. Data Loading & Mapping
def get_user_based_data(path):
    r_path, s_path = os.path.join(path, 'full', 'train', 'real'), os.path.join(path, 'full', 'train', 'spoof')
    user_data = {}
    users = sorted(set(os.listdir(r_path)) & set(os.listdir(s_path)))
    for u in users:
        rp, sp = os.path.join(r_path, u), os.path.join(s_path, u)
        user_data[u] = {
            'real': sorted([os.path.join(rp, f) for f in os.listdir(rp) if f.endswith('.bmp')]),
            'fake': sorted([os.path.join(sp, f) for f in os.listdir(sp) if f.endswith('.bmp')])
        }
    return user_data

def get_synth_map(user_data, base_dir, cat):
    all_orig = []
    for u in sorted(user_data.keys()): all_orig.extend(user_data[u][cat])
    s_dir = os.path.join(base_dir, cat, "samples")
    s_files = sorted([os.path.join(s_dir, f) for f in os.listdir(s_dir) if f.lower().endswith(('.png', '.jpg'))])
    return {o: s for o, s in zip(all_orig, s_files)}

# 3. Scenario Split Logic (Paper Specific)
def create_scenarios(user_data, vae_dir, scenario='S5'):
    r_map = get_synth_map(user_data, vae_dir, 'real')
    f_map = get_synth_map(user_data, vae_dir, 'spoof')
    u_ids = sorted(list(user_data.keys()))
    random.seed(42); random.shuffle(u_ids)
    kf = KFold(n_splits=5)
    splits = []

    for t_idx, v_idx in kf.split(u_ids):
        t_users = [u_ids[i] for i in t_idx]
        v_users = [u_ids[i] for i in v_idx]
        tr_imgs, tr_lbls = [], []
        
        for idx, u in enumerate(t_users):
            for r, f in zip(user_data[u]['real'], user_data[u]['fake']):
                if scenario == 'S5': # Total Replacement: Only Synthetic PAI
                    tr_imgs.extend([r, f_map[f]]); tr_lbls.extend([0, 1])
                elif scenario == 'S3': # 50% Augmentation: 2 folds real, 2 folds synth
                    if idx < len(t_users)//2: tr_imgs.extend([r, f]); tr_lbls.extend([0, 1])
                    else: tr_imgs.extend([r, f_map[f]]); tr_lbls.extend([0, 1])
                elif scenario == 'S4': # 25% Original: 1 fold real, 3 folds synth
                    if idx < len(t_users)//4: tr_imgs.extend([r, f]); tr_lbls.extend([0, 1])
                    else: tr_imgs.extend([r, f_map[f]]); tr_lbls.extend([0, 1])

        ts_imgs, ts_lbls = [], []
        for u in v_users:
            ts_imgs.extend(user_data[u]['real']); ts_lbls.extend([0]*len(user_data[u]['real']))
            ts_imgs.extend(user_data[u]['fake']); ts_lbls.extend([1]*len(user_data[u]['fake']))
        splits.append({'train': (tr_imgs, tr_lbls), 'test': (ts_imgs, ts_lbls)})
    return splits

# 4. Main Training Loop
def main():
    ORIG_PATH, VAE_PATH = "SCUT", r"D:\Study\image processing lab\ipl\VAE-diff\SCUT"
    ORIGINAL_DATA = "SCUT"
    VAE_DATA = r"D:\Study\image processing lab\ipl\VAE-diff\SCUT"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SCENARIO = 'S3' # S3, S4, ya S5 yahan se change karein

    user_data = get_user_based_data(ORIGINAL_DATA)
    splits = create_scenarios(user_data, VAE_DATA, SCENARIO)
    results = []

    for i, fold in enumerate(splits):
        print(f"\n--- Fold {i+1} ({SCENARIO}) ---")
        model = models.mobilenet_v2(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        model = model.to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.0001)
        crit = nn.CrossEntropyLoss()
        
        train_ldr = DataLoader(SCUTDataset(*fold['train'], get_transforms(True)), batch_size=16, shuffle=True)
        test_ldr = DataLoader(SCUTDataset(*fold['test'], get_transforms(False)), batch_size=16, shuffle=False)

        for epoch in range(5):
            model.train()
            for imgs, lbls in tqdm(train_ldr, desc=f"Ep {epoch+1}", leave=False):
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                opt.zero_grad(); crit(model(imgs), lbls).backward(); opt.step()

        model.eval(); y_p, y_t = [], []
        with torch.no_grad():
            for imgs, lbls in test_ldr:
                y_p.extend(torch.argmax(model(imgs.to(DEVICE)), 1).cpu().numpy())
                y_t.extend(lbls.numpy())
        
        y_t, y_p = np.array(y_t), np.array(y_p)
        apcer = np.sum(y_p[y_t==1]==0)/np.sum(y_t==1); bpcer = np.sum(y_p[y_t==0]==1)/np.sum(y_t==0)
        print(f"ACER: {(apcer+bpcer)/2:.4f} | APCER: {apcer:.4f} | BPCER: {bpcer:.4f}")
        results.append((apcer+bpcer)/2)

    print(f"\nAVG ACER for {SCENARIO}: {np.mean(results):.4f}")

if __name__ == "__main__": 
    main()