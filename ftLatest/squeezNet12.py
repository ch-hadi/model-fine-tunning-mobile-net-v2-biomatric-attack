import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Aapki custom utilities (same as before)
from data_utils import (
    get_user_based_data, 
    create_5fold_splits, 
    SCUTDataset, 
    get_transforms
)

def calculate_metrics(y_true, y_pred, y_scores):
    """ PAD metrics: APCER, BPCER, ACER """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    real_mask = (y_true == 0)
    fake_mask = (y_true == 1)
    
    bpcer = np.sum(y_pred[real_mask] == 1) / np.sum(real_mask) if np.sum(real_mask) > 0 else 0.0
    apcer = np.sum(y_pred[fake_mask] == 0) / np.sum(fake_mask) if np.sum(fake_mask) > 0 else 0.0
    acer = (apcer + bpcer) / 2.0
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    return {
        'accuracy': accuracy,
        'apcer': apcer,
        'bpcer': bpcer,
        'acer': acer
    }

def extract_features(model, data_loader, device):
    """ SqueezeNet se high-level features extract karne ka function """
    model.eval()
    features_list = []
    labels_list = []
    
    print("Extracting features...")
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            
            # SqueezeNet ke features nikalna (classifier se pehle wala part)
            features = model.features(images)
            
            # Global Average Pooling → 512-dimensional vector (SqueezeNet)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.vstack(features_list), np.concatenate(labels_list)

def train_fold_knn(fold_idx, fold_data, device, k_neighbors=3):
    """ Ek fold par KNN train aur test - SqueezeNet backbone """
    print("\n" + "="*70)
    print(f"PROCESSING FOLD {fold_idx + 1}  |  SQUEEZENET + KNN")
    print("="*70)
    
    # Test dataset aur loader (same for all conditions)
    test_dataset = SCUTDataset(fold_data['test']['images'], fold_data['test']['labels'], 
                               transform=get_transforms(training=False))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Pre-trained SqueezeNet 1.1 load karna (better efficiency than 1.0)
    # Agar aap 1.0 chahte ho to models.squeezenet1_0(weights='DEFAULT') likh sakte ho
    model = models.squeezenet1_1(weights='DEFAULT').to(device)
    
    # Test features extract once (bahar loop se)
    X_test, y_test = extract_features(model, test_loader, device)
    
    # Train data ke real aur fake indices
    train_images = np.array(fold_data['train']['images'])
    train_labels = np.array(fold_data['train']['labels'])
    train_real_idx = np.where(train_labels == 0)[0]
    train_fake_idx = np.where(train_labels == 1)[0]
    
    print(f"Fold {fold_idx+1}: Train - {len(train_real_idx)} real, {len(train_fake_idx)} fake")
    print(f"             Test  - {np.sum(y_test == 0)} real, {np.sum(y_test == 1)} fake")
    
    # Conditions for fake sample reduction
    conditions = [
        ("baseline_full", 1.0),
        ("reduced_2_5",   2.0/5.0),
        ("reduced_1_5",   1.0/5.0),
    ]
    
    fold_results = {}
    
    for cond_name, fake_frac in conditions:
        print("\n" + "-"*50)
        print(f"Condition: {cond_name} (fake fraction: {fake_frac:.2f})")
        print("-"*50)
        
        if fake_frac >= 1.0:
            selected_train_idx = np.arange(len(train_labels))
        else:
            np.random.seed(fold_idx * 100 + hash(cond_name) % 10000)
            num_fake_select = max(1, int(len(train_fake_idx) * fake_frac))
            selected_fake = np.random.choice(
                train_fake_idx, size=num_fake_select, replace=False
            )
            selected_train_idx = np.concatenate([train_real_idx, selected_fake])
        
        reduced_train_images = train_images[selected_train_idx]
        reduced_train_labels = train_labels[selected_train_idx]
        
        print(f"Selected train: {np.sum(reduced_train_labels == 0)} real, "
              f"{np.sum(reduced_train_labels == 1)} fake")
        
        train_dataset = SCUTDataset(reduced_train_images.tolist(), 
                                    reduced_train_labels.tolist(), 
                                    transform=get_transforms(training=False))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        X_train, y_train = extract_features(model, train_loader, device)
        
        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # KNN
        print(f"Fitting KNN (K={k_neighbors})...")
        knn = KNeighborsClassifier(n_neighbors=k_neighbors, weights='distance', metric='euclidean')
        knn.fit(X_train, y_train)
        
        y_pred = knn.predict(X_test_scaled)
        y_scores = knn.predict_proba(X_test_scaled)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_scores)
        
        print(f"\n--- {cond_name} Metrics ---")
        print(f"  APCER    : {metrics['apcer']:.4f}")
        print(f"  BPCER    : {metrics['bpcer']:.4f}")
        print(f"  ACER     : {metrics['acer']:.4f}")
        print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
        
        fold_results[cond_name] = metrics
        
        # Save
        model_save_path = f"knn_squeezenet_fold_{fold_idx+1}_{cond_name}.joblib"
        joblib.dump({'knn': knn, 'scaler': scaler}, model_save_path)
        print(f"✅ Saved: {model_save_path}")
    
    return fold_results

def main():
    print("="*80)
    print("SQUEEZENET1_1 FEATURES + KNN CLASSIFIER FOR PRESENTATION ATTACK DETECTION")
    print("with full / 2/5 / 1/5 fake samples in training")
    print("="*80)
    
    DATASET_PATH = r"D:\Study\image processing lab\ipl\SCUT"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K_VAL = 3
    
    user_data = get_user_based_data(DATASET_PATH)
    fold_splits = create_5fold_splits(user_data)
    
    all_results = {cond: [] for cond in ["baseline_full", "reduced_2_5", "reduced_1_5"]}
    
    for fold_idx, fold_data in enumerate(fold_splits):
        fold_results = train_fold_knn(fold_idx, fold_data, DEVICE, k_neighbors=K_VAL)
        for cond in all_results:
            all_results[cond].append(fold_results[cond])
    
    # Final summary table
    print("\n" + "="*90)
    print("FINAL 5-FOLD RESULTS ── SqueezeNet1.1 + KNN(k=3, euclidean, distance)")
    print("="*90)
    
    conditions_display = {
        "baseline_full": "Baseline (100% fake samples)",
        "reduced_2_5"  : "Reduced (40% fake samples)",
        "reduced_1_5"  : "Reduced (20% fake samples)"
    }
    
    print(f"{'Condition':<28} | {'APCER':<10} | {'BPCER':<10} | {'ACER':<10} | {'ACC (%)':<10}")
    print("-"*85)
    
    for cond in all_results:
        vals = all_results[cond]
        m_apcer = np.mean([v['apcer'] for v in vals])
        s_apcer = np.std([v['apcer'] for v in vals])
        m_bpcer = np.mean([v['bpcer'] for v in vals])
        s_bpcer = np.std([v['bpcer'] for v in vals])
        m_acer  = np.mean([v['acer'] for v in vals])
        s_acer  = np.std([v['acer'] for v in vals])
        m_acc   = np.mean([v['accuracy'] for v in vals]) * 100
        s_acc   = np.std([v['accuracy'] for v in vals]) * 100
        
        print(f"{conditions_display[cond]:<28} | {m_apcer:.4f}±{s_apcer:.4f} | "
              f"{m_bpcer:.4f}±{s_bpcer:.4f} | {m_acer:.4f}±{s_acer:.4f} | "
              f"{m_acc:6.2f}±{s_acc:5.2f}")
    
    print("="*90)
    print("Done! Models saved with 'squeezenet' in filename.")
    print("You can compare with previous MobileNetV2 results.")

if __name__ == "__main__":
    main()