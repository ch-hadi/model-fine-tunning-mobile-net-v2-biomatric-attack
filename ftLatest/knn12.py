import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from tqdm import tqdm
import joblib  # KNN model save karne ke liye
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Aapki custom utilities
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
    
    # BPCER: Real samples classified as fake
    if np.sum(real_mask) > 0:
        bpcer = np.sum(y_pred[real_mask] == 1) / np.sum(real_mask)
    else:
        bpcer = 0.0
    
    # APCER: Fake samples classified as real
    if np.sum(fake_mask) > 0:
        apcer = np.sum(y_pred[fake_mask] == 0) / np.sum(fake_mask)
    else:
        apcer = 0.0
    
    acer = (apcer + bpcer) / 2.0
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    return {
        'accuracy': accuracy,
        'apcer': apcer,
        'bpcer': bpcer,
        'acer': acer
    }

def extract_features(model, data_loader, device):
    """ MobileNetV2 se high-level features extract karne ka function """
    model.eval()
    features_list = []
    labels_list = []
    
    print("Extracting features...")
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            
            # 1. MobileNetV2 ke convolutional features nikalna
            features = model.features(images)
            
            # 2. Global Average Pooling (taki 1280 dimensions ka vector milay)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
            
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.vstack(features_list), np.concatenate(labels_list)

def train_fold_knn(fold_idx, fold_data, device, k_neighbors=3):
    """ Ek fold par KNN train aur test karne ka function for all conditions """
    print("\n" + "="*60)
    print(f"PROCESSING FOLD {fold_idx + 1} WITH KNN")
    print("="*60)
    
    # 1. Test dataset aur loader (same for all conditions)
    test_dataset = SCUTDataset(fold_data['test']['images'], fold_data['test']['labels'], 
                               transform=get_transforms(training=False))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 2. Pre-trained MobileNetV2 load karna (As a Feature Extractor)
    model = models.mobilenet_v2(weights='DEFAULT').to(device)
    
    # 3. Test features extract once
    X_test, y_test = extract_features(model, test_loader, device)
    
    # 4. Train data ke real aur fake indices
    train_images = np.array(fold_data['train']['images'])
    train_labels = np.array(fold_data['train']['labels'])
    train_real_idx = np.where(train_labels == 0)[0]
    train_fake_idx = np.where(train_labels == 1)[0]
    
    print(f"Fold {fold_idx+1}: Train - {len(train_real_idx)} real, {len(train_fake_idx)} fake")
    print(f"             Test  - {np.sum(y_test == 0)} real, {np.sum(y_test == 1)} fake")
    
    # 5. Conditions for fake sample reduction
    conditions = [
        ("baseline_full", 1.0),
        ("reduced_2_5", 2.0/5.0),
        ("reduced_1_5", 1.0/5.0),
    ]
    
    fold_results = {}
    
    for cond_name, fake_frac in conditions:
        print("\n" + "-"*40)
        print(f"Condition: {cond_name} (fake fraction: {fake_frac})")
        print("-"*40)
        
        # Select training indices
        if fake_frac >= 1.0:
            selected_train_idx = np.arange(len(train_labels))
        else:
            # Randomly select subset of fake indices
            np.random.seed(fold_idx * 100 + hash(cond_name) % 10000)  # Reproducible per fold+condition
            num_fake_select = max(1, int(len(train_fake_idx) * fake_frac))  # At least 1 if possible
            selected_fake = np.random.choice(
                train_fake_idx, 
                size=num_fake_select,
                replace=False
            )
            selected_train_idx = np.concatenate([train_real_idx, selected_fake])
        
        # Reduced train data
        reduced_train_images = train_images[selected_train_idx]
        reduced_train_labels = train_labels[selected_train_idx]
        
        print(f"Selected train: {len(np.where(reduced_train_labels == 0)[0])} real, "
              f"{len(np.where(reduced_train_labels == 1)[0])} fake")
        
        # Dataset and loader for reduced train
        train_dataset = SCUTDataset(reduced_train_images.tolist(), reduced_train_labels.tolist(), 
                                    transform=get_transforms(training=False))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Extract train features
        X_train, y_train = extract_features(model, train_loader, device)
        
        # Feature Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # KNN Classifier Training
        print(f"Fitting KNN (K={k_neighbors})...")
        knn = KNeighborsClassifier(n_neighbors=k_neighbors, weights='distance', metric='euclidean')
        knn.fit(X_train, y_train)
        
        # Prediction
        y_pred = knn.predict(X_test_scaled)
        y_scores = knn.predict_proba(X_test_scaled)[:, 1]  # Fake class ki probability
        
        # Metrics
        metrics = calculate_metrics(y_test, y_pred, y_scores)
        
        print(f"\n--- {cond_name} Metrics ---")
        print(f"  APCER   : {metrics['apcer']:.4f}")
        print(f"  BPCER   : {metrics['bpcer']:.4f}")
        print(f"  ACER    : {metrics['acer']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        
        fold_results[cond_name] = metrics
        
        # Save model
        model_save_path = f"knn_fold_{fold_idx+1}_{cond_name}.joblib"
        joblib.dump({'knn': knn, 'scaler': scaler}, model_save_path)
        print(f"✅ Saved KNN model & Scaler: {model_save_path}")
    
    return fold_results

def main():
    print("="*60)
    print("MOBILENETV2 FEATURES + KNN CLASSIFIER FOR PAD (WITH REDUCTIONS)")
    print("="*60)
    
    # Configuration
    DATASET_PATH = r"D:\Study\image processing lab\ipl\SCUT"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K_VAL = 3  # KNN ke padosi (neighbors)
    
    # Data Load karna
    user_data = get_user_based_data(DATASET_PATH)
    fold_splits = create_5fold_splits(user_data)
    
    all_results = {cond: [] for cond in ["baseline_full", "reduced_2_5", "reduced_1_5"]}
    
    # 5-Fold Cross Validation loop
    for fold_idx, fold_data in enumerate(fold_splits):
        fold_results = train_fold_knn(
            fold_idx, fold_data, DEVICE, k_neighbors=K_VAL
        )
        for cond in all_results:
            all_results[cond].append(fold_results[cond])
    
    # Overall Results calculate karna
    print("\n" + "="*60)
    print("OVERALL 5-FOLD RESULTS (KNN)")
    print("="*60)
    
    conditions_display = {
        "baseline_full": "Baseline (Full Fakes)",
        "reduced_2_5": "Reduced (2/5 Fakes)",
        "reduced_1_5": "Reduced (1/5 Fakes)"
    }
    
    print(f"{'Condition':<25} | {'APCER':<8} | {'BPCER':<8} | {'ACER':<8} | {'ACC (%)':<8}")
    print("-" * 70)
    
    for cond in all_results:
        vals = all_results[cond]
        mean_apcer = np.mean([v['apcer'] for v in vals])
        mean_bpcer = np.mean([v['bpcer'] for v in vals])
        mean_acer = np.mean([v['acer'] for v in vals])
        mean_acc = np.mean([v['accuracy'] for v in vals]) * 100
        
        std_apcer = np.std([v['apcer'] for v in vals])
        std_bpcer = np.std([v['bpcer'] for v in vals])
        std_acer = np.std([v['acer'] for v in vals])
        std_acc = np.std([v['accuracy'] for v in vals]) * 100
        
        print(f"{conditions_display[cond]:<25} | {mean_apcer:.4f} ± {std_apcer:.4f} | "
              f"{mean_bpcer:.4f} ± {std_bpcer:.4f} | {mean_acer:.4f} ± {std_acer:.4f} | "
              f"{mean_acc:.2f} ± {std_acc:.2f}")
    
    print("="*60)
    print("✅ PROCESS COMPLETE!")

if __name__ == "__main__":
    main()