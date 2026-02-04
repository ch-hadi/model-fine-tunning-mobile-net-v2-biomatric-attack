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
    """ Ek fold par KNN train aur test karne ka function """
    print("\n" + "="*60)
    print(f"PROCESSING FOLD {fold_idx + 1} WITH KNN")
    print("="*60)
    
    # 1. Datasets aur Loaders (KNN ke liye training=False kyunki augmentation ki zaroorat nahi)
    train_dataset = SCUTDataset(fold_data['train']['images'], fold_data['train']['labels'], 
                               transform=get_transforms(training=False))
    test_dataset = SCUTDataset(fold_data['test']['images'], fold_data['test']['labels'], 
                              transform=get_transforms(training=False))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 2. Pre-trained MobileNetV2 load karna (As a Feature Extractor)
    model = models.mobilenet_v2(weights='DEFAULT').to(device)
    # Classifier ko nikalne ki zaroorat nahi, hum features function use kar rahe hain
    
    # 3. Features nikalna
    X_train, y_train = extract_features(model, train_loader, device)
    X_test, y_test = extract_features(model, test_loader, device)
    
    # 4. Feature Scaling (KNN ke liye zaroori hai)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 5. KNN Classifier Training
    print(f"Fitting KNN (K={k_neighbors})...")
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, weights='distance', metric='euclidean')
    knn.fit(X_train, y_train)
    
    # 6. Prediction
    y_pred = knn.predict(X_test)
    y_scores = knn.predict_proba(X_test)[:, 1] # Fake class ki probability
    
    # 7. Metrics Calculate karna
    metrics = calculate_metrics(y_test, y_pred, y_scores)
    
    print(f"\nFold {fold_idx+1} Results:")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  APCER: {metrics['apcer']:.4f}, BPCER: {metrics['bpcer']:.4f}, ACER: {metrics['acer']:.4f}")
    
    return metrics, knn, scaler

def main():
    print("="*60)
    print("MOBILENETV2 FEATURES + KNN CLASSIFIER FOR PAD")
    print("="*60)
    
    # Configuration
    DATASET_PATH = r"D:\Study\image processing lab\ipl\SCUT\full\train"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K_VAL = 3 # KNN ke padosi (neighbors)
    
    # Data Load karna
    user_data = get_user_based_data(DATASET_PATH)
    fold_splits = create_5fold_splits(user_data)
    
    all_fold_results = []
    
    # 5-Fold Cross Validation loop
    for fold_idx, fold_data in enumerate(fold_splits):
        fold_metrics, fold_knn, fold_scaler = train_fold_knn(
            fold_idx, fold_data, DEVICE, k_neighbors=K_VAL
        )
        all_fold_results.append(fold_metrics)
        
        # KNN Model save karna
        model_save_path = f"knn_fold_{fold_idx+1}.joblib"
        joblib.dump({'knn': fold_knn, 'scaler': fold_scaler}, model_save_path)
        print(f"✅ Saved KNN model & Scaler: {model_save_path}")
    
    # Overall Results calculate karna
    print("\n" + "="*60)
    print("OVERALL 5-FOLD RESULTS (KNN)")
    print("="*60)
    
    for m_name in ['accuracy', 'apcer', 'bpcer', 'acer']:
        vals = [f[m_name] for f in all_fold_results]
        if m_name == 'accuracy':
            print(f"Average {m_name.capitalize()}: {np.mean(vals)*100:.2f}% ± {np.std(vals)*100:.2f}%")
        else:
            print(f"Average {m_name.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print("\n✅ PROCESS COMPLETE!")

if __name__ == "__main__":
    main()