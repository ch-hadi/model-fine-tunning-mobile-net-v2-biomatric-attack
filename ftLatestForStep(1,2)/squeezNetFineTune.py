import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from tqdm import tqdm
import copy

from data_utils import (
    get_user_based_data, 
    create_5fold_splits, 
    SCUTDataset, 
    get_transforms
)


def calculate_metrics(y_true, y_pred, y_scores):
    """
    Calculate PAD metrics: APCER, BPCER, ACER
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    real_mask = (y_true == 0)
    fake_mask = (y_true == 1)
    
    bpcer = np.sum(y_pred[real_mask] == 1) / np.sum(real_mask) if np.sum(real_mask) > 0 else 0.0
    apcer = np.sum(y_pred[fake_mask] == 0) / np.sum(fake_mask) if np.sum(fake_mask) > 0 else 0.0
    acer = (apcer + bpcer) / 2.0
    
    accuracy = np.sum(y_true == y_pred) / len(y_true) if len(y_true) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'apcer': apcer,
        'bpcer': bpcer,
        'acer': acer
    }


def create_model(num_classes=2, pretrained=True):
    """
    Create SqueezeNet 1.1 model for binary classification
    """
    model = models.squeezenet1_1(weights='DEFAULT' if pretrained else None)
    
    # Freeze feature extractor initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final 1x1 convolution in classifier
    model.classifier[1] = nn.Conv2d(
        512, num_classes,
        kernel_size=(1, 1),
        stride=(1, 1)
    )
    
    print(f"✅ SqueezeNet1.1 model created | Classes: {num_classes} | Pretrained: {pretrained}")
    print("   Final conv: 512 → 2\n")
    
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), correct / total if total > 0 else 0.0


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())  # fake class probability
            
            running_loss += loss.item()
    
    test_loss = running_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    metrics = calculate_metrics(all_labels, all_preds, all_scores)
    
    return test_loss, metrics


def train_fold(fold_idx, fold_data, num_epochs=10, batch_size=16, learning_rate=0.001, device='cpu'):
    print("\n" + "="*80)
    print(f" FOLD {fold_idx + 1}  |  SQUEEZENET 1.1 FINE-TUNING  |  EPOCHS: {num_epochs}")
    print("="*80)
    
    train_dataset = SCUTDataset(
        fold_data['train']['images'],
        fold_data['train']['labels'],
        transform=get_transforms(training=True)
    )
    test_dataset = SCUTDataset(
        fold_data['test']['images'],
        fold_data['test']['labels'],
        transform=get_transforms(training=False)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = create_model(num_classes=2, pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)
    
    best_acer = float('inf')
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 70)
        
        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluation (har epoch ke baad full metrics)
        test_loss, metrics = evaluate_model(model, test_loader, criterion, device)
        
        scheduler.step(test_loss)
        
        # Print detailed metrics har epoch
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f}")
        print(f"  APCER : {metrics['apcer']:.4f}")
        print(f"  BPCER : {metrics['bpcer']:.4f}")
        print(f"  ACER  : {metrics['acer']:.4f}")
        print(f"  Acc   : {metrics['accuracy']*100:.2f}%")
        
        # Best model save (based on ACER)
        if metrics['acer'] < best_acer:
            best_acer = metrics['acer']
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  → Best ACER so far: {best_acer:.4f}")
    
    # Best model load kar ke final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print("\n" + "="*70)
    print(f"FINAL BEST MODEL EVALUATION - FOLD {fold_idx + 1}")
    print("="*70)
    
    _, final_metrics = evaluate_model(model, test_loader, criterion, device)
    
    print(f"  APCER : {final_metrics['apcer']:.4f}")
    print(f"  BPCER : {final_metrics['bpcer']:.4f}")
    print(f"  ACER  : {final_metrics['acer']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']*100:.2f}%")
    
    return final_metrics, model


def main():
    print("="*90)
    print("SQUEEZENET 1.1 FINE-TUNING FOR PAD - EVERY EPOCH METRICS (APCER, BPCER, ACER)")
    print("="*90)
    
    DATASET_PATH = r"D:\Study\image processing lab\ipl\SCUT"
    NUM_EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"  Path       : {DATASET_PATH}")
    print(f"  Epochs     : {NUM_EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"  Device     : {DEVICE}\n")
    
    user_data = get_user_based_data(DATASET_PATH)
    fold_splits = create_5fold_splits(user_data)
    
    all_fold_results = []
    
    for fold_idx, fold_data in enumerate(fold_splits):
        fold_metrics, fold_model = train_fold(
            fold_idx=fold_idx,
            fold_data=fold_data,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            device=DEVICE
        )
        all_fold_results.append(fold_metrics)
        
        torch.save(fold_model.state_dict(), f"squeezenet_fold_{fold_idx+1}_best.pth")
        print(f"Model saved: squeezenet_fold_{fold_idx+1}_best.pth")
    
    # Overall summary
    print("\n" + "="*90)
    print("OVERALL 5-FOLD RESULTS")
    print("="*90)
    
    metrics_names = ['accuracy', 'apcer', 'bpcer', 'acer']
    for m in metrics_names:
        values = [r[m] for r in all_fold_results]
        mean = np.mean(values)
        std = np.std(values)
        if m == 'accuracy':
            print(f"{m.upper():<10} : {mean*100:.2f}% ± {std*100:.2f}%")
        else:
            print(f"{m.upper():<10} : {mean:.4f} ± {std:.4f}")
    
    print("="*90)
    print("Done!")


if __name__ == "__main__":
    main()