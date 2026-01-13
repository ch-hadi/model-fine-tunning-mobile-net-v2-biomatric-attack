import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from tqdm import tqdm
import time
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
    
    APCER (Attack Presentation Classification Error Rate):
        - Proportion of attack samples incorrectly classified as real
        - APCER = (False Acceptance) / (Total Attacks)
        - Lower is better
    
    BPCER (Bona Fide Presentation Classification Error Rate):
        - Proportion of real samples incorrectly classified as fake
        - BPCER = (False Rejection) / (Total Real)
        - Lower is better
    
    ACER (Average Classification Error Rate):
        - Average of APCER and BPCER
        - ACER = (APCER + BPCER) / 2
        - Lower is better
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Separate real (0) and fake (1) samples
    real_mask = (y_true == 0)
    fake_mask = (y_true == 1)
    
    # BPCER: Real samples classified as fake
    real_samples = y_true[real_mask]
    real_predictions = y_pred[real_mask]
    
    if len(real_samples) > 0:
        bpcer = np.sum(real_predictions == 1) / len(real_samples)
    else:
        bpcer = 0.0
    
    # APCER: Fake samples classified as real
    fake_samples = y_true[fake_mask]
    fake_predictions = y_pred[fake_mask]
    
    if len(fake_samples) > 0:
        apcer = np.sum(fake_predictions == 0) / len(fake_samples)
    else:
        apcer = 0.0
    
    # ACER: Average of both
    acer = (apcer + bpcer) / 2.0
    
    # Also calculate standard accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    return {
        'accuracy': accuracy,
        'apcer': apcer,
        'bpcer': bpcer,
        'acer': acer
    }


def create_model(num_classes=2, pretrained=True):
    """
    Create MobileNetV2 model for binary classification
    """
    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(weights='DEFAULT' if pretrained else None)
    
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final classifier
    # MobileNetV2's classifier is a Sequential with one Linear layer
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    print(f"✅ Model created with {num_classes} output classes")
    print(f"   Pre-trained: {pretrained}")
    print(f"   Final layer input features: {num_features}")
    
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate model on test set
    """
    model.eval()
    running_loss = 0.0
    
    all_labels = []
    all_predictions = []
    all_scores = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store for metrics calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_scores.extend(probabilities[:, 1].cpu().numpy())  # Probability of fake class
            
            running_loss += loss.item()
    
    test_loss = running_loss / len(test_loader)
    
    # Calculate PAD metrics
    metrics = calculate_metrics(all_labels, all_predictions, all_scores)
    
    return test_loss, metrics


def train_fold(fold_idx, fold_data, num_epochs=10, batch_size=16, learning_rate=0.001, device='cpu'):
    """
    Train model on one fold
    """
    print("\n" + "="*60)
    print(f"TRAINING FOLD {fold_idx + 1}")
    print("="*60)
    
    # Create datasets
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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = create_model(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (reduce LR if loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    # Training loop
    best_acer = float('inf')
    best_model_state = None
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_metrics = evaluate_model(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Print results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_metrics['accuracy']*100:.2f}%")
        print(f"  APCER: {test_metrics['apcer']:.4f}, BPCER: {test_metrics['bpcer']:.4f}, ACER: {test_metrics['acer']:.4f}")
        
        # Save best model based on ACER
        if test_metrics['acer'] < best_acer:
            best_acer = test_metrics['acer']
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  ✅ New best ACER: {best_acer:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n" + "="*60)
    print(f"FINAL EVALUATION - FOLD {fold_idx + 1}")
    print("="*60)
    
    final_loss, final_metrics = evaluate_model(model, test_loader, criterion, device)
    
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {final_metrics['accuracy']*100:.2f}%")
    print(f"  APCER:    {final_metrics['apcer']:.4f}")
    print(f"  BPCER:    {final_metrics['bpcer']:.4f}")
    print(f"  ACER:     {final_metrics['acer']:.4f}")
    
    return final_metrics, model


def main():
    """
    Main training function - trains all 5 folds
    """
    print("="*60)
    print("MOBILENETV2 FINE-TUNING FOR PAD")
    print("="*60)
    
    # Configuration
    DATASET_PATH = "SCUT"  # CHANGE THIS TO YOUR PATH
    NUM_EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Device: {DEVICE}")
    
    # Load and prepare data
    print("\n" + "="*60)
    print("PREPARING DATA")
    print("="*60)
    
    user_data = get_user_based_data(DATASET_PATH)
    fold_splits = create_5fold_splits(user_data)
    
    # Train all folds
    all_fold_results = []
    
    for fold_idx, fold_data in enumerate(fold_splits):
        fold_metrics, fold_model = train_fold(
            fold_idx, 
            fold_data,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            device=DEVICE
        )
        
        all_fold_results.append(fold_metrics)
        
        # Save model for this fold
        model_save_path = f"model_fold_{fold_idx+1}.pth"
        torch.save(fold_model.state_dict(), model_save_path)
        print(f"✅ Model saved: {model_save_path}")
    
    # Print overall results
    print("\n" + "="*60)
    print("OVERALL RESULTS (5-FOLD CROSS-VALIDATION)")
    print("="*60)
    
    avg_accuracy = np.mean([m['accuracy'] for m in all_fold_results])
    avg_apcer = np.mean([m['apcer'] for m in all_fold_results])
    avg_bpcer = np.mean([m['bpcer'] for m in all_fold_results])
    avg_acer = np.mean([m['acer'] for m in all_fold_results])
    
    std_acer = np.std([m['acer'] for m in all_fold_results])
    
    print(f"\nAverage Metrics (across 5 folds):")
    print(f"  Accuracy: {avg_accuracy*100:.2f}% ± {np.std([m['accuracy'] for m in all_fold_results])*100:.2f}%")
    print(f"  APCER:    {avg_apcer:.4f} ± {np.std([m['apcer'] for m in all_fold_results]):.4f}")
    print(f"  BPCER:    {avg_bpcer:.4f} ± {np.std([m['bpcer'] for m in all_fold_results]):.4f}")
    print(f"  ACER:     {avg_acer:.4f} ± {std_acer:.4f}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()