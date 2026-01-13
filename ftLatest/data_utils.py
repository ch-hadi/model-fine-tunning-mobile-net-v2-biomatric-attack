import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
import random

class SCUTDataset(Dataset):
    """
    Custom Dataset for SCUT fingerprint vein images
    Handles grayscale images and converts to RGB for MobileNetV2
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of full paths to images
            labels: List of labels (0=real, 1=fake)
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load grayscale image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Ensure grayscale
        
        # Convert grayscale to RGB (MobileNetV2 expects 3 channels)
        image = image.convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


def get_user_based_data(dataset_path):
    """
    Organize data by users for proper cross-validation
    Returns: dict mapping user_id -> {'real': [...paths], 'fake': [...paths]}
    """
    real_path = os.path.join(dataset_path, 'full', 'train', 'real')
    spoof_path = os.path.join(dataset_path, 'full', 'train', 'spoof')
    
    user_data = {}
    
    # Get all users that have both real and spoof samples
    real_users = set([d for d in os.listdir(real_path) 
                      if os.path.isdir(os.path.join(real_path, d))])
    spoof_users = set([d for d in os.listdir(spoof_path) 
                       if os.path.isdir(os.path.join(spoof_path, d))])
    
    matched_users = sorted(real_users & spoof_users)
    
    print(f"Found {len(matched_users)} users with both real and spoof samples")
    
    for user_id in matched_users:
        # Get real samples for this user
        real_user_path = os.path.join(real_path, user_id)
        real_images = [os.path.join(real_user_path, f) 
                       for f in os.listdir(real_user_path) 
                       if f.endswith('.bmp')]
        
        # Get spoof samples for this user
        spoof_user_path = os.path.join(spoof_path, user_id)
        spoof_images = [os.path.join(spoof_user_path, f) 
                        for f in os.listdir(spoof_user_path) 
                        if f.endswith('.bmp')]
        
        user_data[user_id] = {
            'real': real_images,
            'fake': spoof_images
        }
    
    return user_data


def create_5fold_splits(user_data, seed=42):
    """
    Create 5-fold cross-validation splits based on users
    Returns: List of 5 tuples (train_data, test_data)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Get list of all user IDs
    user_ids = sorted(list(user_data.keys()))
    
    # Shuffle users
    random.shuffle(user_ids)
    
    # Create 5 folds using KFold
    kfold = KFold(n_splits=5, shuffle=False)  # Already shuffled above
    
    fold_splits = []
    
    for fold_idx, (train_indices, test_indices) in enumerate(kfold.split(user_ids)):
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train users: {len(train_indices)}")
        print(f"  Test users: {len(test_indices)}")
        
        # Get train and test users
        train_users = [user_ids[i] for i in train_indices]
        test_users = [user_ids[i] for i in test_indices]
        
        # Collect all images for train users
        train_images = []
        train_labels = []
        
        for user_id in train_users:
            # Add real samples (label = 0)
            train_images.extend(user_data[user_id]['real'])
            train_labels.extend([0] * len(user_data[user_id]['real']))
            
            # Add fake samples (label = 1)
            train_images.extend(user_data[user_id]['fake'])
            train_labels.extend([1] * len(user_data[user_id]['fake']))
        
        # Collect all images for test users
        test_images = []
        test_labels = []
        
        for user_id in test_users:
            # Add real samples (label = 0)
            test_images.extend(user_data[user_id]['real'])
            test_labels.extend([0] * len(user_data[user_id]['real']))
            
            # Add fake samples (label = 1)
            test_images.extend(user_data[user_id]['fake'])
            test_labels.extend([1] * len(user_data[user_id]['fake']))
        
        print(f"  Train images: {len(train_images)} (Real: {train_labels.count(0)}, Fake: {train_labels.count(1)})")
        print(f"  Test images: {len(test_images)} (Real: {test_labels.count(0)}, Fake: {test_labels.count(1)})")
        
        fold_splits.append({
            'train': {'images': train_images, 'labels': train_labels},
            'test': {'images': test_images, 'labels': test_labels}
        })
    
    return fold_splits


def get_transforms(training=True):
    """
    Get image transformations for training or testing
    """
    if training:
        # Training transforms with data augmentation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # MobileNetV2 expects 224x224
            transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontally
            transforms.RandomRotation(degrees=10),    # Slight rotation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                               std=[0.229, 0.224, 0.225])     # ImageNet stds
        ])
    else:
        # Test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


# Test the data loading
if __name__ == "__main__":
    print("="*60)
    print("TESTING DATA PIPELINE")
    print("="*60)
    
    # Set your dataset path
    dataset_path = "SCUT"  # Adjust this to your actual path
    
    # Get user-based data
    user_data = get_user_based_data(dataset_path)
    
    # Create 5-fold splits
    fold_splits = create_5fold_splits(user_data)
    
    # Test loading first fold
    print("\n" + "="*60)
    print("TESTING FOLD 1 DATA LOADING")
    print("="*60)
    
    train_data = fold_splits[0]['train']
    test_data = fold_splits[0]['test']
    
    # Create datasets
    train_dataset = SCUTDataset(
        train_data['images'], 
        train_data['labels'],
        transform=get_transforms(training=True)
    )
    
    test_dataset = SCUTDataset(
        test_data['images'],
        test_data['labels'],
        transform=get_transforms(training=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Test loading one batch
    images, labels = next(iter(train_loader))
    print(f"\n✅ Batch loaded successfully!")
    print(f"   Batch shape: {images.shape}")  # Should be [8, 3, 224, 224]
    print(f"   Labels shape: {labels.shape}")  # Should be [8]
    print(f"   Labels: {labels.tolist()}")
    
    print("\n" + "="*60)
    print("✅ DATA PIPELINE TEST COMPLETE!")
    print("="*60)