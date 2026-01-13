"""
kalaad.py
Biometric Presentation Attack Detection (PAD) System
Using Deep Learning Features and Transfer Learning with k-NN Classifier
"""

# import os
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import cv2
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
# from tensorflow.keras.preprocessing import image
import warnings
warnings.filterwarnings('ignore')

class TransferLearningFeatureExtractor:
    """Extract features using pre-trained CNN models (Transfer Learning)"""
    
    def __init__(self, model_name='MobileNetV2', target_size=(224, 224)):
        """
        Initialize feature extractor with pre-trained model
        
        Args:
            model_name: 'VGG16', 'ResNet50', or 'MobileNetV2'
            target_size: Input image size for the model
        """
        self.model_name = model_name
        self.target_size = target_size
        
        # Load pre-trained model without top layers (for feature extraction)
        if model_name == 'VGG16':
            self.model = VGG16(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess = vgg_preprocess
        elif model_name == 'ResNet50':
            self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess = resnet_preprocess
        elif model_name == 'MobileNetV2':
            self.model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess = mobilenet_preprocess
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"Loaded {model_name} for feature extraction")
        print(f"Feature dimension: {self.model.output_shape[1]}")
    
    def extract_features(self, img_path):
        """Extract deep learning features from an image"""
        try:
            # Load and preprocess image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                return None
            
            # Resize to target size
            img = cv2.resize(img, self.target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Expand dimensions and preprocess
            img = np.expand_dims(img, axis=0)
            img = self.preprocess(img)
            
            # Extract features
            features = self.model.predict(img, verbose=0)
            return features.flatten()
        
        except Exception as e:
            print(f"Error extracting features from {img_path}: {e}")
            return None


class DatasetLoader:
    """Load and organize biometric datasets"""
    
    @staticmethod
    def load_plus_dataset(base_path):
        """Load PLUS dataset with user-based organization"""
        base_path = Path(base_path)
        data = {'real': {}, 'spoof': {}}
        
        # Load real samples (organized by user folders)
        real_path = base_path / 'real'
        if real_path.exists():
            for user_folder in sorted(real_path.iterdir()):
                if user_folder.is_dir():
                    user_id = user_folder.name
                    data['real'][user_id] = sorted(user_folder.glob('*.png'))
        
        # Load spoof samples (organized by user folders)
        spoof_path = base_path / 'spoof'
        if spoof_path.exists():
            for user_folder in sorted(spoof_path.iterdir()):
                if user_folder.is_dir():
                    user_id = user_folder.name
                    data['spoof'][user_id] = sorted(user_folder.glob('*.png'))
        
        return data
    
    @staticmethod
    def load_idiap_scut_dataset(base_path, subset='cropped'):
        """Load IDIAP or SCUT dataset"""
        base_path = Path(base_path) / subset
        data = {'real': {}, 'spoof': {}}
        
        # Combine train and dev sets
        for split in ['train', 'dev']:
            split_path = base_path / split
            if not split_path.exists():
                continue
            
            for class_type in ['real', 'spoof']:
                class_path = split_path / class_type
                if class_path.exists():
                    for user_folder in sorted(class_path.iterdir()):
                        if user_folder.is_dir():
                            user_id = user_folder.name.split('-')[0]  # Extract ID
                            if user_id not in data[class_type]:
                                data[class_type][user_id] = []
                            data[class_type][user_id].extend(sorted(user_folder.glob('*.png')))
        
        return data


class PADEvaluator:
    """Evaluate PAD system with standard metrics"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculate PAD metrics: APCER, BPCER, ACER
        
        APCER: Attack Presentation Classification Error Rate
        BPCER: Bona fide Presentation Classification Error Rate  
        ACER: Average Classification Error Rate
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # APCER: Proportion of attack samples incorrectly classified as bona fide
        apcer = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # BPCER: Proportion of bona fide samples incorrectly classified as attack
        bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # ACER: Average of APCER and BPCER
        acer = (apcer + bpcer) / 2
        
        # Overall accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        return {
            'APCER': apcer * 100,
            'BPCER': bpcer * 100,
            'ACER': acer * 100,
            'Accuracy': accuracy * 100
        }


class PADSystem:
    """Complete PAD system with k-NN classifier and transfer learning"""
    
    def __init__(self, dataset_path, dataset_name='IDIAP', model_name='MobileNetV2', n_neighbors=3):
        """
        Initialize PAD system
        
        Args:
            dataset_path: Path to dataset
            dataset_name: 'PLUS', 'IDIAP', or 'SCUT'
            model_name: Pre-trained model for feature extraction
            n_neighbors: Number of neighbors for k-NN
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self.n_neighbors = n_neighbors
        
        # Initialize feature extractor
        self.feature_extractor = TransferLearningFeatureExtractor(model_name=model_name)
        
        # Initialize k-NN classifier
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        # Load dataset
        print(f"\nLoading {dataset_name} dataset...")
        if dataset_name == 'IDIAP':
            self.data = DatasetLoader.load_plus_dataset(dataset_path)
        else:
            subset = 'cropped' if dataset_name == 'IDIAP' else 'roi'
            self.data = DatasetLoader.load_idiap_scut_dataset(dataset_path, subset)
        
        # Get balanced user list (users with both real and spoof samples)
        self.balanced_users = sorted(
            set(self.data['real'].keys()) & set(self.data['spoof'].keys())
        )
        print(f"Found {len(self.balanced_users)} balanced users")
    
    def extract_features_from_paths(self, image_paths):
        """Extract features from list of image paths"""
        features = []
        for img_path in image_paths:
            feat = self.feature_extractor.extract_features(img_path)
            if feat is not None:
                features.append(feat)
        return np.array(features)
    
    def prepare_data_for_users(self, users):
        """Prepare features and labels for given users"""
        X, y = [], []
        
        for user_id in users:
            # Real samples (label=1)
            if user_id in self.data['real']:
                real_features = self.extract_features_from_paths(self.data['real'][user_id])
                X.append(real_features)
                y.extend([1] * len(real_features))
            
            # Spoof samples (label=0)
            if user_id in self.data['spoof']:
                spoof_features = self.extract_features_from_paths(self.data['spoof'][user_id])
                X.append(spoof_features)
                y.extend([0] * len(spoof_features))
        
        return np.vstack(X), np.array(y)
    
    def run_baseline_evaluation(self, n_folds=5):
        """
        Baseline evaluation: 5-fold cross-validation with full data
        Training on 4/5, testing on 1/5
        """
        print(f"\n{'='*60}")
        print(f"BASELINE EVALUATION - {self.dataset_name}")
        print(f"Using {self.feature_extractor.model_name} features + {self.n_neighbors}-NN")
        print(f"{'='*60}")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        users = np.array(self.balanced_users)
        
        results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(users), 1):
            print(f"\nFold {fold_idx}/{n_folds}")
            train_users = users[train_idx]
            test_users = users[test_idx]
            
            print(f"  Train users: {len(train_users)}, Test users: {len(test_users)}")
            
            # Prepare training data
            print("  Extracting training features...")
            X_train, y_train = self.prepare_data_for_users(train_users)
            print(f"  Training samples: {len(X_train)} (Real: {sum(y_train)}, Spoof: {len(y_train)-sum(y_train)})")
            
            # Prepare test data
            print("  Extracting test features...")
            X_test, y_test = self.prepare_data_for_users(test_users)
            print(f"  Test samples: {len(X_test)} (Real: {sum(y_test)}, Spoof: {len(y_test)-sum(y_test)})")
            
            # Train k-NN classifier
            self.classifier.fit(X_train, y_train)
            
            # Predict
            y_pred = self.classifier.predict(X_test)
            
            # Calculate metrics
            metrics = PADEvaluator.calculate_metrics(y_test, y_pred)
            results.append(metrics)
            
            print(f"  Results: ACER={metrics['ACER']:.2f}%, "
                  f"APCER={metrics['APCER']:.2f}%, "
                  f"BPCER={metrics['BPCER']:.2f}%, "
                  f"Acc={metrics['Accuracy']:.2f}%")
        
        # Average results
        avg_results = {
            metric: np.mean([r[metric] for r in results])
            for metric in results[0].keys()
        }
        
        print(f"\n{'='*60}")
        print("BASELINE AVERAGE RESULTS:")
        print(f"  ACER:     {avg_results['ACER']:.2f}%")
        print(f"  APCER:    {avg_results['APCER']:.2f}%")
        print(f"  BPCER:    {avg_results['BPCER']:.2f}%")
        print(f"  Accuracy: {avg_results['Accuracy']:.2f}%")
        print(f"{'='*60}\n")
        
        return avg_results
    
    def run_reduced_training_evaluation(self, training_fraction=0.5, n_folds=5):
        """
        Evaluate with reduced training data
        
        Args:
            training_fraction: Fraction of training users to use (0.4 for 2/5, 0.2 for 1/5)
        """
        print(f"\n{'='*60}")
        print(f"REDUCED TRAINING EVALUATION - {self.dataset_name}")
        print(f"Training fraction: {training_fraction} of training users")
        print(f"{'='*60}")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        users = np.array(self.balanced_users)
        
        results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(users), 1):
            print(f"\nFold {fold_idx}/{n_folds}")
            all_train_users = users[train_idx]
            test_users = users[test_idx]
            
            # Randomly select subset of training users
            np.random.seed(42 + fold_idx)
            n_train_subset = int(len(all_train_users) * training_fraction)
            train_users = np.random.choice(all_train_users, size=n_train_subset, replace=False)
            
            print(f"  Train users: {len(train_users)}/{len(all_train_users)}, Test users: {len(test_users)}")
            
            # Prepare data
            print("  Extracting features...")
            X_train, y_train = self.prepare_data_for_users(train_users)
            X_test, y_test = self.prepare_data_for_users(test_users)
            
            print(f"  Training samples: {len(X_train)}")
            print(f"  Test samples: {len(X_test)}")
            
            # Train and predict
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            
            # Calculate metrics
            metrics = PADEvaluator.calculate_metrics(y_test, y_pred)
            results.append(metrics)
            
            print(f"  Results: ACER={metrics['ACER']:.2f}%, Acc={metrics['Accuracy']:.2f}%")
        
        # Average results
        avg_results = {
            metric: np.mean([r[metric] for r in results])
            for metric in results[0].keys()
        }
        
        print(f"\n{'='*60}")
        print(f"AVERAGE RESULTS (Training fraction: {training_fraction}):")
        print(f"  ACER:     {avg_results['ACER']:.2f}%")
        print(f"  APCER:    {avg_results['APCER']:.2f}%")
        print(f"  BPCER:    {avg_results['BPCER']:.2f}%")
        print(f"  Accuracy: {avg_results['Accuracy']:.2f}%")
        print(f"{'='*60}\n")
        
        return avg_results


# Example usage
if __name__ == "__main__":
    # Configuration
    DATASET_PATH = r"D:\Study\image processing\IDIAP\full\train"
    DATASET_NAME = "IDIAP"  # Options: 'PLUS', 'IDIAP', 'SCUT'
    MODEL_NAME = "MobileNetV2"   # Options: 'VGG16', 'ResNet50', 'MobileNetV2'
    N_NEIGHBORS = 3
    
    # Initialize PAD system
    pad_system = PADSystem(
        dataset_path=DATASET_PATH,
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        n_neighbors=N_NEIGHBORS
    )
    
    # Run evaluations
    print("\n" + "="*60)
    print("STARTING PAD EVALUATION WITH TRANSFER LEARNING")
    print("="*60)
    
    # Baseline: Full training data (4/5)
    baseline_results = pad_system.run_baseline_evaluation(n_folds=5)
    
    # Step 1: Reduced training data (2/5)
    step1_results = pad_system.run_reduced_training_evaluation(
        training_fraction=0.5, n_folds=5
    )
    
    # Step 2: Further reduced training data (1/5)
    step2_results = pad_system.run_reduced_training_evaluation(
        training_fraction=0.25, n_folds=5
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL RESULTS")
    print("="*60)
    print(f"Baseline (4/5 training):  ACER={baseline_results['ACER']:.2f}%")
    print(f"Step 1 (2/5 training):    ACER={step1_results['ACER']:.2f}%")
    print(f"Step 2 (1/5 training):    ACER={step2_results['ACER']:.2f}%")
    print("="*60)