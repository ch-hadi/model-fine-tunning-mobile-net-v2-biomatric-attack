"""
Standalone script to use trained MobileNetV2 model for fingerprint PAD
This script loads a saved model and predicts whether fingerprint images are real or fake
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import argparse


def create_model(num_classes=2):
    """
    Recreate the same MobileNetV2 architecture used during training
    Must match the training architecture exactly!
    """
    model = models.mobilenet_v2(weights=None)  # Don't load ImageNet weights
    
    # Replace final classifier (same as training)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model


def load_trained_model(model_path, device='cpu'):
    """
    Load a trained model from .pth file
    
    Args:
        model_path: Path to the .pth file (e.g., 'model_fold_1.pth')
        device: 'cpu' or 'cuda'
    
    Returns:
        Loaded model ready for inference
    """
    # Create model architecture
    model = create_model(num_classes=2)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set to evaluation mode (important!)
    model.eval()
    
    # Move to device
    model = model.to(device)
    
    print(f"✅ Model loaded from: {model_path}")
    print(f"   Device: {device}")
    
    return model


def get_image_transform():
    """
    Get the same image transformation used during training
    MUST be identical to training preprocessing!
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # MobileNetV2 input size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform


def predict_single_image(model, image_path, device='cpu'):
    """
    Predict whether a single image is real or fake
    
    Args:
        model: Loaded trained model
        image_path: Path to image file
        device: 'cpu' or 'cuda'
    
    Returns:
        prediction (0=Real, 1=Fake), confidence score
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('L')  # Load as grayscale
    image = image.convert('RGB')  # Convert to RGB (3 channels)
    
    transform = get_image_transform()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension [1, 3, 224, 224]
    image_tensor = image_tensor.to(device)
    
    # Make prediction (no gradient computation needed)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence


def predict_batch_images(model, image_folder, device='cpu'):
    """
    Predict multiple images from a folder
    
    Args:
        model: Loaded trained model
        image_folder: Path to folder containing images
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary with predictions for each image
    """
    # Get all image files
    valid_extensions = ['.bmp', '.png', '.jpg', '.jpeg']
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not image_files:
        print(f"⚠️ No images found in {image_folder}")
        return {}
    
    print(f"\n📊 Processing {len(image_files)} images...\n")
    
    results = {}
    real_count = 0
    fake_count = 0
    
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        
        try:
            prediction, confidence = predict_single_image(model, img_path, device)
            
            label = "REAL" if prediction == 0 else "FAKE"
            if prediction == 0:
                real_count += 1
            else:
                fake_count += 1
            
            results[img_file] = {
                'prediction': prediction,
                'label': label,
                'confidence': confidence
            }
            
            print(f"✅ {img_file[:50]:<50} → {label:>4} (Confidence: {confidence*100:.2f}%)")
            
        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")
            results[img_file] = {'error': str(e)}
    
    # Summary
    print("\n" + "="*70)
    print(f"SUMMARY: {real_count} Real, {fake_count} Fake")
    print("="*70)
    
    return results


def main():
    """
    Main function with command-line interface
    """
    parser = argparse.ArgumentParser(description='Use trained MobileNetV2 model for fingerprint PAD')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model (.pth file), e.g., model_fold_1.pth')
    parser.add_argument('--image', type=str, 
                       help='Path to single image file')
    parser.add_argument('--folder', type=str, 
                       help='Path to folder containing multiple images')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Detect device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        device = 'cpu'
    
    print("="*70)
    print("FINGERPRINT PAD - MODEL INFERENCE")
    print("="*70)
    
    # Load model
    model = load_trained_model(args.model, device)
    
    # Single image prediction
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image file not found: {args.image}")
            return
        
        print(f"\n📸 Analyzing image: {args.image}\n")
        prediction, confidence = predict_single_image(model, args.image, device)
        
        label = "REAL" if prediction == 0 else "FAKE"
        emoji = "✅" if prediction == 0 else "⚠️"
        
        print("="*70)
        print(f"{emoji} PREDICTION: {label}")
        print(f"   Confidence: {confidence*100:.2f}%")
        print("="*70)
    
    # Batch prediction
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"❌ Folder not found: {args.folder}")
            return
        
        results = predict_batch_images(model, args.folder, device)
    
    else:
        print("❌ Please provide either --image or --folder argument")
        parser.print_help()


if __name__ == "__main__":
    # If run without arguments, show interactive mode
    import sys
    
    if len(sys.argv) == 1:
        print("="*70)
        print("FINGERPRINT PAD - INTERACTIVE MODE")
        print("="*70)
        
        # Interactive mode
        model_path = input("\n📦 Enter path to model file (e.g., model_fold_1.pth): ").strip()
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            sys.exit(1)
        
        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_trained_model(model_path, device)
        
        choice = input("\n🔍 Predict (1) Single image or (2) Folder? Enter 1 or 2: ").strip()
        
        if choice == '1':
            image_path = input("📸 Enter image path: ").strip()
            if os.path.exists(image_path):
                prediction, confidence = predict_single_image(model, image_path, device)
                label = "REAL" if prediction == 0 else "FAKE"
                emoji = "✅" if prediction == 0 else "⚠️"
                
                print("\n" + "="*70)
                print(f"{emoji} PREDICTION: {label}")
                print(f"   Confidence: {confidence*100:.2f}%")
                print("="*70)
            else:
                print(f"❌ Image not found: {image_path}")
        
        elif choice == '2':
            folder_path = input("📁 Enter folder path: ").strip()
            if os.path.exists(folder_path):
                predict_batch_images(model, folder_path, device)
            else:
                print(f"❌ Folder not found: {folder_path}")
        
        else:
            print("❌ Invalid choice")
    
    else:
        # Command-line mode
        main()