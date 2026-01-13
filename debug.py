import os

# Set your dataset path
dataset_path = "SCUT/full/train"  # Change this to your actual path

# Get user IDs from real and spoof folders
real_users = set(os.listdir(os.path.join(dataset_path, "real")))
spoof_users = set(os.listdir(os.path.join(dataset_path, "spoof")))

# Find users that have BOTH real and spoof
matched_users = sorted(real_users & spoof_users)
# only_real = sorted(real_users - spoof_users)
# only_spoof = sorted(spoof_users - real_users)

# print(f"✅ Users with BOTH real and spoof: {len(matched_users)}")
# print(f"Matched users: {matched_users}\n")

# print(f"⚠️ Users with ONLY real: {len(only_real)}")
# if only_real:
#     print(f"Only real: {only_real}\n")

# print(f"⚠️ Users with ONLY spoof: {len(only_spoof)}")
# if only_spoof:
#     print(f"Only spoof: {only_spoof}")
    


# For matched users, count their images
for user_id in matched_users[:5]:  # Check first 5 users
    real_path = os.path.join(dataset_path, "real", user_id)
    spoof_path = os.path.join(dataset_path, "spoof", user_id)
    
    real_count = len([f for f in os.listdir(real_path) if f.endswith('.bmp')])
    spoof_count = len([f for f in os.listdir(spoof_path) if f.endswith('.bmp')])
    
    print(f"User {user_id}: Real={real_count}, Spoof={spoof_count}")
    
from PIL import Image
import numpy as np

# Load one image
img_path = "SCUT/full/train/real/1/1_1_1_0_5.bmp" 
print("--->>",img_path)  # Change this to an actual image path
img = Image.open(img_path)

print(f"Image size: {img.size}")  # (width, height)
print(f"Image mode: {img.mode}")  # RGB, L (grayscale), etc.
print(f"Image array shape: {np.array(img).shape}")