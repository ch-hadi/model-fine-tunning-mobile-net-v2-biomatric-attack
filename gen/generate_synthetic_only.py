# generate_synthetic_only.py
import os
import cv2
import random
from pathlib import Path
import numpy as np

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BASE_PATH = r"C:\Users\Xeven\Desktop\ipl\PLUS"
REAL_FOLDER = os.path.join(BASE_PATH, "real")
SYNTH_FOLDER = os.path.join(BASE_PATH, "synthetic")

os.makedirs(SYNTH_FOLDER, exist_ok=True)
print(f"Synthetic folder: {SYNTH_FOLDER}")

# ------------------------------------------------------------
# FUNCTION: Create synthetic spoof
# ------------------------------------------------------------
def create_synthetic(real_path):
    try:
        img = cv2.imread(real_path)
        if img is None:
            return False

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Simulate print-attack
        blur_k = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
        noise = np.random.normal(0, random.uniform(10, 25), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        alpha = random.uniform(0.6, 0.9)
        img = np.clip((img * alpha), 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            paper = np.random.normal(240, 10, img.shape).astype(np.uint8)
            img = cv2.addWeighted(img, 0.7, paper, 0.3, 0)

        # --- FIX: Extract user ID and create subfolder ---
        user_id = Path(real_path).parent.name  # e.g., "001"
        synth_user_folder = os.path.join(SYNTH_FOLDER, user_id)
        os.makedirs(synth_user_folder, exist_ok=True)

        filename = Path(real_path).name.replace(".png", "_syn.png").replace(".jpg", "_syn.jpg")
        output_path = os.path.join(synth_user_folder, filename)
        # --- END FIX ---

        success = cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return success

    except Exception as e:
        print(f"Error: {e}")
        return False


# MAIN
# --------------------
count = 0
success_count = 0
print("Starting synthetic generation...")
for user in os.listdir(REAL_FOLDER):
    user_path = os.path.join(REAL_FOLDER, user)
    if not os.path.isdir(user_path): continue
    for file in os.listdir(user_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            real_path = os.path.join(user_path, file)
            if create_synthetic(real_path):
                success_count += 1
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} | Saved {success_count}")
print(f"\nDONE! Saved {success_count} synthetic images in subfolders of {SYNTH_FOLDER}")