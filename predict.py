"""
Generate predictions on the hidden test set for the Chihuahua vs Muffin hackathon.
Matches the upgraded ResNet-18 architecture used in the optimized train.py.
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "best_model.pth"
TEST_DIR = "data/test"
OUTPUT_CSV = "submission.csv"
NUM_CLASSES = 2

# Hardware Optimization
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# ============================================================================
# MODEL (Must exactly match the upgraded train.py)
# ============================================================================
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(weights=None)
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(resnet_features, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Dropout(0.4),     
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Dropout(0.4),     
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features)

# ============================================================================
# TRANSFORMS
# ============================================================================
# Must match the val_transform from train.py
val_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============================================================================
# PREDICTION LOOP
# ============================================================================
def predict():
    print("=" * 60)
    print("  Chihuahua vs Muffin - Prediction (Upgraded Model)")
    print("=" * 60)
    print(f"Using device: {device}")

    # 1. Load Model
    print("\n[1/4] Loading model...")
    model = ResNet18Classifier(num_classes=NUM_CLASSES).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at {MODEL_PATH}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("  [OK] Model loaded successfully.")

    # 2. Check Test Directory
    print(f"\n[2/4] Reading test images from {TEST_DIR}...")
    if not os.path.exists(TEST_DIR):
        print(f"[ERROR] Test directory not found at {TEST_DIR}")
        return
        
    image_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        print(f"[ERROR] No images found in {TEST_DIR}")
        return
    print(f"  [OK] Found {len(image_files)} test images.")

    # 3. Run Inference
    print("\n[3/4] Running inference...")
    results = []
    
    with torch.no_grad():
        for filename in tqdm(image_files, desc="Predicting"):
            img_path = os.path.join(TEST_DIR, filename)
            try:
                image = Image.open(img_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                    
                input_tensor = val_transform(image).unsqueeze(0).to(device)
                outputs = model(input_tensor)
                
                # Calculate prediction and confidence
                softmax_outputs = F.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(softmax_outputs, 1)
                
                image_id = os.path.splitext(filename)[0]
                results.append({
                    "image_id": image_id,
                    "prediction": predicted_class.item(),
                    "confidence": f"{confidence.item():.4f}"
                })
            except Exception as e:
                print(f"\n[WARNING] Failed to process {filename}: {e}")

    # 4. Write CSV
    print(f"\n[4/4] Writing predictions to {OUTPUT_CSV}...")
    try:
        with open(OUTPUT_CSV, mode='w', newline='') as csv_file:
            fieldnames = ["image_id", "prediction", "confidence"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"  [OK] Submission saved to {OUTPUT_CSV}")
    except Exception as e:
        print(f"[ERROR] Failed to write CSV: {e}")

    print("\n[DONE] You can now upload submission.csv to Kaggle!")

if __name__ == "__main__":
    predict()