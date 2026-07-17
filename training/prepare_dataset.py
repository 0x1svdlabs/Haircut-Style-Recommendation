import os
import sys
import cv2
import pandas as pd

# Add src/ folder to path to import components
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import face_mesh
import feature_extractor

# Define path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Use 'Data' folder if it exists, otherwise fall back to 'dataset/raw'
if os.path.exists(os.path.join(BASE_DIR, 'Data')):
    RAW_DIR = os.path.join(BASE_DIR, 'Data')
else:
    RAW_DIR = os.path.join(BASE_DIR, 'dataset', 'raw')
    
PROCESSED_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')
LABELS_CSV = os.path.join(BASE_DIR, 'dataset', 'labels.csv')

# Expected face shapes (corresponds to subfolders)
CLASSES = ["oval", "round", "square", "heart", "diamond", "oblong"]

def initialize_directories():
    """
    Ensure the dataset directory structure is created.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Only initialize expected folders in RAW_DIR if it's empty
    if not os.listdir(RAW_DIR):
        for cls in CLASSES:
            os.makedirs(os.path.join(RAW_DIR, cls), exist_ok=True)
        
    print(f"Dataset source folder: {RAW_DIR}")

def prepare_dataset():
    """
    Scan raw data directory, extract features from all found images, 
    and save them into 'dataset/labels.csv'.
    """
    initialize_directories()
    
    data_rows = []
    
    print("\nScanning raw images for feature extraction...")
    
    # Check if there are any images
    total_images_processed = 0
    total_images_failed = 0
    
    for label in CLASSES:
        # Find folder matching the class name case-insensitively
        class_folder = None
        if os.path.exists(RAW_DIR):
            for entry in os.listdir(RAW_DIR):
                if entry.lower() == label.lower() and os.path.isdir(os.path.join(RAW_DIR, entry)):
                    class_folder = os.path.join(RAW_DIR, entry)
                    break
                    
        if class_folder is None:
            continue
            
        files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if not files:
            print(f" - No images found in folder: {os.path.basename(class_folder)}")
            continue
            
        print(f" - Processing {len(files)} images for class '{label}'...")
        
        for file in files:
            img_path = os.path.join(class_folder, file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"   [Warning] Could not read image: {file}")
                total_images_failed += 1
                continue
                
            # Detect landmarks
            landmarks = face_mesh.detect_landmarks(img)
            
            if not landmarks:
                print(f"   [Warning] Face landmarks not detected in: {file}")
                total_images_failed += 1
                continue
                
            # Extract features
            features = feature_extractor.extract_features(landmarks)
            
            if features is None:
                print(f"   [Warning] Feature extraction failed for: {file}")
                total_images_failed += 1
                continue
                
            # Append label to features row
            row = features.copy()
            row["label"] = label
            data_rows.append(row)
            
            total_images_processed += 1
            
    if not data_rows:
        print("\n[Error] No training features extracted! Make sure you place labeled face images under 'dataset/raw/'")
        return
        
    # Convert to pandas DataFrame and save to CSV
    df = pd.DataFrame(data_rows)
    df.to_csv(LABELS_CSV, index=False)
    
    print("\nFeature extraction completed successfully!")
    print(f"Successfully processed: {total_images_processed} images.")
    if total_images_failed > 0:
        print(f"Failed to process: {total_images_failed} images.")
    print(f"Dataset saved to: {LABELS_CSV}")

if __name__ == "__main__":
    prepare_dataset()
