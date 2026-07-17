import os
import sys
import shutil
import cv2
import warnings
warnings.filterwarnings("ignore")

# Define base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add src/ folder to path
sys.path.append(os.path.join(BASE_DIR, "src"))

import face_mesh
import feature_extractor
import classifier

DATA_DIR = os.path.join(BASE_DIR, "Data")
MENTAHAN_DIR = os.path.join(DATA_DIR, "Mentahan")
CLASSES = ["oval", "round", "square", "heart", "diamond", "oblong"]

def main():
    if not os.path.exists(MENTAHAN_DIR):
        print(f"Error: {MENTAHAN_DIR} folder not found.")
        return
        
    # Create class directories in Capitalized names
    for cls in CLASSES:
        os.makedirs(os.path.join(DATA_DIR, cls.capitalize()), exist_ok=True)
        
    # Get all image files in MENTAHAN_DIR
    files = [f for f in os.listdir(MENTAHAN_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if not files:
        print("No image files found in Data/Mentahan/.")
        return
        
    print(f"Found {len(files)} new images in Data/Mentahan/. Classifying and sorting automatically...")
    
    sorted_count = 0
    failed_count = 0
    
    for file in files:
        file_path = os.path.join(MENTAHAN_DIR, file)
        img = cv2.imread(file_path)
        if img is None:
            print(f" [Error] Could not read image: {file}")
            failed_count += 1
            continue
            
        landmarks = face_mesh.detect_landmarks(img)
        if not landmarks:
            print(f" [Warning] No face landmarks detected for {file} -> Moving to Data/Unclassified/")
            unclassified_dir = os.path.join(DATA_DIR, "Unclassified")
            os.makedirs(unclassified_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(unclassified_dir, file))
            failed_count += 1
            continue
            
        features = feature_extractor.extract_features(landmarks)
        if features is None:
            print(f" [Warning] Could not extract features for {file} -> Moving to Data/Unclassified/")
            unclassified_dir = os.path.join(DATA_DIR, "Unclassified")
            os.makedirs(unclassified_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(unclassified_dir, file))
            failed_count += 1
            continue
            
        # Classify using our refined rules (will automatically trigger fallback because model has < 6 classes)
        prediction = classifier.predict(features)
        pred_label = prediction["label"] # e.g. "oval", "round"
        
        # Target folder
        dest_folder = os.path.join(DATA_DIR, pred_label.capitalize())
        dest_path = os.path.join(dest_folder, file)
        
        print(f" -> {file} classified as {pred_label.upper()} -> Moved to {pred_label.capitalize()}/")
        shutil.move(file_path, dest_path)
        sorted_count += 1
        
    print(f"\nSorting finished! Sorted: {sorted_count} images. Unclassified: {failed_count} images.")

if __name__ == "__main__":
    main()
