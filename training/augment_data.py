import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Define base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")

TARGETS = {
    "Square": 12,
    "Heart": 13,
    "Diamond": 12
}

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return result

def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:,:,2] = hsv[:,:,2] * factor
    hsv[:,:,2][hsv[:,:,2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def main():
    print("=== STARTING DATA AUGMENTATION ===")
    for folder_name, target_count in TARGETS.items():
        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            
        # Get existing image files
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        current_count = len(files)
        
        print(f"\nFolder '{folder_name}' currently has {current_count} files. Target is {target_count}.")
        
        if current_count >= target_count:
            print(f" -> Target already met for '{folder_name}'. No augmentation needed.")
            continue
            
        needed = target_count - current_count
        print(f" -> Generating {needed} augmented images...")
        
        # We will loop through the original files and apply transformations to generate new ones
        original_files = list(files)
        if not original_files:
            print(f" [Warning] No base files in '{folder_name}' to perform augmentation on.")
            continue
            
        aug_idx = 0
        while len(files) < target_count:
            for file in original_files:
                if len(files) >= target_count:
                    break
                    
                file_path = os.path.join(folder_path, file)
                img = cv2.imread(file_path)
                if img is None:
                    continue
                    
                aug_idx += 1
                new_img = None
                
                # Cycle through different augmentation techniques
                tech = aug_idx % 4
                if tech == 1:
                    new_img = cv2.flip(img, 1)
                    desc = "flip"
                elif tech == 2:
                    new_img = adjust_brightness(img, 1.15)
                    desc = "bright"
                elif tech == 3:
                    new_img = adjust_brightness(img, 0.85)
                    desc = "dark"
                else:
                    angle = 4 if (aug_idx % 2 == 0) else -4
                    new_img = rotate_image(img, angle)
                    desc = f"rot{angle}"
                    
                new_filename = f"aug_{desc}_{aug_idx}_{file}"
                new_file_path = os.path.join(folder_path, new_filename)
                
                cv2.imwrite(new_file_path, new_img)
                files.append(new_filename)
                print(f"   + Created: {new_filename}")
                
        print(f" -> Finished '{folder_name}'. Total files now: {len(files)}")
    print("\n=== DATA AUGMENTATION COMPLETED ===")

if __name__ == "__main__":
    main()
