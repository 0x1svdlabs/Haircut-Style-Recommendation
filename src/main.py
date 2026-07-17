import cv2
import time
import warnings
warnings.filterwarnings("ignore")

import camera
import face_mesh
import feature_extractor
import classifier
import recommender
import ui

def main():
    print("==================================================")
    print("Real-Time Hairstyle Recommendation System")
    print("Starting webcam... Press [ESC] in the window to quit.")
    print("==================================================")
    
    # Toggle to show measurement lines
    show_measurements = True
    
    # FPS tracking variables
    prev_time = 0
    fps = 0.0
    
    window_name = "Hairstyle Recommendation System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    try:
        while True:
            # 1. Capture frame (HD 1280x720 for fullscreen display)
            success, frame = camera.get_frame(width=1280, height=720)
            if not success or frame is None:
                print("Error: Could not retrieve frame from webcam.")
                # Wait a bit and retry
                time.sleep(0.1)
                continue
                
            # Mirror the frame for a more natural webcam mirror display
            frame = cv2.flip(frame, 1)
            
            # 2. Detect face landmarks
            landmarks = face_mesh.detect_landmarks(frame)
            
            face_shape = ""
            confidence = 0.0
            recommendations = []
            
            if landmarks:
                # 3. Extract geometric facial measurements
                features = feature_extractor.extract_features(landmarks)
                
                if features:
                    # 4. Classify face shape using classifier
                    prediction = classifier.predict(features)
                    face_shape = prediction["label"]
                    confidence = prediction["confidence"]
                    
                    # 5. Get recommended hairstyles based on face shape
                    recommendations = recommender.recommend(face_shape)
            
            # Calculate actual FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            if time_diff > 0:
                fps = 1.0 / time_diff
            prev_time = curr_time
            
            # 6. Draw results in real time
            frame = ui.draw_hud(
                frame, 
                landmarks, 
                face_shape, 
                confidence, 
                recommendations, 
                fps=fps, 
                show_measurements=show_measurements
            )
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Listen to key presses
            key = cv2.waitKey(1) & 0xFF
            
            # ESC key to exit
            if key == 27:
                break
            # 'M' or 'm' to toggle measurements
            elif key == ord('m') or key == ord('M'):
                show_measurements = not show_measurements
                print(f"Measurement lines visibility toggled to: {show_measurements}")
            # 'F' or 'f' to toggle fullscreen
            elif key == ord('f') or key == ord('F'):
                is_fullscreen = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                
    finally:
        # Clean up and release camera
        camera.release_camera()
        cv2.destroyAllWindows()
        print("Camera released and windows closed. Goodbye!")

if __name__ == "__main__":
    main()
