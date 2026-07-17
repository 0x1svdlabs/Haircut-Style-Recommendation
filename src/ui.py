import cv2
import numpy as np

def draw_hud(frame, landmarks, face_shape, confidence, recommendations, fps=None, show_measurements=True):
    """
    Draw a premium real-time HUD (Heads-Up Display) overlay on the camera frame.
    
    Args:
        frame (numpy.ndarray): The input BGR frame.
        landmarks (list of tuple): Facial landmarks (x, y).
        face_shape (str): The predicted face shape.
        confidence (float): Confidence score (0.0 to 1.0).
        recommendations (list of str): List of recommended hairstyle names.
        fps (float, optional): Frames per second to display.
        show_measurements (bool): Whether to draw measurement lines on the face.
        
    Returns:
        numpy.ndarray: The frame with the HUD overlaid.
    """
    if frame is None:
        return frame
        
    h, w, _ = frame.shape
    
    # 1. Draw Semi-transparent Side Panel (Width: 260px)
    panel_width = 280
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, h), (15, 15, 18), -1) # Dark charcoal background
    # Apply alpha blending
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Draw panel separator line
    cv2.line(frame, (panel_width, 0), (panel_width, h), (40, 40, 45), 2)
    
    # 2. Draw Measurement Lines on Face (if landmarks exist and toggle is on)
    if landmarks and len(landmarks) >= 454 and show_measurements:
        # Define colors (B, G, R)
        color_forehead = (255, 128, 0)   # Blue-Cyan
        color_cheekbone = (0, 255, 128)  # Lime Green
        color_jaw = (255, 0, 255)        # Magenta
        color_height = (0, 165, 255)      # Orange/Gold
        
        # Draw key lines
        # Forehead (54 to 284)
        cv2.line(frame, landmarks[54], landmarks[284], color_forehead, 2)
        cv2.circle(frame, landmarks[54], 3, color_forehead, -1)
        cv2.circle(frame, landmarks[284], 3, color_forehead, -1)
        
        # Cheekbone (234 to 454)
        cv2.line(frame, landmarks[234], landmarks[454], color_cheekbone, 2)
        cv2.circle(frame, landmarks[234], 3, color_cheekbone, -1)
        cv2.circle(frame, landmarks[454], 3, color_cheekbone, -1)
        
        # Jaw (172 to 397)
        cv2.line(frame, landmarks[172], landmarks[397], color_jaw, 2)
        cv2.circle(frame, landmarks[172], 3, color_jaw, -1)
        cv2.circle(frame, landmarks[397], 3, color_jaw, -1)
        
        # Face Height (10 to 152)
        cv2.line(frame, landmarks[10], landmarks[152], color_height, 2)
        cv2.circle(frame, landmarks[10], 3, color_height, -1)
        cv2.circle(frame, landmarks[152], 3, color_height, -1)
        
        # Draw subtle dots for all other landmarks
        for i, pt in enumerate(landmarks):
            # Only draw every 5th landmark to avoid cluttering the screen
            if i % 8 == 0:
                cv2.circle(frame, pt, 1, (200, 200, 200), -1)

    # 3. Add text in the Side Panel
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # Title
    cv2.putText(frame, "HAIRSTYLE ADVISOR", (15, 30), font, 0.6, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (15, 42), (panel_width - 15, 42), (80, 80, 80), 1)
    
    # Face Shape Results
    cv2.putText(frame, "DETECTION", (15, 75), font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    
    shape_str = face_shape.capitalize() if face_shape else "Scanning..."
    conf_str = f"{confidence * 100:.1f}%" if confidence > 0 else "N/A"
    
    cv2.putText(frame, f"Shape: {shape_str}", (15, 105), font, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Conf : {conf_str}", (15, 130), font, 0.6, (140, 255, 140), 1, cv2.LINE_AA)
    
    cv2.line(frame, (15, 155), (panel_width - 15, 155), (80, 80, 80), 1)
    
    # Recommendations header
    cv2.putText(frame, "RECOMMENDED HAIRSTYLES", (15, 185), font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    
    # Draw Recommendations List
    start_y = 220
    if recommendations:
        for idx, reco in enumerate(recommendations):
            # Draw number
            cv2.putText(frame, f"{idx+1}.", (15, start_y + idx * 35), font, 0.55, (0, 215, 255), 1, cv2.LINE_AA)
            # Draw style name
            cv2.putText(frame, reco, (40, start_y + idx * 35), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No face detected", (15, start_y), font, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        
    cv2.line(frame, (15, h - 90), (panel_width - 15, h - 90), (80, 80, 80), 1)
    
    # Draw Controls / Info at bottom of panel
    cv2.putText(frame, "CONTROLS:", (15, h - 90), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, "[M] Toggle Lines", (15, h - 70), font, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, "[F] Fullscreen", (15, h - 50), font, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, "[ESC] Quit", (15, h - 30), font, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    
    # 4. Draw FPS (top-right corner of screen)
    if fps is not None:
        fps_str = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_str, (w - 100, 30), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
    return frame
