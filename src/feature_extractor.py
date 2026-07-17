import math

def euclidean_distance(p1, p2):
    """
    Calculate the 2D Euclidean distance between two points.
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def extract_features(landmarks):
    """
    Extract facial measurements from MediaPipe landmarks.
    The measurements are scaled so that the face height is normalized to 200.0,
    ensuring scale invariance (distance from camera).
    
    Args:
        landmarks (list of tuple): A list of (x, y) landmark coordinates.
        
    Returns:
        dict: A dictionary containing face_width, face_height, jaw_width, 
              forehead_width, cheekbone_width, and face_ratio.
              Returns None if landmarks are invalid.
    """
    # MediaPipe Face Mesh has 468 landmarks (or 478 with refine_landmarks)
    if not landmarks or len(landmarks) < 468:
        return None

    # Key landmark indices (standard MediaPipe Face Mesh)
    # Face Height: 10 (top of forehead/hairline) to 152 (bottom of chin)
    p_forehead_top = landmarks[10]
    p_chin_bottom = landmarks[152]
    raw_face_height = euclidean_distance(p_forehead_top, p_chin_bottom)
    
    if raw_face_height == 0:
        return None
        
    # Scale factor to normalize face height to 200.0
    scale_factor = 200.0 / raw_face_height
    
    # Cheekbone Width: 234 (left outer cheek) to 454 (right outer cheek)
    raw_cheekbone_width = euclidean_distance(landmarks[234], landmarks[454])
    cheekbone_width = raw_cheekbone_width * scale_factor
    
    # Forehead Width: 54 (left forehead edge) to 284 (right forehead edge)
    raw_forehead_width = euclidean_distance(landmarks[54], landmarks[284])
    forehead_width = raw_forehead_width * scale_factor
    
    # Jaw Width: 172 (left jaw corner) to 397 (right jaw corner)
    raw_jaw_width = euclidean_distance(landmarks[172], landmarks[397])
    jaw_width = raw_jaw_width * scale_factor
    
    # Face Width: 139 (left zygion/eye-level outer) to 368 (right zygion/eye-level outer)
    # This provides a distinct measurement from cheekbone_width
    raw_face_width = euclidean_distance(landmarks[139], landmarks[368])
    face_width = raw_face_width * scale_factor
    
    # Normalized face height (always 200.0)
    face_height = 200.0
    
    # Face ratio: face_height / face_width
    # Note: in the example output, face_ratio is ~1.42 (e.g. 200 / 140)
    face_ratio = face_height / face_width if face_width > 0 else 0
    
    return {
        "face_width": round(face_width, 2),
        "face_height": round(face_height, 2),
        "jaw_width": round(jaw_width, 2),
        "forehead_width": round(forehead_width, 2),
        "cheekbone_width": round(cheekbone_width, 2),
        "face_ratio": round(face_ratio, 2)
    }
