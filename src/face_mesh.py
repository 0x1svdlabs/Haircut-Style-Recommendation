import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def detect_landmarks(frame):
    """
    Detect facial landmarks using MediaPipe Face Mesh.
    
    Args:
        frame (numpy.ndarray): The input image/frame.
        
    Returns:
        list of tuple: A list of (x, y) coordinates representing the facial landmarks
                       in pixel coordinates. Returns an empty list if no face is detected.
    """
    if frame is None:
        return []
        
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    landmarks = []
    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        # Process the first detected face
        for landmark in results.multi_face_landmarks[0].landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
            
    return landmarks
