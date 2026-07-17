import cv2

# Global camera reference
_cap = None

def get_frame(width=640, height=480):
    """
    Open the webcam if not already open, capture a frame, 
    resize it to the specified width and height, and return the frame.
    
    Args:
        width (int): The target width for the frame.
        height (int): The target height for the frame.
        
    Returns:
        tuple: (success, frame) where success is a boolean and frame is the captured image array.
    """
    global _cap
    if _cap is None:
        _cap = cv2.VideoCapture(0)
        # Set hardware capture size
        _cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
    if not _cap.isOpened():
        return False, None
        
    success, frame = _cap.read()
    if not success:
        return False, None
        
    # Only resize if the captured frame size doesn't match the requested size
    h, w = frame.shape[:2]
    if w != width or h != height:
        frame = cv2.resize(frame, (width, height))
    return True, frame

def release_camera():
    """
    Release the webcam and reset the camera reference.
    """
    global _cap
    if _cap is not None:
        _cap.release()
        _cap = None
