import os
import joblib
import numpy as np

# Path to model files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

_model = None
_scaler = None

def load_model():
    """
    Load the Random Forest model and scaler from files.
    """
    global _model, _scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            _model = joblib.load(MODEL_PATH)
            _scaler = joblib.load(SCALER_PATH)
            if len(_model.classes_) < 6:
                print(f"Warning: Loaded model only supports {len(_model.classes_)} classes ({list(_model.classes_)}). "
                      f"Using fallback rule-based classifier for full 6-shape support.")
            else:
                print(f"Model successfully loaded. Supported classes: {list(_model.classes_)}")
            return True
        except Exception as e:
            print(f"Warning: Failed to load models: {e}")
            return False
    else:
        print("Warning: Trained model files not found. Using fallback mock classifier.")
        return False

# Try loading the model on import
load_model()

def predict(features):
    """
    Predict the face shape from the extracted features.
    
    Args:
        features (dict): A dictionary containing facial measurements:
                         face_width, face_height, jaw_width, 
                         forehead_width, cheekbone_width, face_ratio.
                         
    Returns:
        dict: A dictionary containing 'label' (str) and 'confidence' (float).
    """
    global _model, _scaler
    
    if features is None:
        return {"label": "unknown", "confidence": 0.0}
        
    # Order features exactly as expected by the model
    feature_vector = np.array([[
        features["face_width"],
        features["face_height"],
        features["jaw_width"],
        features["forehead_width"],
        features["cheekbone_width"],
        features["face_ratio"]
    ]])
    
    # Check if models are loaded and have full class coverage (6 classes)
    if _model is not None and _scaler is not None and len(_model.classes_) >= 6:
        try:
            # Scale features
            scaled_features = _scaler.transform(feature_vector)
            # Predict class
            prediction = _model.predict(scaled_features)[0]
            # Predict probabilities
            probabilities = _model.predict_proba(scaled_features)[0]
            # Get class index
            class_idx = list(_model.classes_).index(prediction)
            confidence = float(probabilities[class_idx])
            
            return {
                "label": str(prediction).lower(),
                "confidence": round(confidence, 2)
            }
        except Exception as e:
            print(f"Prediction error: {e}. Falling back to mock prediction.")
            
    # Fallback/Mock prediction logic based on refined rules
    # (Allows testing the camera/UI without training the model first)
    ratio = features["face_ratio"]
    fw = features["forehead_width"]
    cw = features["cheekbone_width"]
    jw = features["jaw_width"]
    
    # Refined rule-based fallback based on ratios
    if ratio > 1.35:
        predicted_shape = "oblong"
    elif ratio < 1.23:
        # Check for round vs square
        if jw >= cw * 0.90:
            predicted_shape = "square"
        else:
            predicted_shape = "round"
    else:
        # Oblong/Round/Square ruled out. Now check heart, diamond, oval
        # Heart: forehead is wide, jaw is narrow
        if fw > jw * 1.12 and jw < cw * 0.80:
            predicted_shape = "heart"
        # Diamond: cheekbones are significantly wider than BOTH forehead and jaw
        elif cw > fw * 1.18 and cw > jw * 1.18:
            predicted_shape = "diamond"
        # Oval: fallback
        else:
            predicted_shape = "oval"
            
    return {
        "label": predicted_shape,
        "confidence": 0.85
    }
