import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_CSV = os.path.join(BASE_DIR, 'dataset', 'labels.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
IMPORTANCE_PLOT_PATH = os.path.join(BASE_DIR, 'models', 'feature_importance.png')

def evaluate_model():
    """
    Load trained model, scaler, and dataset to run performance evaluations 
    and plot Random Forest feature importances.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("[Error] Model or Scaler not found. Please train the model first by running train_model.py")
        return
        
    if not os.path.exists(LABELS_CSV):
        print(f"[Error] Dataset file not found at: {LABELS_CSV}")
        return

    # 1. Load model, scaler and data
    print("Loading models and data for evaluation...")
    rf_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(LABELS_CSV)
    
    if len(df) == 0:
        print("[Error] Dataset is empty.")
        return

    feature_cols = ["face_width", "face_height", "jaw_width", "forehead_width", "cheekbone_width", "face_ratio"]
    X = df[feature_cols]
    y = df["label"]

    # 2. Scale features and make predictions
    X_scaled = scaler.transform(X)
    y_pred = rf_model.predict(X_scaled)

    # 3. Print overall metrics
    acc = accuracy_score(y, y_pred)
    print("==================================================")
    print("MODEL EVALUATION RESULTS (ON ALL DATA)")
    print("Model Machine Learning: Random Forest Classifier")
    print("==================================================")
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(y, y_pred))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y, y_pred))
    print("==================================================")

    # 4. Feature Importance analysis using Matplotlib
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Rearrange feature names in order of importance
    names = [feature_cols[i] for i in indices]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance in Face Shape Classification")
    plt.bar(range(X.shape[1]), importances[indices], align="center", color="skyblue", edgecolor="blue")
    plt.xticks(range(X.shape[1]), names, rotation=45)
    plt.ylabel("Relative Importance")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(IMPORTANCE_PLOT_PATH)
    plt.close()
    
    print(f"Feature importance chart saved to: {IMPORTANCE_PLOT_PATH}")
    print("==================================================")

if __name__ == "__main__":
    evaluate_model()
