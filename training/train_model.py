import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_CSV = os.path.join(BASE_DIR, 'dataset', 'labels.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

def train_model():
    """
    Train a Random Forest classifier using features from labels.csv 
    and save the model and scaler to the 'models/' folder.
    """
    if not os.path.exists(LABELS_CSV):
        print(f"[Error] Dataset file not found at: {LABELS_CSV}")
        print("Please run 'training/prepare_dataset.py' with labeled raw images first.")
        return

    # 1. Load dataset
    print(f"Loading dataset from: {LABELS_CSV}...")
    df = pd.read_csv(LABELS_CSV)
    
    if len(df) == 0:
        print("[Error] Dataset is empty.")
        return
        
    print(f"Dataset loaded. Total rows: {len(df)}")
    print("Class distribution:\n", df['label'].value_counts())

    # 2. Extract features and labels
    feature_cols = ["face_width", "face_height", "jaw_width", "forehead_width", "cheekbone_width", "face_ratio"]
    X = df[feature_cols]
    y = df["label"]

    # 3. Split dataset (80% Train, 20% Test)
    # Using stratify to ensure equal class proportions in train and test splits
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback if there are too few instances of a class to stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # 4. Standardize/Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Initialize and train Random Forest Classifier
    print("Training Random Forest model...")
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    rf_classifier.fit(X_train_scaled, y_train)

    # 6. Evaluate model performance
    y_pred = rf_classifier.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Split: {acc * 100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 7. Save model and scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf_classifier, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print("==================================================")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print("==================================================")

if __name__ == "__main__":
    train_model()
