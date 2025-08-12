# ===============================
# Fall / Near-Fall / No-Fall Detection
# Preprocessing + Training + Evaluation
# Optimized + Scaler Saving + Hyperparameter Tuning
# ===============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------------
# CONFIGURATION
# -------------------------------
CSV_FILE = "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Complete code/Random Python/Fall detector/Resources/Train_Augumented.csv"   # Your augmented dataset file
MOVING_AVG_WINDOW = 5               # Optimum window size (tunable)
TEST_SIZE = 0.2                      # 80% training, 20% testing
RANDOM_STATE = 42                    # Reproducibility

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv(CSV_FILE)

# Drop index column if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# -------------------------------
# 2. SMOOTH NUMERIC SENSOR COLUMNS
# -------------------------------
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

for col in numeric_cols:
    df[col] = df[col].rolling(window=MOVING_AVG_WINDOW, center=True).mean()

for col in numeric_cols:
    df[col].fillna(method='bfill', inplace=True)
    df[col].fillna(method='ffill', inplace=True)

print(f"Smoothing applied with window size = {MOVING_AVG_WINDOW}")

# -------------------------------
# 3. SPLIT FEATURES & LABELS
# -------------------------------
label_col = 'label'  # change if needed

X = df.drop(columns=[label_col])
y = df[label_col]

# -------------------------------
# 4. TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# -------------------------------
# 5. FEATURE SCALING
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/Fall_detector_scaler.pkl")
print("Scaler saved as Fall_detector_scaler.pkl")

# -------------------------------
# 6. HYPERPARAMETER TUNING
# -------------------------------
param_grid = {
    'n_estimators': [100, 200, 300],   # Number of trees
    'max_depth': [None, 10, 20],       # Tree depth
    'min_samples_split': [2, 5, 10],   # Min samples to split
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    param_grid,
    cv=3,              # 3-fold cross-validation
    n_jobs=-1,         # Use all cores
    verbose=2
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

print(f"Best parameters found: {grid_search.best_params_}")

# -------------------------------
# 7. MODEL EVALUATION
# -------------------------------
y_pred = best_model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_model.classes_,
            yticklabels=best_model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# 8. SAVE BEST MODEL
# -------------------------------
joblib.dump(best_model, "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/Fall_detector_rf_model.pkl")
print("Best Random Forest model saved as Fall_detector_rf_model.pkl")
