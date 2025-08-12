import pandas as pd
import numpy as np
import random
from scipy.interpolate import interp1d

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# ==== CONFIGURATION ====
INPUT_FILE = "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Complete code/Random Python/Fall detector/Resources/Train.csv"       # Input original dataset
OUTPUT_FILE = "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Complete code/Random Python/Fall detector/Resources/Train_Augumented.csv"  # Output augmented dataset
AUGMENT_FACTOR = 10            # How many synthetic samples per original sample

# ==== LOAD DATA ====
df = pd.read_csv(INPUT_FILE)

# Identify columns
label_col = 'Label'  # Change if your label column has a different name
feature_cols = [col for col in df.columns if col != label_col]

# ==== AUGMENTATION FUNCTIONS ====
def add_gaussian_noise(data, noise_level=0.01):
    """Add Gaussian noise to each feature."""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def scale_features(data, scale_range=(0.9, 1.1)):
    """Scale feature values randomly."""
    scale_factor = np.random.uniform(*scale_range)
    return data * scale_factor

def time_warp(data, warp_strength=0.2):
    """
    Time-warp by stretching/compressing data along the time axis.
    Assumes data is 1D (single feature series), applied feature-wise.
    """
    orig_steps = np.arange(data.shape[0])
    new_steps = np.linspace(0, data.shape[0] - 1, int(data.shape[0] * (1 + np.random.uniform(-warp_strength, warp_strength))))
    f = interp1d(orig_steps, data, kind='linear', fill_value="extrapolate")
    warped = f(np.linspace(0, new_steps[-1], data.shape[0]))
    return warped

# ==== AUGMENTATION PIPELINE ====
augmented_data = []

for idx, row in df.iterrows():
    features = row[feature_cols].values.astype(float)
    label = row[label_col]
    
    for _ in range(AUGMENT_FACTOR):
        # Apply augmentations in sequence
        aug_features = add_gaussian_noise(features)
        aug_features = scale_features(aug_features)
        aug_features = np.array([time_warp(aug_features)])  # Keep as array

        # Flatten and append with label
        aug_row = list(aug_features.flatten()) + [label]
        augmented_data.append(aug_row)

# ==== CREATE AUGMENTED DATAFRAME ====
aug_df = pd.DataFrame(augmented_data, columns=feature_cols + [label_col])

# Combine with original data
final_df = pd.concat([df, aug_df], ignore_index=True)

# Save
final_df.to_csv(OUTPUT_FILE, index=False)

print(f"Augmented dataset saved to {OUTPUT_FILE}")
