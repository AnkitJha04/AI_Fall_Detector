# ================================
# Fall Detection EDA Script
# ================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# 1. Load Training Data
# --------------------
# Change these paths if the CSV files are in a different folder
train_df = pd.read_csv("C:/Users/ankit/OneDrive/Documents/Personal/Repls/Complete code/Random Python/Fall detector/Resources/Train.csv")
test_df = pd.read_csv("C:/Users/ankit/OneDrive/Documents/Personal/Repls/Complete code/Random Python/Fall detector/Resources/Test.csv")

# --------------------
# 2. Mapping Shortform Labels to Full Names
# --------------------
label_mapping = {
    "SDL": "Slow Fall Left",
    "SDR": "Slow Fall Right",
    "FOL": "Forward Fall",
    "BKL": "Backward Fall Left",
    "BKR": "Backward Fall Right",
    "STD": "Standing",
    "WAL": "Walking",
    "SIT": "Sitting"
}

# Assume your label column is named 'Label' (change if needed)
train_df['Label_Full'] = train_df['Label'].map(label_mapping)

# --------------------
# 3. Plot Overall Label Balance
# --------------------
plt.figure(figsize=(8, 5))
sns.countplot(x='Label_Full', data=train_df, order=train_df['Label_Full'].value_counts().index)
plt.title("Label Balance in Training Dataset", fontsize=14)
plt.xlabel("Activity", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --------------------
# 4. Plot Sub-Activity Counts
# --------------------
# If 'Label' already contains sub-activity codes, this is the same as above but with shortform on x-axis
plt.figure(figsize=(8, 5))
sns.countplot(x='Label', data=train_df, order=train_df['Label'].value_counts().index, palette="viridis")
plt.title("Sub-Activity Counts (Shortform)", fontsize=14)
plt.xlabel("Sub-Activity Code", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.tight_layout()
plt.show()

# --------------------
# 5. Correlation Heatmap for Numeric Features
# --------------------
numeric_cols = train_df.select_dtypes(include=['number']).columns

corr_matrix = train_df[numeric_cols].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, xticklabels=numeric_cols, yticklabels=numeric_cols)
plt.title("Correlation Matrix of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()
