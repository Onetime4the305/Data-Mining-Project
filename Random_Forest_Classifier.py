import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectFromModel

# Step 1: Load the primary and additional datasets
# Load your primary dataset
df = pd.read_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Modified_Steam_Data.xlsx")

# Load additional datasets (assuming they are in separate files or sheets)
df2 = pd.read_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Modified_Steam_Data2.xlsx")
df3 = pd.read_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Modified_Steam_Data3.xlsx")

# Step 2: Data Preprocessing (handle missing values, encode categorical variables)
df = df.dropna()
df['Price_Bin'] = df['Price_Bin'].map({'Free': 0, 'Budget': 1, 'Mid-range': 2, 'Premium': 3})

df2 = df2.dropna()
df2['Price_Bin'] = df2['Price_Bin'].map({'Free': 0, 'Budget': 1, 'Mid-range': 2, 'Premium': 3})

df3 = df3.dropna()
df3['Price_Bin'] = df3['Price_Bin'].map({'Free': 0, 'Budget': 1, 'Mid-range': 2, 'Premium': 3})

# Step 3: Feature Selection (ensure columns are consistent across datasets)
X = df[['Avg.Players', 'Peak Players', 'Hours Played', 'Avg. Hours', 'Price', 'Player_Ratio']]
y = df['Price_Bin']

X2 = df2[['Avg.Players', 'Peak Players', 'Hours Played', 'Avg. Hours', 'Price', 'Player_Ratio']]
y2 = df2['Price_Bin']

X3 = df3[['Avg.Players', 'Peak Players', 'Hours Played', 'Avg. Hours', 'Price', 'Player_Ratio']]
y3 = df3['Price_Bin']

# Step 4: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X2_scaled = scaler.transform(X2)
X3_scaled = scaler.transform(X3)

# Step 5: Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 6: Cross-validation for the primary dataset (df)
cv_scores_df = cross_val_score(rf_classifier, X_scaled, y, cv=5)
print(f"Cross-validated accuracy scores (df): {cv_scores_df}")
print(f"Mean Cross-validated accuracy (df): {cv_scores_df.mean()}")

# Step 7: Cross-validation for the second dataset (df2)
cv_scores_df2 = cross_val_score(rf_classifier, X2_scaled, y2, cv=5)
print(f"Cross-validated accuracy scores (df2): {cv_scores_df2}")
print(f"Mean Cross-validated accuracy (df2): {cv_scores_df2.mean()}")

# Step 8: Cross-validation for the third dataset (df3)
cv_scores_df3 = cross_val_score(rf_classifier, X3_scaled, y3, cv=5)
print(f"Cross-validated accuracy scores (df3): {cv_scores_df3}")
print(f"Mean Cross-validated accuracy (df3): {cv_scores_df3.mean()}")

# Step 9: Combine the datasets for overall performance evaluation
combined_df = pd.concat([df, df2, df3], axis=0)

# Select features and target variable for the combined dataset
X_combined = combined_df[['Avg.Players', 'Peak Players', 'Hours Played', 'Avg. Hours', 'Price', 'Player_Ratio']]
y_combined = combined_df['Price_Bin']

# Apply scaling on the combined dataset
X_combined_scaled = scaler.fit_transform(X_combined)

# Step 10: Cross-validation for the combined dataset
cv_scores_combined = cross_val_score(rf_classifier, X_combined_scaled, y_combined, cv=5)
print(f"Cross-validated accuracy scores (combined): {cv_scores_combined}")
print(f"Mean Cross-validated accuracy (combined): {cv_scores_combined.mean()}")

# Step 11: Train the model on the combined data (if you want to train on the entire dataset)
rf_classifier.fit(X_combined_scaled, y_combined)

# Step 12: Evaluate the model on the combined dataset
y_combined_pred = rf_classifier.predict(X_combined_scaled)
print("Classification Report for Combined Data:\n", classification_report(y_combined, y_combined_pred))
print("Accuracy Score for Combined Data:", accuracy_score(y_combined, y_combined_pred))

