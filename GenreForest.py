import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np

# Step 1: Load the datasets
df = pd.read_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Modified_Steam_Data.xlsx")
df2 = pd.read_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Modified_Steam_Data2.xlsx")
df3 = pd.read_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Modified_Steam_Data3.xlsx")

# Data Preprocessing (handle missing values, encode categorical variables)
df = df.dropna()
df['Genre'] = df['Genre'].map({
    'FPS': 0, 'MOBA': 1, 'Battle Royale': 2, 'Open World': 3, 'Survival': 4,
    'Design & Illustration': 5, 'Clicker': 6, 'Simulation': 7, 'MMO': 8,
    'RPG': 9, 'Sports': 10, 'Action': 11, 'Strategy': 12, 'Side Scroller': 13,
    'VR': 14, 'Roguelike': 15, 'Adventure': 16, 'Sandbox': 17
})

df2 = df2.dropna()
df2['Genre'] = df2['Genre'].map({
    'FPS': 0, 'MOBA': 1, 'Battle Royale': 2, 'Open World': 3, 'Survival': 4,
    'Design & Illustration': 5, 'Clicker': 6, 'Simulation': 7, 'MMO': 8,
    'RPG': 9, 'Sports': 10, 'Action': 11, 'Strategy': 12, 'Side Scroller': 13,
    'VR': 14, 'Roguelike': 15, 'Adventure': 16, 'Sandbox': 17
})

df3 = df3.dropna()
df3['Genre'] = df3['Genre'].map({
    'FPS': 0, 'MOBA': 1, 'Battle Royale': 2, 'Open World': 3, 'Survival': 4,
    'Design & Illustration': 5, 'Clicker': 6, 'Simulation': 7, 'MMO': 8,
    'RPG': 9, 'Sports': 10, 'Action': 11, 'Strategy': 12, 'Side Scroller': 13,
    'VR': 14, 'Roguelike': 15, 'Adventure': 16, 'Sandbox': 17
})

# Combine the datasets for overall performance
combined_df = pd.concat([df, df2, df3], axis=0)

# Filter out underrepresented genres (with less than a threshold count)
threshold = 5
genre_counts = combined_df['Genre'].value_counts()
filtered_df = combined_df[combined_df['Genre'].isin(genre_counts[genre_counts >= threshold].index)]

# Feature Selection
X_filtered = filtered_df[['Avg.Players', 'Peak Players', 'Hours Played', 'Avg. Hours', 'Price', 'Player_Ratio']]
y_filtered = filtered_df['Genre']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation for the filtered data
cv_scores_filtered = cross_val_score(rf_classifier, X_scaled, y_filtered, cv=5)

# Plotting the cross-validation results
plt.figure(figsize=(10, 6))
sns.boxplot(data=cv_scores_filtered)
plt.title('Cross-Validation Accuracy Scores (Filtered Data)')
plt.ylabel('Accuracy')
plt.show()

# Train the model and make predictions
rf_classifier.fit(X_scaled, y_filtered)
y_pred = rf_classifier.predict(X_scaled)

# Classification report
print("Classification Report for Filtered Data:\n", classification_report(y_filtered, y_pred))

# Accuracy score
print("Accuracy Score for Filtered Data:", accuracy_score(y_filtered, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_filtered, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=filtered_df['Genre'].unique(), yticklabels=filtered_df['Genre'].unique())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Genre')
plt.ylabel('True Genre')
plt.show()

# Plotting feature importance
features = ['Avg.Players', 'Peak Players', 'Hours Played', 'Avg. Hours', 'Price', 'Player_Ratio']
feature_importances = rf_classifier.feature_importances_

plt.figure(figsize=(10, 6))
sns.barplot(x=features, y=feature_importances)
plt.title('Feature Importance')
plt.ylabel('Importance')
plt.show()
