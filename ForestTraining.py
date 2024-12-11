import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
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
threshold = 3
genre_counts = combined_df['Genre'].value_counts()
filtered_df = combined_df[combined_df['Genre'].isin(genre_counts[genre_counts >= threshold].index)]

# Feature Selection
X_filtered = filtered_df[['Avg.Players', 'Peak Players', 'Hours Played', 'Avg. Hours', 'Price', 'Player_Ratio']]
y_filtered = filtered_df['Genre']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_filtered, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 3: Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Step 4: Make predictions on the test set
y_test_pred = rf_classifier.predict(X_test)

# Step 5: Classification report for the test data
print("Classification Report for Test Data:\n", classification_report(y_test, y_test_pred))

# Step 6: Accuracy score for the test data
print("Accuracy Score for Test Data:", accuracy_score(y_test, y_test_pred))

# Step 7: Confusion Matrix for the test data
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=filtered_df['Genre'].unique(), yticklabels=filtered_df['Genre'].unique())
plt.title('Confusion Matrix for Test Data')
plt.xlabel('Predicted Genre')
plt.ylabel('True Genre')
plt.show()

# Optional: Plot feature importance for the trained model
features = ['Avg.Players', 'Peak Players', 'Hours Played', 'Avg. Hours', 'Price', 'Player_Ratio']
feature_importances = rf_classifier.feature_importances_

plt.figure(figsize=(10, 6))
sns.barplot(x=features, y=feature_importances)
plt.title('Feature Importance')
plt.ylabel('Importance')
plt.show()
