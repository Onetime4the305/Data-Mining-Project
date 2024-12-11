import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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
y_filtered = filtered_df['Avg.Players']  # Now 'Avg.Players' is the target

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_filtered, test_size=0.2, random_state=42)

# Step 3: Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Step 4: Train the model on the training data
rf_regressor.fit(X_train, y_train)

# Step 5: Make predictions on the test data
y_pred_test = rf_regressor.predict(X_test)

# Step 6: Evaluate the model's performance on the test set
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

# Output the results
print("Mean Squared Error for Test Data:", mse_test)
print("R² Score for Test Data:", r2_test)

# Optional: Plot the true vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # Perfect fit line
plt.title("True vs Predicted Values for Test Data")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.show()

# Optional: Plot the feature importances
feature_importances = rf_regressor.feature_importances_

plt.figure(figsize=(10, 6))
sns.barplot(x=X_filtered.columns, y=feature_importances)
plt.title('Feature Importance')
plt.ylabel('Importance')
plt.show()
