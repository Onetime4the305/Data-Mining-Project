import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load the datasets
df = pd.read_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Modified_Steam_Data8.xlsx")
df2 = pd.read_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Modified_Steam_Data7.xlsx")
df3 = pd.read_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Modified_Steam_Data6.xlsx")
df4 = pd.read_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Modified_Steam_Data5.xlsx")  # Week 4 actual data

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

df4 = df4.dropna()  # Week 4 data preprocessing
df4['Genre'] = df4['Genre'].map({
    'FPS': 0, 'MOBA': 1, 'Battle Royale': 2, 'Open World': 3, 'Survival': 4,
    'Design & Illustration': 5, 'Clicker': 6, 'Simulation': 7, 'MMO': 8,
    'RPG': 9, 'Sports': 10, 'Action': 11, 'Strategy': 12, 'Side Scroller': 13,
    'VR': 14, 'Roguelike': 15, 'Adventure': 16, 'Sandbox': 17
})

# Combine the datasets for overall performance
combined_df = pd.concat([df, df2, df3], axis=0)

# Filter out underrepresented genres (with less than a threshold count)
threshold = 1
genre_counts = combined_df['Genre'].value_counts()
filtered_df = combined_df[combined_df['Genre'].isin(genre_counts[genre_counts >= threshold].index)]

# Feature Selection
X_filtered = filtered_df[['Avg.Players', 'Peak Players', 'Hours Played', 'Avg. Hours', 'Price', 'Player_Ratio']]
y_filtered = filtered_df['Avg.Players']

# Split the filtered data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Preprocess the Week 4 data for prediction
X_week4 = df4[['Avg.Players', 'Peak Players', 'Hours Played', 'Avg. Hours', 'Price', 'Player_Ratio']]
X_week4_scaled = scaler.transform(X_week4)

# Make predictions for Week 4 data
predictions_week4 = model.predict(X_week4_scaled)

# Multiply Avg.Players predictions by 100,000
predictions_week4_scaled = predictions_week4

# Compare the predicted and actual values
week_4_actual_data = pd.read_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Modified_Steam_Data5.xlsx")  # Actual data for Week 4

# Extract the actual `Avg.Players` for Week 4 from the dataset
week_4_actual_data['Predicted Avg.Players'] = predictions_week4_scaled

# Print the comparison between predicted and actual Avg.Players values for Week 4
print(week_4_actual_data[['Name', 'Avg.Players', 'Predicted Avg.Players']])
