import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load and clean data
data = pd.read_csv('steam_game_data.csv')

# Feature engineering (e.g., encoding genre)
data = pd.get_dummies(data, columns=['Genre'], drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
numerical_features = ['Price', 'Hours Played', 'Current Players', 'Price Category']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Define target and features
X = data.drop('Peak Players', axis=1)  # Features
y = data['Peak Players']  # Target variable

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Feature importance
importances = rf.feature_importances_
feature_names = X.columns
importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print(importances_df.sort_values(by='Importance', ascending=False))
