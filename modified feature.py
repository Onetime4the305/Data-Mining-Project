import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Steam1RawData.xlsx")

# Clean column names by stripping any leading/trailing spaces
df.columns = df.columns.str.strip()

# Verify column names to ensure they match what you're using
print(df.columns)

# Define price bins and labels with the corrected ranges
bins = [-np.inf, 0, 19.99, 49.99, np.inf]
labels = ['Free', 'Budget', 'Mid-range', 'Premium']

# Apply the price bins and create the 'Price_Bin' column
df['Price_Bin'] = pd.cut(df['Price'], bins=bins, labels=labels)

# Handle the 'Player_Ratio' (Avg.Players / Peak Players) and avoid division by zero
df['Player_Ratio'] = df['Avg.Players'] / df['Peak Players'].replace(0, np.nan)

# Handle 'Price_Adjusted_Hours' and avoid division by zero (set to 0 for free games instead of NaN)
df['Price_Adjusted_Hours'] = df['Hours Played'] / df['Price'].replace(0, np.nan)

# Set 'Price_Adjusted_Hours' to 0 for free games (Price == 0)
df.loc[df['Price'] == 0, 'Price_Adjusted_Hours'] = 0

# Replace infinite values in 'Price_Adjusted_Hours' with zero
df['Price_Adjusted_Hours'] = df['Price_Adjusted_Hours'].replace([np.inf, -np.inf], 0)

# Handle 'Price_Avg_Hours_Interaction' as a product of Price and Avg.Hours
df['Price_Avg_Hours_Interaction'] = df['Price'] * df['Avg. Hours']

# Optionally, save the modified DataFrame to a new Excel file
output_file = r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Modified_Steam_Data8.xlsx"
df.to_excel(output_file, index=False)

# Show the first few rows of the modified DataFrame to verify
print(df.head())
