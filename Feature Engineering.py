import pandas as pd

# Load the data
df = pd.read_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Normalized_Steam_Data3.xlsx")

# Print column names to check for any further discrepancies
print(df.columns)


# 1. Price Binning - Correct the bin edges to make them unique
bins = [0, 20, 50, 100, float('inf')]  # Define price bins: Free, Budget, Mid-range, Premium
labels = ['Free', 'Budget', 'Mid-range', 'Premium']  # Corresponding labels
df['Price_Bin'] = pd.cut(df['Price'], bins=bins, labels=labels)

# 2. Average Players vs. Peak Players Ratio
if 'Avg.Players' in df.columns and 'Peak Players' in df.columns:
    df['Player_Ratio'] = df['Avg.Players'] / df['Peak Players']

# 5. Price-Adjusted Hours Played
if 'Hours Played' in df.columns and 'Price' in df.columns:
    df['Price_Adjusted_Hours'] = df['Hours Played'] / df['Price']

# 11. Interaction Term: Price * Avg. Hours Played
if 'Price' in df.columns and 'Avg. Hours' in df.columns:
    df['Price_Avg_Hours_Interaction'] = df['Price'] * df['Avg. Hours']

# Check the new features
print(df[['Price', 'Price_Bin', 'Avg.Players', 'Peak Players', 'Player_Ratio', 'Hours Played', 'Price_Adjusted_Hours', 'Avg. Hours', 'Price_Avg_Hours_Interaction']].head())

# Optionally, save the modified DataFrame to a new file
df.to_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Feature_Engineered_Steam_Data3.xlsx", index=False)
