import pandas as pd

# Load the dataset
df = pd.read_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Steam4RawData.xlsx")

# Normalize the player data columns (Avg.Players and Peak Players) by dividing by 100,000
df['Avg.Players'] = df['Avg.Players'] / 100000
df['Peak Players'] = df['Peak Players'] / 100000

# Normalize the hours data columns (Hours Played and Avg. Hours) by dividing by 1,000,000
df['Hours Played'] = df['Hours Played'] / 1000000
df['Avg. Hours'] = df['Avg. Hours'] / 1000000

# Check the normalization effect
print(df.head())

# Save the cleaned and normalized data to a new file
df.to_excel(r"C:\Users\dejea\OneDrive\Desktop\Data mining\Project\Normalized_Steam_Data4.xlsx", index=False)

# Verify the results are saved and normalized
print(df.head())
