import pandas as pd

# Load the dataset
df = pd.read_csv('data/creditcard.csv')

# Display the first 5 rows
print("First 5 rows:")
print(df.head())

# Info about columns, data types, and non-null values
print("\nInfo:")
print(df.info())

# Statistics summary
print("\nDescribe:")
print(df.describe())

# Check class distribution (0 = legit, 1 = fraud)
print("\nClass distribution:")
print(df['Class'].value_counts())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Remove duplicates if any
df = df.drop_duplicates()

# Save cleaned data (optional step)
df.to_csv('data/creditcard_clean.csv', index=False)
print("\nCleaned data saved as data/creditcard_clean.csv")
