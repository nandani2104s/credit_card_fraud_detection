import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv('data/creditcard_clean.csv')

# Log-transform the 'Amount' feature to reduce skew
df['Amount_log'] = np.log1p(df['Amount'])

# (Optional) Drop 'Amount' if you only want log-transformed feature
# df = df.drop(['Amount'], axis=1)

# Save engineered data
df.to_csv('data/creditcard_features.csv', index=False)
print("Feature engineering complete. Saved as data/creditcard_features.csv")

# Check new columns
print(df[['Amount', 'Amount_log']].head())
