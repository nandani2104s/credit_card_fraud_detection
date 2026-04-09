import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load features
df = pd.read_csv('data/creditcard_features.csv')

# Separate features and target
X = df.drop(['Class'], axis=1)
y = df['Class']

# Train-test split (stratify to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build Random Forest model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'data/random_forest_model.joblib')
print("Model trained and saved as data/random_forest_model.joblib")
