import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from imblearn.over_sampling import SMOTE

# Load features
df = pd.read_csv('data/creditcard_features.csv')

# Separate features and target
X = df.drop(['Class'], axis=1)
y = df['Class']

# Train-test split (stratify to keep class balance in test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE only on training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Original training set fraud count:", sum(y_train))
print("SMOTE training set fraud count:", sum(y_train_smote))

# Build and train Random Forest model on balanced data
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train_smote, y_train_smote)

# Save the trained model
joblib.dump(model, 'data/random_forest_model_smote.joblib')
print("SMOTE-balanced model trained and saved as data/random_forest_model_smote.joblib")
