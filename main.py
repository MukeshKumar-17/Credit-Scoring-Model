
# main.py - separate features and target

from src.data_loader import load_data

# load dataset
df = load_data("../DataC/german.data-numeric")

# separate features (X) and target (y)
X = df.drop('Target', axis=1)
y = df['Target']

# convert target: 1->0 (good), 2->1 (bad)
y = y.replace({1: 0, 2: 1})

# check the shapes
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# verify target conversion
print(f"\nTarget value counts:\n{y.value_counts()}")

# split into train and test
from sklearn.model_selection import train_test_split

# random_state=42 ensures the same split every time (reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining shapes: {X_train.shape}, {y_train.shape}")
print(f"Testing shapes: {X_test.shape}, {y_test.shape}")

# standard scaling (mean=0, std=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use same scaler!

# train model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

print("\nModel trained successfully!")

# evaluate model
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f} ({accuracy*100:.1f}%)")

# detailed evaluation
from sklearn.metrics import confusion_matrix, classification_report

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Feature Importance ---
import pandas as pd

# Get model weights (coefficients)
weights = model.coef_[0]
feature_names = X.columns

# Create a dataframe to view them easily
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Weight': weights
})

# Sort by importance
feature_importance = feature_importance.sort_values(by='Weight', ascending=False)

print("\n--- Feature Importance ---")
print("Top 5 Risk Factors (Bad Credit):")
print(feature_importance.head(5))

print("\nTop 5 Protective Factors (Good Credit):")
print(feature_importance.tail(5))

# --- Model Comparison ---
from sklearn.ensemble import RandomForestClassifier

print("\n--- Model Comparison ---")

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate
rf_pred = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print(f"Random Forest Accuracy:     {rf_acc:.4f}")

# --- Final Decision ---
print("\n--- Model Selection ---")
if rf_acc > accuracy:
    print("🏆 Random Forest is selected for the MVP!")
    print("Reason: Better accuracy.")
    print("Note: In real credit scoring, we might still prefer Logistic Regression for explainability.")
else:
    print("🏆 Logistic Regression is selected for the MVP!")
    print("Reason: Simpler, easier to explain, and performs just as well here.")
    print("Note: In finance, being able to explain 'WHY' a loan was rejected is critical.")