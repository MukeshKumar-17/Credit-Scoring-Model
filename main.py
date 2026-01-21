
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining shapes: {X_train.shape}, {y_train.shape}")
print(f"Testing shapes: {X_test.shape}, {y_test.shape}")

# train model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# evaluate model
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f} ({accuracy*100:.1f}%)")