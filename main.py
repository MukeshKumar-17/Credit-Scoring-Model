
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