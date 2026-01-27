# Credit Scoring Model

A machine learning web application to predict credit risk using the German Credit Dataset.

## Features

- **3 ML Models**: Logistic Regression, Random Forest, XGBoost
- **Interactive UI**: Built with Streamlit
- **Real-time Predictions**: Good, Medium, or Bad credit risk
- **Model Comparison**: View accuracy of all models
- **Saved Models**: Pre-trained models ready for deployment

## Demo

Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                  # Streamlit web application
├── main.py                 # Model training script
├── Dataset/                # German Credit Dataset
│   ├── german.data-numeric
│   └── german.doc
├── models/                 # Saved trained models
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
└── README.md
```

## Models Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~77% |
| Random Forest | ~81% |
| XGBoost | ~80-82% |

## Dataset

**German Credit Dataset** (UCI Machine Learning Repository)
- 1000 samples
- 20 features (Account Status, Duration, Credit History, Purpose, etc.)
- Binary classification: Good (0) vs Bad (1) credit

## Installation

```bash
# Install dependencies
pip install streamlit pandas numpy scikit-learn xgboost

# Run the app
streamlit run app.py
```

## Usage

1. Select a model from the sidebar
2. Enter customer information using the form
3. Click "Predict Credit Risk"
4. View prediction result with confidence score

## Load Saved Models

```python
import joblib

# Load model
model = joblib.load('models/xgboost.pkl')
scaler = joblib.load('models/scaler.pkl')

# Make prediction
prediction = model.predict(input_data)
```

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- XGBoost
- Pandas / NumPy

## Author

Mukesh Kumar K

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
