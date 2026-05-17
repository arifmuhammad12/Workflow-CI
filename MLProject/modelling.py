import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
  accuracy_score,
  precision_score,
  recall_score,
  f1_score,
  classification_report
)

# Set experiment
mlflow.set_experiment(
  "telco-customer-churn"
)

# Enable autolog
mlflow.sklearn.autolog(
  registered_model_name="TelcoChurnModel"
)

# Load dataset
df = pd.read_csv(
  "telco_preprocessing/telco_clean.csv"
)

# Split fitur dan target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.2,
  random_state=42,
  stratify=y
)



  # Random Forest anti overfitting
model = RandomForestClassifier(
  n_estimators=100,
  max_depth=5,
  min_samples_split=5,
  min_samples_leaf=2,
  class_weight="balanced",
  random_state=42
)

  # Training
model.fit(
  X_train,
  y_train
)

  # Prediction
y_pred = model.predict(X_test)

  # Evaluation
accuracy = accuracy_score(
  y_test,
  y_pred
)

precision = precision_score(
  y_test,
  y_pred
)

recall = recall_score(
  y_test,
  y_pred
)

f1 = f1_score(
  y_test,
  y_pred
)

  # Print metric test
print(f"Accuracy : {accuracy}")
print(f"Precision: {precision}")
print(f"Recall   : {recall}")
print(f"F1 Score : {f1}")

print("\nClassification Report:\n")
print(
  classification_report(
    y_test,
    y_pred
  )
)

  # Save model
joblib.dump(
  model,
  "random_forest_model.pkl"
)

mlflow.sklearn.log_model(
  model,
  "model"
)