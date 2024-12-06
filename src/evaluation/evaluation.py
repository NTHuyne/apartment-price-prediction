import pandas as pd
import joblib
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os

def evaluate_models():
    # Load data
    X_test = pd.read_csv('C:\\Users\\THIS PC\\Desktop\\apartment-price-prediction\\data\\finalized\\X_test.csv')
    y_test = pd.read_csv('C:\\Users\\THIS PC\\Desktop\\apartment-price-prediction\\data\\finalized\\y_test.csv').values.ravel()
    preprocessor = joblib.load('C:\\Users\\THIS PC\\Desktop\\apartment-price-prediction\\model\\feature_engineering\\preprocessor.pkl')
    X_test_transformed = preprocessor.transform(X_test)

    # Models to evaluate
    model_path = "C:\\Users\\THIS PC\\Desktop\\apartment-price-prediction\\model\\models"
    model_files = ["LinearRegression.pkl", "XGBoost.pkl", "SVR.pkl"]
    metrics = []
    
    for model_file in model_files:
        with open(os.path.join(model_path, model_file), 'rb') as f:
            model = pickle.load(f)
        
        # Predict
        y_pred = model.predict(X_test_transformed)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics.append((model_file.split('.')[0], mae, rmse))
    
    # Print evaluation results
    for model_name, mae, rmse in metrics:
        text = f"Model: {model_name} | MAE: {mae:.4f} | RMSE: {rmse:.4f}"
        print(text)
        with open("evaluation_results.txt", "a") as f:
            f.write(text + "\n")

if __name__ == "__main__":
    evaluate_models()
