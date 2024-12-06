import pandas as pd
import joblib
import pickle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestRegressor

def log_results(model_name, params, mae, rmse):
    """Log results into a JSON file."""
    results = {
        "model": model_name,
        "params": params,
        "mae": mae,
        "rmse": rmse
    }
    with open("tuning_results.txt", "a") as f:
        f.write(json.dumps(results) + "\n")

def train_and_save_models(save_path="C:\\Users\\THIS PC\\Desktop\\apartment-price-prediction\\model\\models", data_path="C:\\Users\\THIS PC\\Desktop\\apartment-price-prediction\\data\\finalized"):
    # Load data
    X_train = pd.read_csv(os.path.join(data_path,'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_path,'y_train.csv')).values.ravel()
    preprocessor = joblib.load('C:\\Users\\THIS PC\\Desktop\\apartment-price-prediction\\model\\feature_engineering\\preprocessor.pkl')
    X_train_transformed = preprocessor.fit_transform(X_train)
    joblib.dump(preprocessor, "C:\\Users\\THIS PC\\Desktop\\apartment-price-prediction\\model\\feature_engineering\\preprocessor.pkl")
    
    # Models
    models = {
        "LinearRegression": LinearRegression(),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(),
        # "MLP": MLPRegressor(max_iter=500)
    }

    # pca = TruncatedSVD(n_components=500)
    # pca.fit(X_train_transformed)
    # X_train_transformed = pca.transform(X_train_transformed)
    # joblib.dump(pca, "C:\\Users\\THIS PC\\Desktop\\apartment-price-prediction\\model\\feature_engineering\\pca.pkl")
    
    # Hyperparameters for tuning
    param_grids = {
        "SVR": {'C': [0.1, 1, 10], 'gamma': [0.1, 1]},
        "XGBoost": {'n_estimators': [600, 700, 800], 'max_depth': [7, 9], 'learning_rate': [0.1, 0.2, 0.3]},
        # "MLP": {'hidden_layer_sizes': [(128, 64), (128, 128, 64), (64, 128, 128, 64, )], 'learning_rate_init': [0.001, 0.01]}
    }
    
    # Train and save models
    for model_name, model in models.items():
        if model_name in param_grids:
            grid = GridSearchCV(model, param_grids[model_name], cv=5, scoring='neg_mean_absolute_error', return_train_score=True, verbose=3, refit=True)
            grid.fit(X_train_transformed, y_train)
            best_model = grid.best_estimator_
            best_params = grid.best_params_
            # Evaluate on training set
            y_pred_train = best_model.predict(X_train_transformed)
            mae = mean_absolute_error(y_train, y_pred_train)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            log_results(model_name, best_params, mae, rmse)
        else:
            model.fit(X_train_transformed, y_train)
            y_pred_train = model.predict(X_train_transformed)
            mae = mean_absolute_error(y_train, y_pred_train)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            log_results(model_name, {}, mae, rmse)
            best_model = model
        
        # Save the best model
        with open(os.path.join(save_path, f"{model_name}.pkl"), 'wb') as f:
            pickle.dump(best_model, f)
        print(f"{model_name} trained and saved.")

if __name__ == "__main__":
    train_and_save_models()
