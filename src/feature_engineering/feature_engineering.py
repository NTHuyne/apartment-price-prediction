import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
import joblib

def preprocess_data(data_path, save_data_path, save_pipeline_path):
    # Load data
    data = pd.read_csv(data_path)
    data.drop_duplicates(inplace=True)
    # data = clean_and_concat_street_precinct(data)

    # data.drop(columns=['duAn'], inplace=True)

    # Handle missing values and outlier removal
    categorical_cols = ['District', 'phapLy', 'huong', 'Precinct', 'duAn']
    numerical_cols = ['acreage_value', 'noBed', 'noBathroom', 'soLau']

    # Fill missing numerical values with -1
    for col in ['noBed', 'noBathroom', 'soLau']:
        data[col] = data[col].fillna(-1)
    
    # Pipeline for categorical data
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="constant", fill_value = "NO INFO")),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    #Pipeline for numerical data
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="median")),  
        ('scaler', StandardScaler() )
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ]
    )
    
    # Transform data
    X = data.drop(columns=['price_value', 'Street', 'price_per_area'])
    y = data['price_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save pipeline and data splits
    joblib.dump(preprocessor, save_pipeline_path)
    X_train.to_csv(os.path.join(save_data_path, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(save_data_path, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(save_data_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(save_data_path, 'y_test.csv'), index=False)
    print("Data preprocessing completed and saved.")

if __name__ == "__main__":
    preprocess_data("C:\\Users\\THIS PC\\Desktop\\apartment-price-prediction\\data\\normalized\\clean_dataset4.csv", "C:\\Users\\THIS PC\\Desktop\\apartment-price-prediction\\data\\finalized", "C:\\Users\\THIS PC\\Desktop\\apartment-price-prediction\\model\\feature_engineering\\preprocessor.pkl")
