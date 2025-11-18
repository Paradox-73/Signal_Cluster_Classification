# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from src.config import Config
import numpy as np # Import numpy for log/exp transformations

def load_data(train_path=Config.TRAIN_FILE, test_path=Config.TEST_FILE):
    """
    Loads training and testing data from CSV files.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def add_engineered_features(df):
    """
    Adds additional engineered features to the DataFrame.
    These features include products, ratios, logarithmic, and exponential transformations.
    """
    df_copy = df.copy()

    # Product and Ratio Features
    df_copy['signal_x_response'] = df_copy['signal_strength'] * df_copy['response_level']
    df_copy['signal_div_response'] = df_copy['signal_strength'] / (df_copy['response_level'] + 1e-6) # Add small epsilon to avoid division by zero
    df_copy['response_div_signal'] = df_copy['response_level'] / (df_copy['signal_strength'] + 1e-6)

    # Logarithmic Transformations (handle non-positive values)
    df_copy['log_signal'] = np.log1p(df_copy['signal_strength'] - df_copy['signal_strength'].min() + 1) # log1p for robustness
    df_copy['log_response'] = np.log1p(df_copy['response_level'] - df_copy['response_level'].min() + 1)

    # Exponential Transformations
    df_copy['exp_signal'] = np.exp(df_copy['signal_strength'] / 100) # Scaling down for more manageable values
    df_copy['exp_response'] = np.exp(df_copy['response_level'] / 100)

    print("Added additional engineered features.")
    return df_copy

def preprocess_data(train_df, test_df, use_polynomial_features=False, polynomial_degree=2, use_additional_engineered_features=False):
    """
    Applies preprocessing steps to the data, including feature scaling, optional polynomial feature creation,
    optional additional engineered features, and target encoding.
    """
    # Separate features and target
    X = train_df[['signal_strength', 'response_level']]
    y = train_df['category']
    X_test = test_df[['signal_strength', 'response_level']]

    # Apply additional engineered features if enabled
    if use_additional_engineered_features:
        X = add_engineered_features(X)
        X_test = add_engineered_features(X_test)

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Optional: Polynomial Features
    if use_polynomial_features:
        poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
        X_scaled_poly = poly.fit_transform(X_scaled_df)
        X_test_scaled_poly = poly.transform(X_test_scaled_df)

        # Update feature names
        poly_feature_names = poly.get_feature_names_out(X_scaled_df.columns)
        X_scaled_df = pd.DataFrame(X_scaled_poly, columns=poly_feature_names, index=X.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled_poly, columns=poly_feature_names, index=X_test.index)
        print(f"Applied Polynomial Features with degree {polynomial_degree}. New feature count: {X_scaled_df.shape[1]}")


    # Target Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_scaled_df, y_encoded, X_test_scaled_df, label_encoder, scaler

def split_data(X, y):
    """
    Splits the training data into training and validation sets.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )
    return X_train, X_val, y_train, y_val
