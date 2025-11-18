# src/predict.py

import pandas as pd
import joblib
import os
from src.config import Config
# Removed load_data, preprocess_data as they are not directly used in this module's functions

# Removed load_model function

def generate_predictions(model, X_test_scaled, label_encoder):
    """
    Generates predictions on the test set using the trained model.
    """
    predictions_encoded = model.predict(X_test_scaled)
    predictions_decoded = label_encoder.inverse_transform(predictions_encoded)
    return predictions_decoded

def create_submission_file(test_df, predictions, model_name, submission_dir=Config.BASE_DIR):
    """
    Creates the submission file in the required format with a unique name for each model.
    """
    submission_filename = f"submission_{model_name}.csv"
    submission_path = os.path.join(submission_dir, '..', 'submissions',submission_filename) # Go up one level from src
    submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'personality_cluster': predictions})
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file created at {submission_path}")

# Removed if __name__ == '__main__': block
