# SignalCluster Classification Challenge

## Project Overview

Welcome to the SignalCluster Classification Challenge. This project aims to develop a model that predicts the personality cluster of each observation based on two signal measurements: `signal_strength` and `response_level`. The problem is a multiclass classification task, and submissions will be evaluated using Macro F1 Score.

The dataset contains synthetic two-dimensional signal observations, where each sample represents a point defined by two continuous features. The goal is to classify each sample into one of several cluster categories based on its signal characteristics.

## File and Folder Structure

The project is organized to separate data, source code, notebooks for exploration, and saved models.

```
E:\IIITB\ML\Project 2\
├── data\
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks\
│   └── eda_and_visualization.ipynb
├── src\
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── train.py
│   └── predict.py
├── models\
│   └── best_model.pkl
├── requirements.txt
├── main.py
└── submission.csv
```

-   `data/`: Contains the raw data files (`train.csv`, `test.csv`, `sample_submission.csv`).
-   `notebooks/eda_and_visualization.ipynb`: A Jupyter notebook for exploratory data analysis, visualizations, and initial model prototyping.
-   `src/`: The main source code for the project.
    -   `__init__.py`: An empty file to mark the `src` directory as a Python package.
    -   `config.py`: Stores all configuration variables, such as file paths, model parameters, and random states.
    -   `data_preprocessing.py`: Contains functions for loading, cleaning, and preparing the data for modeling (e.g., feature scaling, target encoding).
    -   `train.py`: Handles the model training process, including model selection, training, and evaluation. It saves the trained model.
    -   `predict.py`: Uses the trained model to generate predictions on the test set and creates the submission file.
-   `models/`: This directory will store the final, trained machine learning model (`best_model.pkl`).
-   `requirements.txt`: Lists all the Python libraries required to run the project.
-   `main.py`: The main script to execute the entire pipeline from data preprocessing to training and prediction.
-   `submission.csv`: The final output file with predictions for the test set.

## Execution Flow

The project will be executed via the `main.py` script, which orchestrates the entire process:

1.  **Load Data**: Reads `train.csv` and `test.csv` using functions from `src/data_preprocessing.py`.
2.  **Preprocess Data**: Applies necessary preprocessing steps like feature scaling and polynomial feature generation.
3.  **Train and Evaluate Models**: Trains and evaluates a suite of classification models (Logistic Regression, SVC, RandomForestClassifier, XGBClassifier, LGBMClassifier, KNeighborsClassifier, DecisionTreeClassifier, GaussianNB, AdaBoostClassifier, GradientBoostingClassifier) along with ensemble methods (VotingClassifier, StackingClassifier) using `src/train.py`. Hyperparameter tuning is performed using `RandomizedSearchCV` with `StratifiedKFold`.
4.  **Generate Predictions**: Loads the best performing model and generates predictions on the preprocessed test data using `src/predict.py`.
5.  **Create Submission File**: Formats the predictions into the required `submission.csv` file.

To run the entire pipeline, execute the following command in your terminal:

```bash
python main.py
```

## Why this Project Structure?

This project structure follows best practices for machine learning projects, promoting:

-   **Modularity**: Each component (data preprocessing, training, prediction) is separated into its own module, making the codebase easier to understand, maintain, and test.
-   **Reusability**: Functions and classes within `src/` can be easily reused across different experiments or projects.
-   **Reproducibility**: Configuration parameters are centralized in `config.py`, ensuring consistent settings for experiments. The separation of concerns also helps in reproducing results.
-   **Scalability**: As the project grows, new models or preprocessing steps can be added without significantly altering existing code.
-   **Collaboration**: A clear structure facilitates collaboration among team members, as each person can work on different modules independently.

## Requirements

The `requirements.txt` file includes the following Python libraries:

```
kaggle
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
xgboost
lightgbm
```


## Detailed Implementation

### Data Preprocessing
The raw `signal_strength` and `response_level` features undergo several preprocessing steps:
-   **Feature Scaling**: `StandardScaler` from scikit-learn is applied to normalize the features, which is beneficial for many models.
-   **Polynomial Features**: To capture non-linear relationships and interactions between the original features, `PolynomialFeatures` with a degree of 2 are generated. This expands the feature space from 2 to 5 dimensions (original features, their squares, and their interaction term).
-   **Target Encoding**: The categorical `personality_cluster` labels are converted into numerical format using `LabelEncoder`.

### Model Selection and Training
A comprehensive suite of classification models was trained and evaluated to determine the best performer for this multiclass classification problem. The models include:
-   Logistic Regression
-   Support Vector Classifier (SVC)
-   Random Forest Classifier
-   XGBoost Classifier
-   LightGBM Classifier
-   K-Nearest Neighbors Classifier
-   Decision Tree Classifier
-   Gaussian Naive Bayes
-   AdaBoost Classifier
-   Gradient Boosting Classifier

For each model (except GaussianNB, which doesn't have hyperparameters to tune in this context), hyperparameter tuning was performed using `RandomizedSearchCV` with 50 iterations and 5-fold `StratifiedKFold` cross-validation. This approach helps in finding optimal hyperparameters while accounting for potential class imbalance. Class weights were computed and applied to models to further mitigate the impact of imbalanced classes.

### Ensemble Methods
To potentially boost performance beyond individual models, two ensemble techniques were employed:
-   **Voting Classifier**: This ensemble combines predictions from multiple diverse base models. It was tuned to find optimal weights for the contributing models.
-   **Stacking Classifier**: This method trains a meta-learner to combine the predictions of several base models.

### Evaluation Metric
All models were evaluated using the **Macro F1 Score**, which is the specified metric for this challenge. Macro F1 Score calculates the F1 score for each class independently and then takes the unweighted average, ensuring that all clusters are treated fairly regardless of their frequency.

## Results

The models were evaluated on a validation set, and their Macro F1 Scores are as follows:

-   **SVC**: 0.9863669235842568
-   **VotingClassifier**: 0.9862000802253159
-   RandomForestClassifier: 0.9821174640885086
-   XGBClassifier: 0.9816674199538168
-   KNeighborsClassifier: 0.9811084645125656
-   StackingClassifier: 0.9813431987634115
-   LGBMClassifier: 0.9765283548135514
-   DecisionTreeClassifier: 0.9768129790997214
-   LogisticRegression: 0.9660957400087834
-   GradientBoostingClassifier: 0.9663466335395335
-   AdaBoostClassifier: 0.9125499695853961
-   GaussianNB: 0.8702708697479057

The **SVC** and **VotingClassifier** achieved the highest Macro F1 Scores on the validation set, indicating excellent performance in classifying the signal clusters.

## Further Improvements (Future Work)

To potentially push the Macro F1 Score even higher and gain deeper insights, the following areas could be explored:

1.  **More Extensive Hyperparameter Tuning**: Conduct more exhaustive searches (e.g., `GridSearchCV` or more iterations for `RandomizedSearchCV`) for the best-performing models.
2.  **Advanced Ensemble Techniques**: Experiment with more sophisticated ensemble methods or custom weighting schemes for the `VotingClassifier`.
3.  **Enhanced Feature Engineering**: Explore additional non-linear transformations or interaction terms beyond degree-2 polynomial features.
4.  **Comprehensive Decision Boundary Visualizations**: Implement methods to visualize decision boundaries for all models, even those trained on higher-dimensional feature spaces, possibly using dimensionality reduction techniques like PCA or t-SNE.
5.  **Additional Exploratory Data Analysis (EDA)**:
    *   Utilize **Violin Plots** for a richer view of feature distributions across classes.
    *   Perform **Outlier Analysis** using formal detection methods.
    *   Conduct **Feature Importance Analysis** for tree-based models to understand feature contributions.
    *   Perform **Error Analysis** on misclassified samples to identify patterns and areas for model improvement.
