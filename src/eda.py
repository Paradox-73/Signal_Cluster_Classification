# python -m src.eda

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA # Added for dimensionality reduction
from sklearn.manifold import TSNE # Added for dimensionality reduction
from sklearn.preprocessing import LabelEncoder # Added for class_names decoding


from src.config import Config

def plot_decision_boundary_2d(model, X, y, feature_names, class_names, title, filename):
    """
    Visualizes the decision boundary of a classifier in a 2D feature space.
    Assumes X has exactly two features.
    """
    # Create color maps for the plot
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Determine mesh grid
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Predict class for each point in the meshgrid
    # Predict class for each point in the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    # Plot also the training points
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y,
                    palette=cmap_bold, alpha=1.0, edgecolor="black", s=80,
                    legend='full')
    
    plt.title(title)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    # Create custom legend handles for class names
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[i], 
                          markerfacecolor=cmap_bold(i), markersize=10) for i in range(len(class_names))]
    plt.legend(title="Classes", handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    figures_dir = os.path.join(Config.BASE_DIR, '..', 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved decision boundary plot to {filepath}")


def visualize_model_decision_boundaries(model, X, y, model_name, label_encoder, method='pca'):
    """
    Visualizes the decision boundary of a classifier, using dimensionality reduction if features > 2.
    
    Args:
        model: Trained classifier model.
        X (pd.DataFrame): Feature data.
        y (np.array): Target data (encoded).
        model_name (str): Name of the model for plot title and filename.
        label_encoder (LabelEncoder): Encoder used for target variable to get class names.
        method (str): 'pca' or 'tsne' for dimensionality reduction if needed.
    """
    class_names = label_encoder.inverse_transform(np.unique(y))
    
    if X.shape[1] == 2:
        print(f"Plotting decision boundary for {model_name} on original 2 features.")
        plot_decision_boundary_2d(model, X, pd.Series(y, index=X.index), X.columns, class_names,
                                  f'Decision Boundary: {model_name} (Original Features)',
                                  f'decision_boundary_{model_name}_original.png')
    elif X.shape[1] > 2:
        print(f"Features > 2. Applying {method.upper()} for decision boundary visualization for {model_name}.")
        if method == 'pca':
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X)
            feature_names = ['Principal Component 1', 'Principal Component 2']
            title = f'Decision Boundary: {model_name} (PCA Reduced Features)'
            filename = f'decision_boundary_{model_name}_pca.png'
        elif method == 'tsne':
            # t-SNE is computationally intensive, consider sampling for large datasets
            # And it doesn't support transform method for new data, so fit_transform needs to be on all data
            # For visualization, we'll fit on X and use its output
            tsne = TSNE(n_components=2, random_state=Config.RANDOM_STATE, perplexity=min(30, X.shape[0]-1))
            X_reduced = tsne.fit_transform(X)
            feature_names = ['t-SNE Component 1', 't-SNE Component 2']
            title = f'Decision Boundary: {model_name} (t-SNE Reduced Features)'
            filename = f'decision_boundary_{model_name}_tsne.png'
        else:
            print(f"Invalid dimensionality reduction method: {method}. Skipping decision boundary plot for {model_name}.")
            return

        # Convert X_reduced to DataFrame for consistency with plot_decision_boundary_2d
        X_reduced_df = pd.DataFrame(X_reduced, columns=feature_names, index=X.index)
        plot_decision_boundary_2d(model, X_reduced_df, pd.Series(y, index=X.index), feature_names, class_names, title, filename)
    else:
        print(f"Cannot plot decision boundary for {model_name}: Insufficient features ({X.shape[1]}).")



def visualize_clusters(model, X, y, model_name, label_encoder, method='pca'):
    """
    Visualizes clusters (or classified regions) for models like KNN, using dimensionality reduction if features > 2.
    
    Args:
        model: Trained classifier model (e.g., KNeighborsClassifier).
        X (pd.DataFrame): Feature data.
        y (np.array): True target data (encoded).
        model_name (str): Name of the model for plot title and filename.
        label_encoder (LabelEncoder): Encoder used for target variable to get class names.
        method (str): 'pca' or 'tsne' for dimensionality reduction if needed.
    """
    class_names = label_encoder.inverse_transform(np.unique(y))
    
    X_plot = X.copy()
    plot_feature_names = X.columns.tolist()
    plot_title_suffix = "Original Features"
    plot_filename_suffix = "original_features"

    if X.shape[1] > 2:
        print(f"Features > 2. Applying {method.upper()} for cluster visualization for {model_name}.")
        if method == 'pca':
            pca = PCA(n_components=2)
            X_plot_reduced = pca.fit_transform(X)
            plot_feature_names = ['Principal Component 1', 'Principal Component 2']
            plot_title_suffix = "PCA Reduced Features"
            plot_filename_suffix = "pca"
        elif method == 'tsne':
            tsne = TSNE(n_components=2, random_state=Config.RANDOM_STATE, perplexity=min(30, X.shape[0]-1))
            X_plot_reduced = tsne.fit_transform(X)
            plot_feature_names = ['t-SNE Component 1', 't-SNE Component 2']
            plot_title_suffix = "t-SNE Reduced Features"
            plot_filename_suffix = "tsne"
        else:
            print(f"Invalid dimensionality reduction method: {method}. Skipping cluster plot for {model_name}.")
            return
        X_plot = pd.DataFrame(X_plot_reduced, columns=plot_feature_names, index=X.index)
    elif X.shape[1] < 2:
        print(f"Cannot visualize clusters for {model_name}: Insufficient features ({X.shape[1]}).")
        return

    # Predict clusters/classes
    try:
        y_pred = model.predict(X) # Model predicts on original features
    except Exception as e:
        print(f"Warning: Could not get predictions from model {model_name} for cluster visualization: {e}. Plotting true labels instead.")
        y_pred = y # Fallback to true labels if prediction fails

    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF']) # Consistent colormap

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_plot.iloc[:, 0], y=X_plot.iloc[:, 1], hue=y_pred,
                    palette=cmap_bold, alpha=0.8, edgecolor="black", s=80,
                    legend='full')
    
    plt.title(f'Clusters/Classes by {model_name} ({plot_title_suffix})')
    plt.xlabel(plot_feature_names[0])
    plt.ylabel(plot_feature_names[1])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Create custom legend handles for class names
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[i], 
                          markerfacecolor=cmap_bold(i), markersize=10) for i in range(len(class_names))]
    plt.legend(title="Predicted Classes", handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    figures_dir = os.path.join(Config.BASE_DIR, '..', 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, f'clusters_{model_name}_{plot_filename_suffix}.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved cluster visualization plot to {filepath}")


def plot_feature_importance(model, feature_names, model_name):
    """
    Plots feature importances for models that support it (e.g., tree-based models, linear models).
    """
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models, coef_ can be 1D or 2D (multi-class)
        if model.coef_.ndim > 1:
            # Take the absolute mean of coefficients across classes for multi-class
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            importances = np.abs(model.coef_)
    
    if importances is not None:
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
        plt.title(f'Feature Importance for {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()

        figures_dir = os.path.join(Config.BASE_DIR, '..', 'reports', 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        filepath = os.path.join(figures_dir, f'feature_importance_{model_name}.png')
        plt.savefig(filepath)
        plt.close()
    print(f"Saved feature importance plot to {filepath}")

def plot_error_analysis(X, y_true, y_pred, feature_names, label_encoder, model_name, method='pca'):
    """
    Visualizes misclassified samples, using dimensionality reduction if features > 2.
    
    Args:
        X (pd.DataFrame): Feature data.
        y_true (np.array): True target data (encoded).
        y_pred (np.array): Predicted target data (encoded).
        feature_names (list): Names of the features.
        label_encoder (LabelEncoder): Encoder used for target variable to get class names.
        model_name (str): Name of the model for plot title and filename.
        method (str): 'pca' or 'tsne' for dimensionality reduction if needed.
    """
    class_names = label_encoder.inverse_transform(np.unique(y_true))
    
    X_plot = X.copy()
    plot_feature_names = feature_names
    plot_title_suffix = "Original Features"
    plot_filename_suffix = "original_features"

    if X.shape[1] > 2:
        print(f"Features > 2. Applying {method.upper()} for error analysis visualization for {model_name}.")
        if method == 'pca':
            pca = PCA(n_components=2)
            X_plot_reduced = pca.fit_transform(X)
            plot_feature_names = ['Principal Component 1', 'Principal Component 2']
            plot_title_suffix = "PCA Reduced Features"
            plot_filename_suffix = "pca"
        elif method == 'tsne':
            tsne = TSNE(n_components=2, random_state=Config.RANDOM_STATE, perplexity=min(30, X.shape[0]-1))
            X_plot_reduced = tsne.fit_transform(X)
            plot_feature_names = ['t-SNE Component 1', 't-SNE Component 2']
            plot_title_suffix = "t-SNE Reduced Features"
            plot_filename_suffix = "tsne"
        else:
            print(f"Invalid dimensionality reduction method: {method}. Skipping error analysis plot for {model_name}.")
            return
        X_plot = pd.DataFrame(X_plot_reduced, columns=plot_feature_names, index=X.index)
    elif X.shape[1] < 2:
        print(f"Cannot perform error analysis visualization for {model_name}: Insufficient features ({X.shape[1]}).")
        return

    # Create a DataFrame for plotting
    plot_df = X_plot.copy()
    plot_df['true_label'] = y_true
    plot_df['predicted_label'] = y_pred
    plot_df['correct'] = (y_true == y_pred)
    
    # Map encoded labels back to original class names for better readability
    plot_df['true_label_name'] = label_encoder.inverse_transform(y_true)
    plot_df['predicted_label_name'] = label_encoder.inverse_transform(y_pred)

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=plot_df.iloc[:, 0], y=plot_df.iloc[:, 1],
                    hue='true_label_name', style='correct', markers={True: 'o', False: 'X'},
                    s=100, alpha=0.7, palette='viridis', data=plot_df)
    
    plt.title(f'Error Analysis for {model_name} ({plot_title_suffix})')
    plt.xlabel(plot_feature_names[0])
    plt.ylabel(plot_feature_names[1])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='True Label (X = Misclassified)')
    plt.tight_layout()

    figures_dir = os.path.join(Config.BASE_DIR, '..', 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, f'error_analysis_{model_name}_{plot_filename_suffix}.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved error analysis plot to {filepath}")
    
    

def plot_interaction_effects(df, feature1, feature2, target, filename):
    """
    Plots interaction effects between two features on the target variable.
    Uses seaborn.pointplot to show the mean target value for different categories
    of one feature across levels of another.
    """
    plt.figure(figsize=(12, 8))
    sns.pointplot(data=df, x=feature1, y=target, hue=feature2, dodge=True, palette='viridis', errorbar='sd')
    plt.title(f'Interaction Effect of {feature1} and {feature2} on {target}')
    plt.xlabel(feature1)
    plt.ylabel(f'Mean {target}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title=feature2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    figures_dir = os.path.join(Config.BASE_DIR, '..', 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved interaction plot to {filepath}")
        

def plot_polynomial_feature_distributions(X_poly_df, y, label_encoder):
    """
    Visualizes the distributions of polynomial features and their correlations with the target variable.
    """
    class_names = label_encoder.inverse_transform(np.unique(y))
    
    # Create a DataFrame for plotting
    plot_df = X_poly_df.copy()
    plot_df['category'] = label_encoder.inverse_transform(y)

    print("\n--- Visualizing Polynomial Feature Distributions ---")
    figures_dir = os.path.join(Config.BASE_DIR, '..', 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Histograms for polynomial features
    for col in X_poly_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=plot_df, x=col, hue='category', kde=True, palette='viridis', multiple='stack')
        plt.title(f'Distribution of Polynomial Feature: {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        filepath = os.path.join(figures_dir, f'hist_poly_feature_{col.replace(" ", "_").replace("^", "")}.png')
        plt.savefig(filepath)
        plt.close()
        print(f"Saved histogram for polynomial feature {col} to {filepath}")

    # Correlation Heatmap for polynomial features with target
    # First, convert target back to numeric for correlation calculation if not already
    numeric_y = y # y is already encoded as numeric
    corr_df = X_poly_df.copy()
    corr_df['target'] = numeric_y
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Polynomial Features and Target')
    filepath = os.path.join(figures_dir, 'correlation_heatmap_poly_features.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved correlation heatmap for polynomial features to {filepath}")
        
        

def run_eda():
    print("Running Exploratory Data Analysis...")

    # Load data
    train_df = pd.read_csv(Config.TRAIN_FILE)
    test_df = pd.read_csv(Config.TEST_FILE)

    print("\n--- Training Data Info ---")
    train_df.info()
    print("\n--- Training Data Description ---")
    print(train_df.describe())
    print("\n--- Training Data Missing Values ---")
    print(train_df.isnull().sum())
    print("\n--- Training Data 'category' distribution ---")
    print(train_df['category'].value_counts())

    print("\n--- Test Data Info ---")
    test_df.info()
    print("\n--- Test Data Description ---")
    print(test_df.describe())
    print("\n--- Test Data Missing Values ---")
    print(test_df.isnull().sum())

    # Create reports/figures directory if it doesn't exist
    figures_dir = os.path.join(Config.BASE_DIR, '..', 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Preprocess data to get polynomial features if enabled for EDA
    # Note: This is a separate preprocessing for EDA visualization, not for model training.
    # It allows us to visualize the generated polynomial features.
    from src.data_preprocessing import preprocess_data # Import here to avoid circular dependency

    X_eda_train, y_eda_encoded, _, label_encoder, _ = preprocess_data(
        train_df.drop('category', axis=1), 
        test_df, # dummy test_df, not used for EDA here
        use_polynomial_features=Config.USE_POLYNOMIAL_FEATURES,
        polynomial_degree=Config.POLYNOMIAL_DEGREE,
        use_additional_engineered_features=Config.USE_ADDITIONAL_ENGINEERED_FEATURES
    )

    # --- Visualizations ---

    # 1. Scatter plot of 'signal_strength' vs 'response_level' colored by 'category'
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=train_df, x='signal_strength', y='response_level', hue='category', palette='viridis', s=100, alpha=0.7)
    plt.title('Signal Strength vs Response Level by Category')
    plt.xlabel('Signal Strength')
    plt.ylabel('Response Level')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(figures_dir, 'scatter_signal_response_by_category.png'))
    plt.close()
    print(f"Saved scatter plot to {os.path.join(figures_dir, 'scatter_signal_response_by_category.png')}")

    # 2. Histograms/Density plots for 'signal_strength' and 'response_level' for each 'category'
    for feature in ['signal_strength', 'response_level']:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=train_df, x=feature, hue='category', kde=True, palette='viridis', multiple='stack')
        plt.title(f'Distribution of {feature} by Category')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.savefig(os.path.join(figures_dir, f'hist_{feature}_by_category.png'))
        plt.close()
        print(f"Saved histogram for {feature} to {os.path.join(figures_dir, f'hist_{feature}_by_category.png')}")

    # 3. Box plots for 'signal_strength' and 'response_level' for each 'category'
    for feature in ['signal_strength', 'response_level']:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=train_df, x='category', y=feature, hue='category', palette='viridis', legend=False)
        plt.title(f'Box Plot of {feature} by Category')
        plt.xlabel('Category')
        plt.ylabel(feature)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(figures_dir, f'boxplot_{feature}_by_category.png'))
        plt.close()
        print(f"Saved box plot for {feature} to {os.path.join(figures_dir, f'boxplot_{feature}_by_category.png')}")

    # 4. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(train_df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig(os.path.join(figures_dir, 'correlation_heatmap.png'))
    plt.close()
    print(f"Saved correlation heatmap to {os.path.join(figures_dir, 'correlation_heatmap.png')}")

    # 5. Pair Plot
    sns.pairplot(train_df.select_dtypes(include=['float64', 'int64', 'object']), hue='category', palette='viridis')
    plt.suptitle('Pair Plot of Numerical Features by Category', y=1.02)
    plt.savefig(os.path.join(figures_dir, 'pair_plot.png'))
    plt.close()
    print(f"Saved pair plot to {os.path.join(figures_dir, 'pair_plot.png')}")

    # 6. Violin Plots for 'signal_strength' and 'response_level' for each 'category'
    for feature in ['signal_strength', 'response_level']:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=train_df, x='category', y=feature, hue='category', palette='viridis', legend=False)
        plt.title(f'Violin Plot of {feature} by Category')
        plt.xlabel('Category')
        plt.ylabel(feature)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(figures_dir, f'violinplot_{feature}_by_category.png'))
        plt.close()
        print(f"Saved violin plot for {feature} to {os.path.join(figures_dir, f'violinplot_{feature}_by_category.png')}")

    # 7. Joint Plot / Hexbin Plot for 'signal_strength' vs 'response_level'
    # Using kdeplot for joint distribution and hist for marginals as a general approach
    # For a hexbin plot, `sns.jointplot(kind='hex')` can be used.
    g = sns.jointplot(data=train_df, x='signal_strength', y='response_level', hue='category', palette='viridis', kind='kde', height=8)
    g.fig.suptitle('Joint Plot of Signal Strength vs Response Level by Category', y=1.02)
    plt.savefig(os.path.join(figures_dir, 'jointplot_signal_response_by_category.png'))
    plt.close()
    print(f"Saved joint plot to {os.path.join(figures_dir, 'jointplot_signal_response_by_category.png')}")

    # Optionally, a hexbin plot
    plt.figure(figsize=(10, 8))
    sns.jointplot(data=train_df, x='signal_strength', y='response_level', kind='hex', height=8, cmap='viridis')
    plt.suptitle('Hexbin Plot of Signal Strength vs Response Level', y=1.02)
    plt.savefig(os.path.join(figures_dir, 'hexbin_signal_response.png'))
    plt.close()
    print(f"Saved hexbin plot to {os.path.join(figures_dir, 'hexbin_signal_response.png')}")

    # 8. Boxen Plots for Outlier Analysis
    for feature in ['signal_strength', 'response_level']:
        plt.figure(figsize=(10, 6))
        sns.boxenplot(data=train_df, x='category', y=feature, hue='category', palette='viridis', legend=False)
        plt.title(f'Boxen Plot of {feature} by Category (Outlier Analysis)')
        plt.xlabel('Category')
        plt.ylabel(feature)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(figures_dir, f'boxenplot_{feature}_by_category.png'))
        plt.close()
        print(f"Saved boxen plot for {feature} to {os.path.join(figures_dir, f'boxenplot_{feature}_by_category.png')}")

    # Plot polynomial feature distributions if enabled
    if Config.USE_POLYNOMIAL_FEATURES or Config.USE_ADDITIONAL_ENGINEERED_FEATURES:
        print("\n--- Plotting Distributions of Engineered Features ---")
        plot_polynomial_feature_distributions(X_eda_train, y_eda_encoded, label_encoder)


    print("\nEDA complete. Visualizations saved to reports/figures directory.")

if __name__ == '__main__':
    run_eda()
