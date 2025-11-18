# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np # Added
from matplotlib.colors import ListedColormap # Added


from src.config import Config

def visualize_decision_boundary(model, X, y, feature_names, class_names, title, filename):
    """
    Visualizes the decision boundary of a classifier in a 2D feature space.
    Assumes X has exactly two features.
    """
    if X.shape[1] != 2:
        print(f"Warning: Decision boundary visualization requires exactly 2 features. Found {X.shape[1]}. Skipping.")
        return

    # Create color maps for the plot
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Determine mesh grid
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

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
    plt.xticks(())
    plt.yticks(())
    
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

    print("\nEDA complete. Visualizations saved to reports/figures directory.")

if __name__ == '__main__':
    run_eda()
