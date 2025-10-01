# =============================================================================
# Task 5: Decision Trees and Random Forests
# =============================================================================

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. Load and Prepare the Dataset ---
print("--- 0. Loading and Preparing Data ---")
# Load the dataset
# Make sure 'heart.csv' is in the same directory as this script.
try:
    data = pd.read_csv('heart.csv')
    print("Dataset loaded successfully.")
    print("Dataset shape:", data.shape)
    print("First 5 rows:\n", data.head())
except FileNotFoundError:
    print("Error: 'heart.csv' not found. Please download the dataset and place it in the correct directory.")
    exit()

# Separate features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")


# =============================================================================
# 1. Train a Decision Tree Classifier and Visualize It
# =============================================================================
print("\n--- 1. Training and Visualizing a Decision Tree ---")

# Initialize the Decision Tree Classifier
# We'll use max_depth=3 for a simpler, more interpretable tree visualization.
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)

# Train the model
dt_classifier.fit(X_train, y_train)

print("Decision Tree with max_depth=3 has been trained.")

# Visualize the tree
dot_data = export_graphviz(dt_classifier,
                           out_file=None,
                           feature_names=X.columns,
                           class_names=['No Disease', 'Has Disease'],
                           filled=True,
                           rounded=True,
                           special_characters=True)

graph = graphviz.Source(dot_data)
# This will save a file 'decision_tree.pdf' and try to render it.
graph.render("decision_tree")

print("Decision tree visualization saved as 'decision_tree.pdf'.")


# =============================================================================
# 2. Analyze Overfitting and Control Tree Depth
# =============================================================================
print("\n--- 2. Analyzing Overfitting by Comparing Tree Depths ---")

# a) Train a full-depth (overfit) tree by not setting max_depth
dt_overfit = DecisionTreeClassifier(random_state=42)
dt_overfit.fit(X_train, y_train)

# b) Use the pruned tree from step 1 (max_depth=3)
dt_pruned = dt_classifier # Already trained

# Evaluate both models on training and testing data
acc_overfit_train = accuracy_score(y_train, dt_overfit.predict(X_train))
acc_overfit_test = accuracy_score(y_test, dt_overfit.predict(X_test))

acc_pruned_train = accuracy_score(y_train, dt_pruned.predict(X_train))
acc_pruned_test = accuracy_score(y_test, dt_pruned.predict(X_test))

print(f"Full-Depth Tree (Overfit) | Train Accuracy: {acc_overfit_train:.4f}, Test Accuracy: {acc_overfit_test:.4f}")
print(f"Pruned Tree (max_depth=3) | Train Accuracy: {acc_pruned_train:.4f}, Test Accuracy: {acc_pruned_test:.4f}")
print("\nNotice the large gap between train and test accuracy for the full-depth tree.")
print("This indicates overfitting. The pruned tree generalizes better to unseen data.")


# =============================================================================
# 3. Train a Random Forest and Compare Accuracy
# =============================================================================
print("\n--- 3. Training a Random Forest Classifier ---")

# Initialize the Random Forest Classifier
# n_estimators is the number of trees in the forest.
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)
print("Random Forest with 100 estimators has been trained.")

# Evaluate the Random Forest
y_pred_rf = rf_classifier.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"\nAccuracy Comparison on Test Data:")
print(f"Pruned Decision Tree: {acc_pruned_test:.4f}")
print(f"Random Forest:        {acc_rf:.4f}")
print("\nThe Random Forest model typically provides better accuracy as it combines multiple trees.")


# =============================================================================
# 4. Interpret Feature Importances
# =============================================================================
print("\n--- 4. Interpreting Feature Importances from Random Forest ---")

# Get feature importances from the trained Random Forest model
importances = rf_classifier.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Top 5 most important features:")
print(feature_importance_df.head())

# Plotting the feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importances.png')
print("\nFeature importance plot saved as 'feature_importances.png'.")


# =============================================================================
# 5. Evaluate Using Cross-Validation
# =============================================================================
print("\n--- 5. Evaluating Models with 5-Fold Cross-Validation ---")

# Use the entire dataset (X, y) for cross-validation for a more robust evaluation

# Cross-validation for the pruned Decision Tree
cv_scores_dt = cross_val_score(dt_pruned, X, y, cv=5)

# Cross-validation for the Random Forest
cv_scores_rf = cross_val_score(rf_classifier, X, y, cv=5)

print(f"Pruned Decision Tree CV Mean Accuracy: {np.mean(cv_scores_dt):.4f} (+/- {np.std(cv_scores_dt):.4f})")
print(f"Random Forest CV Mean Accuracy:        {np.mean(cv_scores_rf):.4f} (+/- {np.std(cv_scores_rf):.4f})")
print("\nCross-validation provides a more reliable measure of model performance.")

print("\n\n===== Task Complete =====")