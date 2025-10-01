Decision Trees and Random Forests for Heart Disease Prediction
This project provides a complete Python script to explore, train, and evaluate tree-based machine learning models for classification. Using the popular Heart Disease dataset, it demonstrates the entire workflow from training a single Decision Tree to evaluating a more robust Random Forest model.

Project Workflow
The script decision_trees_and_forests.py performs the following steps:

Data Loading & Preparation: Loads the heart.csv dataset and splits it into training and testing sets.

Decision Tree Training & Visualization: Trains a Decision Tree classifier with a limited depth and visualizes the resulting tree structure using Graphviz, saving it as decision_tree.pdf.

Overfitting Analysis: Compares a full-depth (unconstrained) Decision Tree with a pruned (depth-limited) tree to demonstrate how controlling complexity can prevent overfitting.

Random Forest Training: Trains a Random Forest classifier and compares its accuracy against the single Decision Tree.

Feature Importance: Extracts and plots the feature importances from the Random Forest model to understand which factors are most influential in predicting heart disease. The plot is saved as feature_importances.png.

Cross-Validation: Provides a robust evaluation of both the Decision Tree and Random Forest models using 5-fold cross-validation.

