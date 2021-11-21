# Obstetrics-Research-Project-
Predicting a specific medical state based on obstetrics and gestation data with gradient-boosting decision trees and other tree-based models.
Data and preprocessing code not included in repository due to information sensitivity.
Data preprocessed with outliers and missing values handling with feature-specific interpolation (by domain-expert advice).

The repository includes 3 scripts:
First data exploration is in the code 'data_exploration.py',
including exploratory data analysis (EDA) - descriptive statistics and features covariance matrix.  

Models initialization and training are in the 'main.py' code.
Data suffer from extreme label imbalance (from 3% to 7% of the classes are from the positive class).
Imbalance is handled with undersampling of the majority class and oversampling of the minority class (with SMOTE). 
Main training of 3 tree-based models (cat boost -gradient descent decision tree (GDBT), random forest, and simple decision tree). 
Hyper-parameters optimized with Optuna library for the maximal value of one of 3 scoring matrices: ROC AUC, recall, F1 score. (Future edition may include scoring based on higher value for Beta in the Fbeta scoring, emphasizes misclassification of the positive samples).

Trained models scores and feature importance is visualized in the 'model_inspect.py' script. 
Plots generated: ROC, confusion matrices, shap values, tree visualization (for non-ensemble classifier)
