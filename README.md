# Obstetrics-Research-Project-
Predicting a specific medical state (emergency cesarean surgery) based on obstetrics and gestation data with logistic regression, gradient-boosting decision trees, and other tree-based models.
Data is not included in the repository due to information sensitivity, and also some of the names of the features were omitted.
Data preprocessed with outliers and missing values handling with feature-specific interpolation (by domain-expert advice).

The repository includes 3 scripts:
First data exploration is in the code 'data_exploration.py',
including exploratory data analysis (EDA) - descriptive statistics and features covariance matrix.  

Dataset preparation is done in the 'dateset_preperation.py'. Includes missing values investigating, excluding medical irrelevant cases (not the study population or cases with a specific condition known to cause a high risk to surgery, therefore have no medical benefits for being used for prediction)

Models initialization and training are in the 'main.py' code.
Data suffer from extreme label imbalance (from 3% to 7% of the classes are from the positive class).
Imbalance is handled with undersampling of the majority class and oversampling of the minority class (with SMOTE). 
Main training of logistic regression and 3 tree-based models (cat boost -gradient descent decision tree (GDBT), random forest, and simple decision tree). 
Hyper-parameters optimized with Optuna library for the maximal value of one of 3 scoring matrices: ROC AUC, recall, F1 score. 
Plots generated: ROC, confusion matrices, shap values, tree visualization (for non-ensemble classifier)

Few example figures from the projects: AUCs of the models, the 'fetus weight' U curve risk effect (which is reasonable- a very small or very big fetuses are correlated with a higher chance of surgery), and the interaction between the fetus weight and use of epidural analgesia (which experts found to be surprising).

![Figure 2021-12-29 105122 (35)](https://user-images.githubusercontent.com/62498821/152750053-8987772c-6213-40db-a0ed-e8a50672f0e5.png)
![FETUS AND EPIDURAL](https://user-images.githubusercontent.com/62498821/152753030-50d60109-a52e-4783-8802-fb69325beee5.jpg)
