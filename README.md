# Customer Churn Prediction with Multiple Classifiers

## Overview
This project focuses on predicting customer churn for a telecom company using a variety of machine learning models. The dataset includes customer demographics and account information. The goal is to predict whether a customer will churn (leave) based on various features, such as tenure, monthly charges, contract type, and more.

## Dataset Overview
The dataset contains the following key columns:

- `customerID`: Unique customer identifier.
- `Churn`: Target variable indicating whether the customer churned (1) or not (0).
- `Features`: Demographic and account-related attributes like gender, tenure, monthly charges, total charges, contract type, and more.

## Installation Requirements
To run this project, the following Python libraries are required:

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- imblearn
- scipy

## Data Processing
### Load and Inspect Data:
- Load the dataset and inspect basic information such as missing values and descriptive statistics.

### Handle Missing Data:
- Convert non-numeric values in the `TotalCharges` column to numeric and fill missing values with 0.

### Encode Categorical Variables:
- Apply label encoding to all categorical columns except `customerID` and `Churn`.

## Exploratory Data Analysis (EDA)
### Churn Distribution:
- Visualize the distribution of churned vs. non-churned customers using a count plot.

### Monthly Charges vs. Churn:
- Create a boxplot to visualize how monthly charges vary with churn status.

### Correlation Matrix:
- Generate a heatmap to display correlations between numerical features.

### Gender and Churn:
- Use a pie chart to visualize the proportion of churn by gender.

### Senior Citizen Status and Churn:
- Create a pie chart to show churn proportions for senior citizens vs. non-senior citizens.

### Churn Count by Contract Type:
- Use a grouped bar chart to visualize churn counts by contract type.

### Chi-Square Test:
- Conduct a chi-square test between `Churn` and `Contract Type` to assess statistical significance.

## Feature Scaling and Model Training
### Feature Scaling:
- Apply `StandardScaler` to numerical features such as tenure, `MonthlyCharges`, `TotalCharges`.

### Train-Test Split:
- Split the dataset into training (80%) and testing (20%) sets.

### Balancing the Dataset:
- Use SMOTE to oversample the minority class to balance the dataset.

## Model Training and Evaluation
### Decision Tree, SVM, KNN, and Naive Bayes:
- Train each model on the balanced and scaled dataset.
- Evaluate models based on accuracy, F1-score, recall, precision, and confusion matrix.

### Random Forest Classifier with Hyperparameter Tuning:
- Perform a `GridSearchCV` to tune hyperparameters such as `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- Display the best hyperparameters and evaluate model performance.

### Feature Selection with RFE:
- Use Recursive Feature Elimination (RFE) to select the most important features and retrain the Random Forest model.

### AdaBoost Classifier with Hyperparameter Tuning:
- Perform `GridSearchCV` to optimize the number of estimators and learning rate.
- Evaluate the tuned model on the test set.

### Gradient Boosting Classifier with Hyperparameter Tuning:
- Perform `GridSearchCV` to tune hyperparameters such as `n_estimators`, `learning_rate`, and `max_depth`.
- Evaluate the tuned model on the test set.

### Voting Classifier:
- Combine the best models (Random Forest, AdaBoost, and Gradient Boosting) using a soft voting classifier to improve prediction performance.

## Model Evaluation
### Confusion Matrix:
- Visualize the confusion matrix for each model to understand true positives, false positives, true negatives, and false negatives.

### Feature Importance Plot:
- For Random Forest and Gradient Boosting, plot the feature importance to highlight the most important predictors of churn.

### ROC-AUC Curve:
- Plot the ROC-AUC curve for the Random Forest, AdaBoost, and Gradient Boosting models to assess their ability to distinguish between churned and non-churned customers.

## Performance Evaluation
### Model Performance Comparison:

| Model              | Accuracy | F1-Score | Recall | Precision |
|--------------------|----------|----------|--------|-----------|
| Logistic Regression| 0.8169   | 0.6272   | 0.5818 | 0.6803    |
| Random Forest      | 0.7956   | 0.5500   | 0.4719 | 0.6592    |
| AdaBoost           | 0.8124   | 0.6014   | 0.5662 | 0.6541    |
| Gradient Boosting  | 0.8137   | 0.5998   | 0.5664 | 0.6575    |

### Statistical Analysis
#### Chi-Square Test (Churn vs. Contract Type):
- Chi-Square Statistic: 1184.60
- Degrees of Freedom: 2
- p-value: 0.0000 (significant association)

#### Feature Importance (Random Forest):
- Top Features:
  - `TotalCharges`: 0.1834
  - `MonthlyCharges`: 0.1623
  - `Tenure`: 0.1436
- Other Notable Features:
  - `Contract_0`: 0.0623
  - `OnlineSecurity`: 0.0431
  - `TechSupport`: 0.0376

## Key Insights
### Model Performance:
- Logistic Regression performs better in terms of accuracy, F1-score, and recall for predicting churn.

### Feature Importance:
- `TotalCharges`, `MonthlyCharges`, and `Tenure` are the most critical predictors of churn in the Random Forest model.

### Chi-Square Test:
- A significant statistical association exists between `Contract Type` and churn, making it a strong predictor.
  
---
# Churn Prediction Model Evaluation

## Overview
This repository evaluates various machine learning models on a classification dataset to predict customer churn. The models include Decision Tree, SVM, KNN, Naive Bayes, Random Forest, AdaBoost, and Gradient Boosting. The analysis focuses on identifying the best-performing model based on accuracy, precision, recall, F1-score, and ROC-AUC.

---

## Results and Model Evaluation

### Selected Features for Prediction
- `tenure`, `OnlineSecurity`, `TechSupport`, `Contract`, `MonthlyCharges`

### Model Performance Summary
| Model              | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|--------------------|----------|----------|-----------|--------|---------|
| Random Forest      | 0.8469   | 0.8543   | 0.8250    | 0.8856 | 0.9235  |
| AdaBoost           | 0.8097   | 0.8220   | 0.7811    | 0.8675 | 0.8935  |
| Gradient Boosting  | 0.8444   | 0.8484   | 0.8408    | 0.8561 | 0.9281  |


### Decision Tree
- **Training Accuracy**: 0.9987  
- **Test Accuracy**: 0.8005  
- **Classification Report**:  
  | Metric      | Class 0 | Class 1 | Macro Avg | Weighted Avg |
  |-------------|---------|---------|-----------|--------------|
  | Precision   | 0.81    | 0.79    | 0.80      | 0.80         |
  | Recall      | 0.78    | 0.82    | 0.80      | 0.80         |
  | F1-Score    | 0.79    | 0.81    | 0.80      | 0.80         |

### Support Vector Machine (SVM)
- **Training Accuracy**: 0.8492  
- **Test Accuracy**: 0.8333  
- **Classification Report**:  
  | Metric      | Class 0 | Class 1 | Macro Avg | Weighted Avg |
  |-------------|---------|---------|-----------|--------------|
  | Precision   | 0.83    | 0.84    | 0.83      | 0.83         |
  | Recall      | 0.83    | 0.84    | 0.83      | 0.83         |
  | F1-Score    | 0.83    | 0.84    | 0.83      | 0.83         |

### K-Nearest Neighbors (KNN)
- **Training Accuracy**: 0.8595  
- **Test Accuracy**: 0.7976  
- **Classification Report**:  
  | Metric      | Class 0 | Class 1 | Macro Avg | Weighted Avg |
  |-------------|---------|---------|-----------|--------------|
  | Precision   | 0.84    | 0.77    | 0.80      | 0.80         |
  | Recall      | 0.73    | 0.86    | 0.80      | 0.80         |
  | F1-Score    | 0.78    | 0.81    | 0.80      | 0.80         |

### Naive Bayes
- **Training Accuracy**: 0.7838  
- **Test Accuracy**: 0.7870  
- **Classification Report**:  
  | Metric      | Class 0 | Class 1 | Macro Avg | Weighted Avg |
  |-------------|---------|---------|-----------|--------------|
  | Precision   | 0.81    | 0.77    | 0.79      | 0.79         |
  | Recall      | 0.75    | 0.82    | 0.79      | 0.79         |
  | F1-Score    | 0.78    | 0.80    | 0.79      | 0.79         |

### Random Forest
- **Best Parameters**: `{'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}`
- **Training Accuracy**: 0.8948  
- **Test Accuracy**: 0.8469  
- **Classification Report**:  
  | Metric      | Class 0 | Class 1 | Macro Avg | Weighted Avg |
  |-------------|---------|---------|-----------|--------------|
  | Precision   | 0.87    | 0.83    | 0.85      | 0.85         |
  | Recall      | 0.81    | 0.89    | 0.85      | 0.85         |
  | F1-Score    | 0.84    | 0.85    | 0.85      | 0.85         |

### AdaBoost
- **Training Accuracy**: 0.7993  
- **Test Accuracy**: 0.8097  
- **Classification Report**:  
  | Metric      | Class 0 | Class 1 | Macro Avg | Weighted Avg |
  |-------------|---------|---------|-----------|--------------|
  | Precision   | 0.85    | 0.78    | 0.81      | 0.81         |
  | Recall      | 0.75    | 0.87    | 0.81      | 0.81         |
  | F1-Score    | 0.80    | 0.82    | 0.81      | 0.81         |

### Gradient Boosting
- **Training Accuracy**: 0.8924  
- **Test Accuracy**: 0.8444  
- **Classification Report**:  
  | Metric      | Class 0 | Class 1 | Macro Avg | Weighted Avg |
  |-------------|---------|---------|-----------|--------------|
  | Precision   | 0.85    | 0.84    | 0.84      | 0.84         |
  | Recall      | 0.83    | 0.86    | 0.84      | 0.84         |
  | F1-Score    | 0.84    | 0.85    | 0.84      | 0.84         |

---

## Final Conclusion
- **Random Forest** is the most robust model, achieving the highest accuracy (0.847), balanced precision-recall, and a superior ROC-AUC (0.9235).  
- **AdaBoost** and **Gradient Boosting** also performed well, but slightly below Random Forest in generalization ability.  
- **Decision Tree** exhibited overfitting with high training accuracy but lower test accuracy.  
- SVM, KNN, and Naive Bayes demonstrated acceptable performance but were outperformed by ensemble models.

----

### Deliverables
- **Jupyter Notebook**: [CustomerChurnCapstoneFinal](https://github.com/jsathyan3245/CapstoneProject20.1/blob/main/CustomerChurnCapstone.ipynb)


