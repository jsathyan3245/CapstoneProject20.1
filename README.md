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

## Conclusion
This project successfully demonstrated the application of machine learning techniques, including Random Forest, Logistic Regression, AdaBoost, Gradient Boosting, and SVM, to predict customer churn. 

- **Best Model:** Logistic Regression was the most accurate and performed best in terms of F1-score and recall. This model is recommended for predicting customer churn in this dataset.
- **Model Comparison:** While Logistic Regression outperformed the other models overall, Random Forest, AdaBoost, and Gradient Boosting showed promising results and could be further improved through hyperparameter tuning and feature engineering.
- **Feature Insights:** TotalCharges, MonthlyCharges, and Tenure are key predictors of churn, as identified by the Random Forest model. These features should be considered for targeted interventions aimed at reducing churn.




### Deliverables
- **Jupyter Notebook**: [CustomerChurnCapstoneFinal](https://github.com/jsathyan3245/CapstoneProject20.1/blob/main/CustomerChurnCapstone.ipynb)


