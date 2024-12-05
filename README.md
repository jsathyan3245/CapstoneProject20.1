# Customer Churn Prediction with Random Forest

## Overview
This project focuses on predicting customer churn for a telecom company using a dataset that includes customer demographics and account information. The goal is to predict whether a customer will churn (leave) based on various features, such as tenure, monthly charges, contract type, and more.

---

## Dataset Overview
The dataset includes the following key columns:
- **customerID**: Unique customer identifier.
- **Churn**: Target variable indicating whether the customer churned (1) or not (0).
- **Features**: Demographic and account-related attributes like gender, tenure, monthly charges, total charges, contract type, and more.

---

## Installation Requirements
To run this project, the following Python libraries are needed:
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `scipy`


---

## Data Processing
1. **Load and Inspect Data**: Load the dataset and inspect basic information such as missing values and descriptive statistics.
2. **Handle Missing Data**: 
   - Fill missing values in the `TotalCharges` column with 0.
   - Convert any non-numeric values to numeric.
3. **Encode Categorical Variables**: 
   - Apply label encoding to all categorical columns except `customerID` and `Churn`.

---

## Exploratory Data Analysis (EDA)
1. **Churn Distribution**: 
   - Use a count plot to visualize the distribution of churned vs. non-churned customers.
2. **Monthly Charges vs. Churn**: 
   - Create a boxplot to visualize how monthly charges vary with churn status.
3. **Correlation Matrix**: 
   - Generate a heatmap to display correlations between numerical features.
4. **Gender and Churn**: 
   - Use a pie chart to visualize the proportion of churn by gender.
5. **Senior Citizen Status and Churn**: 
   - Create a pie chart to show churn proportions for senior citizens vs. non-senior citizens.
6. **Churn Count by Contract Type**: 
   - Use a grouped bar chart to visualize churn counts by contract type.
7. **Chi-Square Test**: 
   - Conduct a chi-square test between `Churn` and `Contract Type` to assess statistical significance.

---

## Feature Scaling and Model Training
1. **Feature Scaling**: 
   - Apply `StandardScaler` to numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`).
2. **Train-Test Split**: 
   - Split the dataset into training (80%) and testing (20%) sets.
3. **Random Forest Classifier**: 
   - Train the model on the training set and evaluate it on the test set.
---
## Model Evaluation
1. **Confusion Matrix**: 
   - Visualize the performance of the Random Forest model by displaying the true positives, true negatives, false positives, and false negatives.
2. **Feature Importance**: 
   - Display the importance of each feature in predicting churn using the Random Forest model.
3. **ROC-AUC Curve**: 
   - Plot the ROC curve to assess the model's ability to distinguish between churned and non-churned customers.
4. **Calibration Curve**: 
   - Plot the calibration curve to evaluate how well the predicted probabilities align with the true outcomes.
---

## Performance Evaluation

### Model Performance Comparison

| Model                | Accuracy | F1-Score | Recall  | Precision |
|----------------------|----------|----------|---------|-----------|
| Logistic Regression  | 0.8169   | 0.6272   | 0.5818  | 0.6803    |
| Random Forest        | 0.7956   | 0.5500   | 0.4719  | 0.6592    |

### Logistic Regression Detailed Metrics
- **Accuracy**: 0.8211
- **Precision**: 0.69 (Class 1), 0.86 (Class 0)
- **Recall**: 0.60 (Class 1), 0.90 (Class 0)
- **F1-Score**: 0.64 (Class 1), 0.88 (Class 0)
- **ROC-AUC**: 0.8622

### Random Forest Detailed Metrics
- **Accuracy**: 0.7949
- **Precision**: 0.65 (Class 1), 0.83 (Class 0)
- **Recall**: 0.49 (Class 1), 0.90 (Class 0)
- **F1-Score**: 0.56 (Class 1), 0.87 (Class 0)
- **Macro Average F1-Score**: 0.71
- **Weighted Average F1-Score**: 0.78

---

### Statistical Analysis

#### Chi-Square Test (Churn vs. Contract Type)
- **Chi-Square Statistic**: 1184.60
- **Degrees of Freedom**: 2
- **p-value**: 0.0000 (significant association)

#### Feature Importance (Random Forest)
- **Top Features**:
  - TotalCharges: 0.1834
  - MonthlyCharges: 0.1623
  - Tenure: 0.1436
- **Other Notable Features**:
  - Contract_0: 0.0623
  - OnlineSecurity: 0.0431
  - TechSupport: 0.0376

---

### Key Insights
1. **Model Performance**: 
   - Logistic Regression outperforms Random Forest in accuracy, F1-score, and recall for predicting churn.
2. **Feature Importance**: 
   - TotalCharges, MonthlyCharges, and Tenure are the most critical predictors of churn in the Random Forest model.
3. **Chi-Square Test**: 
   - A significant statistical association exists between Contract Type and churn, making it a strong predictor.

---

### Conclusion
This project successfully demonstrated the application of machine learning techniques, including Random Forest and Logistic Regression, to predict customer churn. While Logistic Regression performed better overall, further tuning and feature engineering may enhance the Random Forest model's predictive accuracy.

---

### Deliverables
- **Jupyter Notebook**: [CapstoneFinal](https://github.com/jsathyan3245/CapstoneProject20.1/blob/main/CapstoneFinalVersion.ipynb)


