# Customer Churn Prediction

## Abstract
This project aims to predict customer churn in a telecom company using machine learning models. We preprocess the data, perform exploratory data analysis (EDA), and develop multiple models to identify the best predictor of churn. The Logistic Regression model achieved the highest accuracy of 82%.

## Introduction
Customer churn is a critical issue for telecom companies as it directly impacts revenue. Predicting churn can help companies take proactive measures to retain customers. This project aims to build a model to predict churn and identify key factors contributing to customer churn.

## Data Description
The dataset used is the Telco Customer Churn dataset from Kaggle, which contains information on customer demographics, services availed, and account information. The target variable is `Churn`, indicating whether a customer has left the service.

## Data Preprocessing
### Handling Missing Values
The `TotalCharges` column had missing values which were filled using the median value of the column.

### Encoding Categorical Variables
Categorical variables such as `gender`, `InternetService`, etc., were encoded using one-hot encoding to convert them into numerical format.

### Scaling Features
Numerical features like `tenure`, `MonthlyCharges`, and `TotalCharges` were scaled using StandardScaler for better model performance.

## Exploratory Data Analysis (EDA)
### Distribution of Features
The numerical features `tenure`, `MonthlyCharges`, and `TotalCharges` showed a varied distribution which was visualized using histograms and box plots.

### Correlation Matrix
A heatmap was used to visualize the correlations between different features, highlighting any multicollinearity.

### Target Variable Analysis
The target variable `Churn` was analyzed using a count plot, showing the distribution of customers who churned versus those who did not.

### Feature-Target Relationships
Bar plots and box plots were used to explore the relationships between categorical features and the target variable.

## Model Development
### Data Splitting
The data was split into training (80%) and testing (20%) sets to evaluate model performance.

### Model Selection
Three models were selected for this project:
- Logistic Regression
- Multi-Layer Perceptron
- Random Forest

### Training Models
Each model was trained on the training data using default parameters initially.

## Model Evaluation
### Evaluation Metrics
The models were evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Results
The performance of each model was as follows:

| Model              | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 0.82     | 0.69      | 0.60   | 0.64     | 0.75    |
| MLP                | 0.81     | 0.69      | 0.53   | 0.60     | 0.72    |
| Random Forest      | 0.79     | 0.65      | 0.46   | 0.54     | 0.68    |

### Best Model
The Logistic Regresion model performed the best with an accuracy of 82% and an ROC-AUC of 0.75. 

## Conclusion and Future Work
### Summary
The project built a model to predict customer churn with the Logistic Regression model achieving the best performance.

### Insights
Feature importance analysis revealed that `Contract`, `tenure`, and `MonthlyCharges` were significant predictors of churn.

### Future Work
Future improvements must include fixing the imbalance within this dataset, exploring additional features, and deploying the model as a web application.

## References
- Telco Customer Churn Dataset: [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
