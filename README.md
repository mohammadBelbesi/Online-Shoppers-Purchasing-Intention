# Online Shopper Revenue Prediction

## Project Overview
This project aims to predict whether an online shopper will generate revenue based on several features such as visitor behavior, site metrics, and user demographics. Using various machine learning algorithms, we built and evaluated predictive models that classify user behavior as either leading to revenue generation or not.

## Goals of the Project
- Run and compare multiple machine learning algorithms to determine their effectiveness on the dataset.
- Use cross-validation and random sampling to assess model robustness.
- Implement **AdaBoost** to further enhance model performance.
- Identify the most important features that influence user behavior and model accuracy.

## Dataset
We used the **"Online Shoppers Purchasing Intention"** dataset, which contains 12,330 samples with 18 features, both numerical and categorical, and a binary target variable `Revenue`, which indicates whether a transaction occurred (`1` for a purchase, `0` otherwise).

### Key Features
- **Numerical Features**: `Administrative`, `Informational`, `ProductRelated`, `BounceRates`, `ExitRates`, `PageValues`.
- **Categorical Features**: `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`.

### Preprocessing
- **Standardization**: Numerical features were standardized using `StandardScaler` for consistency.
- **One-Hot Encoding**: Categorical features were one-hot encoded to make them compatible with machine learning models.
- **Handling Missing Values**: Missing values were imputed using the mean for numerical features and the most frequent value for categorical features.

## Methodology
The following base models were implemented and evaluated:

- **Decision Tree Classifier**: A tree-based model used for classification tasks.
- **Logistic Regression**: A widely used linear model for binary classification.
- **K-Nearest Neighbors (KNN)**: A non-parametric method for classification.
- **AdaBoost Classifier**: A boosting algorithm that enhances weak classifiers using `DecisionTreeClassifier` as the base estimator.

### Evaluation Metrics
- **Accuracy**: Proportion of correctly classified instances.
- **Precision**: Accuracy of positive predictions.
- **Recall**: Ability of the model to identify positive instances.
- **F1 Score**: Weighted average of precision and recall.
- **AUC (Area Under the Curve)**: Measures the model’s ability to distinguish between classes.
- **Confusion Matrix**: Summarizes the number of correct and incorrect predictions.

### Model Evaluation Approaches
- **Cross-validation (5-fold)**: The dataset was split into 5 equal parts, training the model on 80% of the data and testing on the remaining 20%, rotating for each iteration.
- **Random Sampling (50%)**: Training data was randomly reduced to 50% for five iterations, ensuring no overlap between samples.
- **AdaBoost Implementation**: The boosting technique was applied five times using a decision tree as the base learner.

## Results
### Cross-Validation Results
- AdaBoost consistently outperformed other models in terms of accuracy and F1 score, making it the best-performing model for this dataset.

### Random Sampling Results
- Models were evaluated by randomly sampling 50% of the training data five times. The results remained relatively stable across subsets.

### Feature Importance
The **AdaBoost** classifier highlighted the following top features:
1. `PageValues`
2. `ProductRelated_Duration`
3. `ExitRates`
4. `BounceRates`

These features directly influence user behavior and are key indicators of whether a transaction will occur.

## Key Insights
- AdaBoost may not significantly improve performance when the base learner is already strong, and it could even lead to overfitting due to added complexity.
- The feature importance analysis showed that user behavior metrics (`PageValues`, `ProductRelated_Duration`, `ExitRates`, and `BounceRates`) are crucial in predicting revenue.
- Cross-validation demonstrated that the models are robust and perform consistently across different folds.
- Random sampling showed minor variations in results, indicating that the model’s performance is relatively stable even with a smaller dataset.

## Future Work
- Explore advanced models like **Gradient Boosting** or **XGBoost** for improved performance.
- Perform hyperparameter tuning to optimize the models further.
