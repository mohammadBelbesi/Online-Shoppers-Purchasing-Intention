import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Load the dataset
file_path = './online_shoppers_intention.csv'
data = pd.read_csv(file_path)

# Define feature types
numerical_features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                      'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
                      'SpecialDay']
categorical_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']

# Define target
target = 'Revenue'

# Step 1: Data Preprocessing
# Preprocess numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 2: Model Selection
models = {
    'DecisionTree': DecisionTreeClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'AdaBoost': AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, algorithm='SAMME')
}

# Step 3: Train/Test Split
X = data.drop(target, axis=1)
y = data[target]

# Splitting the dataset into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function for evaluating the models
def evaluate_model(clf, model_name):
    # Train the model on the full training set
    clf.fit(X_train, y_train)
    
    # Evaluate on the test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc,
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }

# Step 4: Cross-validation (5-fold, 80% training data) - Requirement 1
results = {}
for model_name, model in models.items():
    # Create pipeline with preprocessing and model
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    # Cross-validation (5-fold)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validated accuracy for {model_name}: {scores.mean():.4f}")
    
    # Evaluate the model and store results
    results[model_name] = evaluate_model(clf, model_name)

# Step 5: Random Sampling (50% of the data) - Requirement 2
random_results = {}
for model_name, model in models.items():
    model_scores = []
    for i in range(5):
        # Randomly sample 50% of the training data
        X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=i)
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        
        # Fit and evaluate the model
        clf.fit(X_train_sample, y_train_sample)
        model_scores.append(evaluate_model(clf, model_name))
    
    # Store the scores for random sampling
    random_results[model_name] = model_scores

# Step 6: AdaBoost (5 Models) - Requirement 3
adaboost_results = []
for i in range(5):
    # Train AdaBoost on the full training set, repeat 5 times
    ada_boost_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, algorithm='SAMME')
    ada_boost_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', ada_boost_model)])
    
    ada_boost_pipeline.fit(X_train, y_train)
    adaboost_results.append(evaluate_model(ada_boost_pipeline, "AdaBoost"))

# Step 7: Feature Importance for AdaBoost
def plot_feature_importance(model, X_train, model_name):
    # Get feature names after transformation
    feature_names = numerical_features + list(preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_features))
    
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        # Get feature importances from models that support it
        importance = model.named_steps['classifier'].feature_importances_
    else:
        # For models that do not support feature_importances_, use permutation importance
        result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)
        importance = result.importances_mean
    
    # Ensure the number of importances matches the number of features
    if len(importance) != len(feature_names):
        print(f"Warning: Number of importances ({len(importance)}) does not match the number of features ({len(feature_names)}).")
    
    # Create a DataFrame for feature importances and plot
    feature_importances = pd.Series(importance, index=feature_names)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title(f'Feature Importance for {model_name}')
    plt.show()

# AdaBoost Pipeline
ada_boost_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, algorithm='SAMME'))])
ada_boost_pipeline.fit(X_train, y_train)

# Plot the feature importances
plot_feature_importance(ada_boost_pipeline, X_train, "AdaBoost")

# Cross-Validation Results (this is fine)
results_df = pd.DataFrame(results).T

# Random Sampling (50% of the data) Results
# Convert each model's random sampling results into a DataFrame
random_results_dfs = {}
for model_name, model_scores in random_results.items():
    random_results_dfs[model_name] = pd.DataFrame(model_scores)

# AdaBoost Results (this is fine)
adaboost_results_df = pd.DataFrame(adaboost_results)

# Display Results
print("Cross-Validation Results:")
print(results_df)

print("\nRandom 50% Sampling Results (each model has a separate DataFrame):")
for model_name, df in random_results_dfs.items():
    print(f"\nRandom Sampling Results for {model_name}:")
    print(df)

print("\nAdaBoost Results:")
print(adaboost_results_df)