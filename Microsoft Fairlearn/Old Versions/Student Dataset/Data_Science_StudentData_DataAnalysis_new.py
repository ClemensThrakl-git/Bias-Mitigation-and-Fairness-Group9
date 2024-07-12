# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the cleaned dataset
data = pd.read_csv('cleaned_student_data2.csv')

# Check the distribution of the target variable G3 before binarization
print("Distribution of G3 before binarization:")
print(data['G3'].value_counts())

# Define the target variable
target = 'G3'

# Using the median of G3 as the threshold for binarization
median_G3 = data[target].median()
print(f"Median of G3: {median_G3}")

# Binarize the target variable: Above median (Pass) or Below median (Fail)
y = np.where(data[target] >= median_G3, 1, 0)

# Check the distribution of classes after binarization
print(f"Distribution of target variable after binarization: {np.bincount(y)}")

# Split the dataset into training and testing sets
X = data.drop(columns=[target])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check the distribution of classes in train and test sets
print(f"Training set class distribution: {np.bincount(y_train)}")
print(f"Test set class distribution: {np.bincount(y_test)}")

# Define models for training
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train the models
trained_models = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[model_name] = model
    print(f'{model_name} model trained.')

# Evaluate the models
for model_name, model in trained_models.items():
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print(f'\n{model_name} Model Evaluation:')
    print(f'Training Accuracy: {accuracy_score(y_train, y_pred_train)}')
    print(f'Testing Accuracy: {accuracy_score(y_test, y_pred_test)}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred_test))
    print('Classification Report:')
    print(classification_report(y_test, y_pred_test))

# Feature Importance Analysis (for Random Forest)
if 'Random Forest' in trained_models:
    rf_model = trained_models['Random Forest']
    feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\nTop 10 important features according to Random Forest:")
    print(feature_importance.head(10))

    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance.head(10), y=feature_importance.head(10).index)
    plt.title('Top 10 Important Features - Random Forest')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.show()

# 1. Correlation Heatmap for Top Features
top_features = feature_importance.head(10).index

plt.figure(figsize=(10, 8))
sns.heatmap(data[top_features].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Top 10 Important Features')
plt.show()

# 2. Pairplot for Top Features
sns.pairplot(data[top_features.union([target])], diag_kind='kde', kind='scatter')
plt.suptitle(f'Pairplot of Top 10 Features and {target}', y=1.02)
plt.show()

# 3. Bar Plot of Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance.head(10), y=feature_importance.head(10).index)
plt.title('Top 10 Important Features - Random Forest')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.show()

# 4. Detailed Correlation Analysis for Top 3 Features
top_3_features = feature_importance.head(3).index

for feature in top_3_features:
    corr_feature = data.corr()[feature].sort_values()
    top_5_neg = corr_feature.head(5).index
    top_5_pos = corr_feature.tail(5).index
    top_10_corr_features = top_5_neg.append(top_5_pos)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(data[[feature] + list(top_10_corr_features)].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title(f'Correlation of {feature} with Top 5 Positive and Negative Features')
    plt.show()

print("Done")
