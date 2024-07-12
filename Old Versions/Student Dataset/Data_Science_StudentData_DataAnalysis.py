# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the cleaned dataset
data = pd.read_csv('cleaned_student_data2.csv')

# Exploratory Data Analysis (EDA)

# 1. Visualizations

# Select a subset of the most relevant features based on correlation with the target
# Let's assume these are some of the relevant features we want to explore
relevant_features = ['age', 'Medu', 'Fedu', 'studytime', 'failures', 'goout', 'Dalc', 'Walc', 'health', 'absences']

# Distribution plots for relevant features
plt.figure(figsize=(18, 12))
for i, column in enumerate(relevant_features, 1):
    plt.subplot(4, 3, i)
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Boxplots for relevant features
plt.figure(figsize=(18, 12))
for i, column in enumerate(relevant_features, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)
plt.tight_layout()
plt.show()

# Scatter plots for relationships between relevant features and target variable
target = 'G3'
plt.figure(figsize=(18, 12))
for i, column in enumerate(relevant_features, 1):
    plt.subplot(4, 3, i)
    sns.scatterplot(x=data[column], y=data[target])
    plt.title(f'Relationship between {column} and {target}')
    plt.xlabel(column)
    plt.ylabel(target)
plt.tight_layout()
plt.show()

# Heatmap for correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Cleaned Data')
plt.show()

# 2. Descriptive Statistics
print("Summary Statistics for Numerical Features:")
print(data.describe())

# 3. Correlation Analysis
corr_with_target = data.corr()[target].sort_values(ascending=False)
print(f"Top 10 features positively correlated with {target}:")
print(corr_with_target.head(10))
print(f"\nTop 10 features negatively correlated with {target}:")
print(corr_with_target.tail(10))

# 4. Feature Relationships with Sensitive Features
sensitive_features = ['age', 'Medu', 'Fedu', 'Dalc']
plt.figure(figsize=(18, 12))
for i, feature in enumerate(sensitive_features, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(x=data[feature], y=data[target])
    plt.title(f'Relationship between {feature} and {target}')
    plt.xlabel(feature)
    plt.ylabel(target)
plt.tight_layout()
plt.show()

# Building and Evaluating Predictive Models

# 1. Model Selection: Linear Regression and Random Forest
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# 2. Split the dataset into training and testing sets
X = data.drop(columns=[target])
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the models
trained_models = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[model_name] = model
    print(f'{model_name} model trained.')

# 4. Evaluate the models
for model_name, model in trained_models.items():
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print(f'\n{model_name} Model Evaluation:')
    print(f'Training RMSE: {mean_squared_error(y_train, y_pred_train, squared=False)}')
    print(f'Testing RMSE: {mean_squared_error(y_test, y_pred_test, squared=False)}')
    print(f'Training MAE: {mean_absolute_error(y_train, y_pred_train)}')
    print(f'Testing MAE: {mean_absolute_error(y_test, y_pred_test)}')
    print(f'Training R^2: {r2_score(y_train, y_pred_train)}')
    print(f'Testing R^2: {r2_score(y_test, y_pred_test)}')

# 5. Feature Importance Analysis (for Random Forest)
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

# Define the target variable
target = 'G3'

# Random Forest Model to determine feature importance
X = data.drop(columns=[target])
y = data[target]

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Feature importance
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_features = feature_importance.head(10).index

# 1. Correlation Heatmap for Top Features
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
