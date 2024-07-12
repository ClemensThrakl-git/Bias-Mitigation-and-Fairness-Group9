import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, selection_rate, true_positive_rate, false_positive_rate, false_negative_rate
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# Load the cleaned dataset
data = pd.read_csv('cleaned_communities_crime_data.csv')

# Define the target and features
target = 'ViolentCrimesPerPop'
features = data.drop(columns=[target])
sensitive_features = ['racePctWhite', 'racepctblack']

# Binarize the target variable based on the mean
threshold = data[target].mean()
data['ViolentCrimesPerPop_binary'] = (data[target] > threshold).astype(int)

# Discretize the sensitive features
data['racePctWhite_bin'] = pd.cut(data['racePctWhite'], bins=5, labels=False)
data['racepctblack_bin'] = pd.cut(data['racepctblack'], bins=5, labels=False)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, data['ViolentCrimesPerPop_binary'], test_size=0.2, random_state=42
)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Define a function to calculate different metrics
def compute_metrics(y_true, y_pred):
    metrics = {
        'mean_absolute_error': mean_absolute_error(y_true, y_pred),
        'root_mean_squared_error': mean_squared_error(y_true, y_pred, squared=False),
        'r2_score': r2_score(y_true, y_pred),
        'selection_rate': selection_rate(y_true, y_pred),
        'false_positive_rate': false_positive_rate(y_true, y_pred),
        'false_negative_rate': false_negative_rate(y_true, y_pred),
        'true_positive_rate': true_positive_rate(y_true, y_pred)
    }
    return metrics

# Manually perform reweighing
def compute_sample_weights(data, sensitive_features, target):
    df = data.copy()
    df['weight'] = 1.0
    
    # Calculate the prevalence of each group
    group_counts = df.groupby(sensitive_features).size()
    total_count = len(df)
    
    for group, count in group_counts.items():
        group_weight = total_count / (len(group_counts) * count)
        df.loc[(df[sensitive_features] == group).all(axis=1), 'weight'] = group_weight
    
    return df['weight']

# Compute sample weights for training data
sample_weights = compute_sample_weights(data.loc[X_train.index], ['racePctWhite_bin', 'racepctblack_bin'], 'ViolentCrimesPerPop_binary')

# Train a Random Forest model on the reweighed data
rf_model_rw = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_rw.fit(X_train, y_train, sample_weight=sample_weights)

# Predict using the trained model
y_pred_rw = rf_model_rw.predict(X_test)

# Compute metrics for different groups
metrics_rw = MetricFrame(
    metrics=compute_metrics,
    y_true=y_test,
    y_pred=y_pred_rw,
    sensitive_features=data.loc[X_test.index, ['racePctWhite_bin', 'racepctblack_bin']]
)

# Print the overall metrics
print("Overall Metrics after Reweighing:")
print(metrics_rw.overall)

# Print metrics by sensitive feature groups
print("\nMetrics by Sensitive Feature Groups after Reweighing:")
print(metrics_rw.by_group)

# Extract metrics for visualization
metrics_by_group_rw = metrics_rw.by_group.apply(pd.Series)
mae_rw = metrics_by_group_rw['mean_absolute_error']
rmse_rw = metrics_by_group_rw['root_mean_squared_error']
r2_rw = metrics_by_group_rw['r2_score']
selection_rate_rw = metrics_by_group_rw['selection_rate']
fpr_rw = metrics_by_group_rw['false_positive_rate']
fnr_rw = metrics_by_group_rw['false_negative_rate']
tpr_rw = metrics_by_group_rw['true_positive_rate']

# Plot Mean Absolute Error by Group after Reweighing
mae_rw.plot(kind='bar', figsize=(12, 6), title='Mean Absolute Error by Group after Reweighing')
plt.ylabel('Mean Absolute Error')
plt.show()

# Plot Root Mean Squared Error by Group after Reweighing
rmse_rw.plot(kind='bar', figsize=(12, 6), title='Root Mean Squared Error by Group after Reweighing')
plt.ylabel('Root Mean Squared Error')
plt.show()

# Plot R^2 Score by Group after Reweighing
r2_rw.plot(kind='bar', figsize=(12, 6), title='R^2 Score by Group after Reweighing')
plt.ylabel('R^2 Score')
plt.show()

# Plot Selection Rate by Group after Reweighing
selection_rate_rw.plot(kind='bar', figsize=(12, 6), title='Selection Rate by Group after Reweighing')
plt.ylabel('Selection Rate')
plt.show()

# Plot False Positive Rate by Group after Reweighing
fpr_rw.plot(kind='bar', figsize=(12, 6), title='False Positive Rate by Group after Reweighing')
plt.ylabel('False Positive Rate')
plt.show()

# Plot False Negative Rate by Group after Reweighing
fnr_rw.plot(kind='bar', figsize=(12, 6), title='False Negative Rate by Group after Reweighing')
plt.ylabel('False Negative Rate')
plt.show()

# Plot True Positive Rate by Group after Reweighing
tpr_rw.plot(kind='bar', figsize=(12, 6), title='True Positive Rate by Group after Reweighing')
plt.ylabel('True Positive Rate')
plt.show()

# Perform adversarial debiasing using Exponentiated Gradient Reduction
mitigator = ExponentiatedGradient(estimator=RandomForestClassifier(random_state=42), 
                                  constraints=DemographicParity())

mitigator.fit(X_train, y_train, sensitive_features=data.loc[X_train.index, ['racePctWhite_bin', 'racepctblack_bin']])

y_pred_ad = mitigator.predict(X_test)

# Compute metrics for different groups
metrics_ad = MetricFrame(
    metrics=compute_metrics,
    y_true=y_test,
    y_pred=y_pred_ad,
    sensitive_features=data.loc[X_test.index, ['racePctWhite_bin', 'racepctblack_bin']]
)

# Print the overall metrics
print("Overall Metrics after Adversarial Debiasing:")
print(metrics_ad.overall)

# Print metrics by sensitive feature groups
print("\nMetrics by Sensitive Feature Groups after Adversarial Debiasing:")
print(metrics_ad.by_group)

# Extract metrics for visualization
metrics_by_group_ad = metrics_ad.by_group.apply(pd.Series)
mae_ad = metrics_by_group_ad['mean_absolute_error']
rmse_ad = metrics_by_group_ad['root_mean_squared_error']
r2_ad = metrics_by_group_ad['r2_score']
selection_rate_ad = metrics_by_group_ad['selection_rate']
fpr_ad = metrics_by_group_ad['false_positive_rate']
fnr_ad = metrics_by_group_ad['false_negative_rate']
tpr_ad = metrics_by_group_ad['true_positive_rate']

# Plot Mean Absolute Error by Group after Adversarial Debiasing
mae_ad.plot(kind='bar', figsize=(12, 6), title='Mean Absolute Error by Group after Adversarial Debiasing')
plt.ylabel('Mean Absolute Error')
plt.show()

# Plot Root Mean Squared Error by Group after Adversarial Debiasing
rmse_ad.plot(kind='bar', figsize=(12, 6), title='Root Mean Squared Error by Group after Adversarial Debiasing')
plt.ylabel('Root Mean Squared Error')
plt.show()

# Plot R^2 Score by Group after Adversarial Debiasing
r2_ad.plot(kind='bar', figsize=(12, 6), title='R^2 Score by Group after Adversarial Debiasing')
plt.ylabel('R^2 Score')
plt.show()

# Plot Selection Rate by Group after Adversarial Debiasing
selection_rate_ad.plot(kind='bar', figsize=(12, 6), title='Selection Rate by Group after Adversarial Debiasing')
plt.ylabel('Selection Rate')
plt.show()

# Plot False Positive Rate by Group after Adversarial Debiasing
fpr_ad.plot(kind='bar', figsize=(12, 6), title='False Positive Rate by Group after Adversarial Debiasing')
plt.ylabel('False Positive Rate')
plt.show()

# Plot False Negative Rate by Group after Adversarial Debiasing
fnr_ad.plot(kind='bar', figsize=(12, 6), title='False Negative Rate by Group after Adversarial Debiasing')
plt.ylabel('False Negative Rate')
plt.show()

# Plot True Positive Rate by Group after Adversarial Debiasing
tpr_ad.plot(kind='bar', figsize=(12, 6), title='True Positive Rate by Group after Adversarial Debiasing')
plt.ylabel('True Positive Rate')
plt.show()

# Perform post-processing using ThresholdOptimizer
postprocess_est = ThresholdOptimizer(estimator=rf_model, 
                                     constraints="equalized_odds", 
                                     prefit=True)

postprocess_est.fit(X_train, y_train, sensitive_features=data.loc[X_train.index, 'racePctWhite_bin'])

y_pred_pp = postprocess_est.predict(X_test, sensitive_features=data.loc[X_test.index, 'racePctWhite_bin'])

# Compute metrics for different groups
metrics_pp = MetricFrame(
    metrics=compute_metrics,
    y_true=y_test,
    y_pred=y_pred_pp,
    sensitive_features=data.loc[X_test.index, 'racePctWhite_bin']
)

# Print the overall metrics
print("Overall Metrics after Post-processing:")
print(metrics_pp.overall)

# Print metrics by sensitive feature groups
print("\nMetrics by Sensitive Feature Groups after Post-processing:")
print(metrics_pp.by_group)

# Extract metrics for visualization
metrics_by_group_pp = metrics_pp.by_group.apply(pd.Series)
mae_pp = metrics_by_group_pp['mean_absolute_error']
rmse_pp = metrics_by_group_pp['root_mean_squared_error']
r2_pp = metrics_by_group_pp['r2_score']
selection_rate_pp = metrics_by_group_pp['selection_rate']
fpr_pp = metrics_by_group_pp['false_positive_rate']
fnr_pp = metrics_by_group_pp['false_negative_rate']
tpr_pp = metrics_by_group_pp['true_positive_rate']

# Plot Mean Absolute Error by Group after Post-processing
mae_pp.plot(kind='bar', figsize=(12, 6), title='Mean Absolute Error by Group after Post-processing')
plt.ylabel('Mean Absolute Error')
plt.show()

# Plot Root Mean Squared Error by Group after Post-processing
rmse_pp.plot(kind='bar', figsize=(12, 6), title='Root Mean Squared Error by Group after Post-processing')
plt.ylabel('Root Mean Squared Error')
plt.show()

# Plot R^2 Score by Group after Post-processing
r2_pp.plot(kind='bar', figsize=(12, 6), title='R^2 Score by Group after Post-processing')
plt.ylabel('R^2 Score')
plt.show()

# Plot Selection Rate by Group after Post-processing
selection_rate_pp.plot(kind='bar', figsize=(12, 6), title='Selection Rate by Group after Post-processing')
plt.ylabel('Selection Rate')
plt.show()

# Plot False Positive Rate by Group after Post-processing
fpr_pp.plot(kind='bar', figsize=(12, 6), title='False Positive Rate by Group after Post-processing')
plt.ylabel('False Positive Rate')
plt.show()

# Plot False Negative Rate by Group after Post-processing
fnr_pp.plot(kind='bar', figsize=(12, 6), title='False Negative Rate by Group after Post-processing')
plt.ylabel('False Negative Rate')
plt.show()

# Plot True Positive Rate by Group after Post-processing
tpr_pp.plot(kind='bar', figsize=(12, 6), title='True Positive Rate by Group after Post-processing')
plt.ylabel('True Positive Rate')
plt.show()


# Comparison of results before and after bias mitigation

# Create a DataFrame for comparison
comparison_df = pd.DataFrame({
    'Reweighing': metrics_rw.overall,
    'Adversarial Debiasing': metrics_ad.overall,
    'Post-processing': metrics_pp.overall
})

# Plot comparison
comparison_df.plot(kind='bar', figsize=(12, 6), title='Comparison of Overall Metrics Before and After Bias Mitigation')
plt.ylabel('Metric Value')
plt.show()
