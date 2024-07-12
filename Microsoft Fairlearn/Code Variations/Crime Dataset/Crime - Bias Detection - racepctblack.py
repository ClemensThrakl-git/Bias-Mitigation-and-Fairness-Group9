import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from fairlearn.metrics import MetricFrame

# Load the cleaned dataset
data = pd.read_csv('cleaned_communities_crime_data.csv')

# Define the target and features
target = 'ViolentCrimesPerPop'
features = data.drop(columns=[target])
sensitive_features = ['racepctblack']

# Discretize the sensitive feature
data['racepctblack_bin'] = pd.cut(data['racepctblack'], bins=5, labels=False)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, data[target], test_size=0.2, random_state=42
)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict using the trained model
y_pred = rf_model.predict(X_test)

# Define individual metric functions
def mean_absolute_error_metric(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def root_mean_squared_error_metric(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def r2_score_metric(y_true, y_pred):
    return r2_score(y_true, y_pred)

# Compute metrics for different groups
metrics = MetricFrame(
    metrics={
        'mean_absolute_error': mean_absolute_error_metric,
        'root_mean_squared_error': root_mean_squared_error_metric,
        'r2_score': r2_score_metric,
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=data.loc[X_test.index, 'racepctblack_bin']
)

# Print the overall metrics
print("Overall Metrics:")
print(metrics.overall)

# Print metrics by sensitive feature groups
print("\nMetrics by Sensitive Feature Groups:")
print(metrics.by_group)

# Extract metrics for visualization
metrics_by_group = metrics.by_group.apply(pd.Series)
mae = metrics_by_group['mean_absolute_error']
rmse = metrics_by_group['root_mean_squared_error']
r2 = metrics_by_group['r2_score']

# Plot Mean Absolute Error by Group
mae.plot(kind='bar', figsize=(12, 6), title='Mean Absolute Error by Group')
plt.ylabel('Mean Absolute Error')
plt.show()

# Plot Root Mean Squared Error by Group
rmse.plot(kind='bar', figsize=(12, 6), title='Root Mean Squared Error by Group')
plt.ylabel('Root Mean Squared Error')
plt.show()

# Plot R^2 Score by Group
r2.plot(kind='bar', figsize=(12, 6), title='R^2 Score by Group')
plt.ylabel('R^2 Score')
plt.show()

# Reweighing Debiasing
# Manually perform reweighing
def compute_sample_weights(data, sensitive_features, target):
    df = data.copy()
    df['weight'] = 1.0
    # Calculate the prevalence of each group
    group_counts = df.groupby(sensitive_features).size()
    total_count = len(df)
    for group, count in group_counts.items():
        group_weight = total_count / (len(group_counts) * count)
        df.loc[df[sensitive_features] == group, 'weight'] = group_weight
    return df['weight']

# Compute sample weights for training data
sample_weights = compute_sample_weights(data.loc[X_train.index], 'racepctblack_bin', target)

# Train a Random Forest model on the reweighed data
rf_model_rw = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_rw.fit(X_train, y_train, sample_weight=sample_weights)

# Predict using the trained model
y_pred_rw = rf_model_rw.predict(X_test)

# Compute metrics for different groups
metrics_rw = MetricFrame(
    metrics={
        'mean_absolute_error': mean_absolute_error_metric,
        'root_mean_squared_error': root_mean_squared_error_metric,
        'r2_score': r2_score_metric,
    },
    y_true=y_test,
    y_pred=y_pred_rw,
    sensitive_features=data.loc[X_test.index, 'racepctblack_bin']
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

# Perform post-processing using ThresholdOptimizer (only valid for classification)
# Since ThresholdOptimizer is for classification, we will skip it in this regression context.

# Comparison of results before and after bias mitigation

# Create a DataFrame for comparison
comparison_df = pd.DataFrame({
    'Initial': metrics.overall,
    'Reweighing': metrics_rw.overall,
})

# Plot comparison
comparison_df.plot(kind='bar', figsize=(12, 6), title='Comparison of Overall Metrics Before and After Bias Mitigation')
plt.ylabel('Metric Value')
plt.show()

# Function to plot comparison of metrics for different mitigation steps
def plot_metric_comparison(metrics_initial, metrics_rw, metric_name, ylabel, title):
    labels = metrics_initial.index

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, metrics_initial[metric_name], width, label='Initial')
    rects2 = ax.bar(x + width/2, metrics_rw[metric_name], width, label='Reweighing')

    ax.set_xlabel('Groups')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    plt.show()

# Extracting overall metrics
overall_metrics_initial = metrics.overall
overall_metrics_rw = metrics_rw.overall

# Plot overall metrics comparison
def plot_overall_metric_comparison():
    metric_names = ['mean_absolute_error', 'root_mean_squared_error', 'r2_score']
    for metric in metric_names:
        plot_metric_comparison(
            overall_metrics_initial, overall_metrics_rw, 
            metric, metric.replace('_', ' ').title(), 
            f'Comparison of {metric.replace("_", " ").title()}'
        )

# Plot overall metric comparison
plot_overall_metric_comparison()

# Plot metrics by group comparison
def plot_metrics_by_group_comparison(metrics_by_group_initial, metrics_by_group_rw):
    metric_names = ['mean_absolute_error', 'root_mean_squared_error', 'r2_score']
    for metric in metric_names:
        plot_metric_comparison(
            metrics_by_group_initial, metrics_by_group_rw, 
            metric, metric.replace('_', ' ').title(), 
            f'Comparison of {metric.replace("_", " ").title()} by Group'
        )

# Extracting metrics by group
metrics_by_group_initial = metrics.by_group.apply(pd.Series)
metrics_by_group_rw = metrics_rw.by_group.apply(pd.Series)

# Plot metrics by group comparison
plot_metrics_by_group_comparison(metrics_by_group_initial, metrics_by_group_rw)
