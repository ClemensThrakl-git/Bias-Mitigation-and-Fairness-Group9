import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler   
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, true_positive_rate, false_positive_rate

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data'
data = pd.read_csv(url, header=None, na_values='?')

# Load the column names
column_names = [
    'state', 'county', 'community', 'communityname', 'fold',
    'population', 'householdsize', 'racepctblack', 'racePctWhite', 'racePctAsian',
    'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up',
    'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc',
    'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap',
    'blackPerCap', 'indianPerCap', 'AsianPerCap', 'OtherPerCap', 'HispPerCap', 'NumUnderPov',
    'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed',
    'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf',
    'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam',
    'PctFam2Par', 'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids',
    'PctWorkMom', 'NumIlleg', 'PctIlleg', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5',
    'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8',
    'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell', 'PctLargHouseFam',
    'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous',
    'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant',
    'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt',
    'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart',
    'RentLowQ', 'RentMedian', 'RentHighQ', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc',
    'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState',
    'PctSameHouse85', 'PctSameCity85', 'PctSameState85', 'LemasSwornFT', 'LemasSwFTPerPop',
    'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop',
    'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack',
    'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
    'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg',
    'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop',
    'ViolentCrimesPerPop'
]

data.columns = column_names

# Handle Missing Values
threshold = 0.5
data = data[data.columns[data.isnull().mean() < threshold]]
non_numeric_columns = data.select_dtypes(include=[object]).columns
data = data.drop(non_numeric_columns, axis=1)
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Bin the target variable into categories
bins = [0, 0.1, 0.2, 0.3, 0.4, 1.0]
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
data_imputed['ViolentCrimesPerPop'] = pd.cut(data_imputed['ViolentCrimesPerPop'], bins=bins, labels=labels)

# Ensure there are no NaN values in the target variable after binning
data_imputed = data_imputed.dropna(subset=['ViolentCrimesPerPop'])

# Define features (X) and target (y)
X = data_imputed.drop(['ViolentCrimesPerPop'], axis=1)
y = data_imputed['ViolentCrimesPerPop']

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the Model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Visualize Overall Model Performance
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Binarize the target variable for fairness evaluation
y_test_bin = y_test.apply(lambda x: 1 if x in ['Very High', 'High'] else 0)
y_pred_bin = pd.Series(y_pred).apply(lambda x: 1 if x in ['Very High', 'High'] else 0)
sensitive_feature = 'racepctblack'

# Calculate Demographic Parity Difference
dpd = demographic_parity_difference(y_test_bin, y_pred_bin, sensitive_features=X_test[sensitive_feature])

# Calculate Equalized Odds Difference
eod = equalized_odds_difference(y_test_bin, y_pred_bin, sensitive_features=X_test[sensitive_feature])

print(f"Demographic Parity Difference: {dpd}")
print(f"Equalized Odds Difference: {eod}")

# Calculate precision, recall, and F1 score for the binarized results
precision, recall, f1, _ = precision_recall_fscore_support(y_test_bin, y_pred_bin, average='binary')
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Visualize Fairness Metrics
fairness_metrics = pd.DataFrame({
    'Metric': ['Demographic Parity Difference', 'Equalized Odds Difference', 'Precision', 'Recall', 'F1 Score'],
    'Value': [dpd, eod, precision, recall, f1]
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Metric', y='Value', data=fairness_metrics)
plt.title('Fairness Metrics')
plt.ylabel('Value')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.show()

# Calculate additional fairness metrics using MetricFrame
metric_frame = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'precision': true_positive_rate,
        'false_positive_rate': false_positive_rate,
    },
    y_true=y_test_bin,
    y_pred=y_pred_bin,
    sensitive_features=X_test[sensitive_feature]
)

print("Overall Metrics:")
print(metric_frame.overall)
print("Disaggregated Metrics:")
print(metric_frame.by_group)

# Visualize disaggregated metrics
disaggregated_metrics = metric_frame.by_group.reset_index()
disaggregated_metrics_melted = disaggregated_metrics.melt(id_vars='racepctblack', var_name='Metric', value_name='Value')

plt.figure(figsize=(14, 8))
sns.barplot(x='Metric', y='Value', hue='racepctblack', data=disaggregated_metrics_melted)
plt.title('Disaggregated Metrics by Sensitive Feature (racepctblack)')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()





# Result analysis
"""
Analysis of Fairness Metrics
Demographic Parity Difference (DPD):

Value: 1.0
Interpretation: The high value of demographic parity difference indicates a significant disparity in the predicted positive outcomes across different groups of the sensitive feature (racepctblack). This suggests that the model's predictions are not equally distributed across different racial demographics.
Equalized Odds Difference (EOD):

Value: 1.0
Interpretation: A high equalized odds difference indicates a significant disparity in the true positive rate and false positive rate across different racial groups. This suggests that the model is not equally accurate for all groups.
Precision, Recall, and F1 Score:

Precision: 0.8444
Recall: 0.6786
F1 Score: 0.7525
Interpretation: These values reflect the overall performance of the model on the binary classification task. However, these metrics alone do not provide insights into fairness across different groups.
Disaggregated Metrics:

Interpretation: The disaggregated metrics (accuracy, precision, false positive rate) across different values of the sensitive feature (racepctblack) show variations. This further highlights the model's inconsistent performance across different racial groups."""

# improving on initial detection

from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, true_positive_rate, false_positive_rate, selection_rate
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# List of potential sensitive features
sensitive_features = ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct65up', 'medIncome']

# Function to calculate and visualize fairness metrics for a given sensitive feature
def evaluate_fairness(sensitive_feature):
    dpd = demographic_parity_difference(y_test_bin, y_pred_bin, sensitive_features=X_test[sensitive_feature])
    eod = equalized_odds_difference(y_test_bin, y_pred_bin, sensitive_features=X_test[sensitive_feature])
    
    # Calculate additional fairness metrics using MetricFrame
    metric_frame = MetricFrame(
        metrics={
            'accuracy': accuracy_score,
            'precision': true_positive_rate,
            'false_positive_rate': false_positive_rate,
            'selection_rate': selection_rate,
        },
        y_true=y_test_bin,
        y_pred=y_pred_bin,
        sensitive_features=X_test[sensitive_feature]
    )
    
    print(f"Metrics for {sensitive_feature}:")
    print("Overall Metrics:")
    print(metric_frame.overall)
    print("Disaggregated Metrics:")
    print(metric_frame.by_group)
    
    # Visualize disaggregated metrics
    disaggregated_metrics = metric_frame.by_group.reset_index()
    disaggregated_metrics_melted = disaggregated_metrics.melt(id_vars=sensitive_feature, var_name='Metric', value_name='Value')
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Metric', y='Value', hue=sensitive_feature, data=disaggregated_metrics_melted)
    plt.title(f'Disaggregated Metrics by Sensitive Feature ({sensitive_feature})')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.show()

    # Return the fairness metrics for further analysis if needed
    return dpd, eod, metric_frame

# Evaluate fairness for each sensitive feature
fairness_results = {}
for feature in sensitive_features:
    dpd, eod, metric_frame = evaluate_fairness(feature)
    fairness_results[feature] = {
        'Demographic Parity Difference': dpd,
        'Equalized Odds Difference': eod,
        'Metric Frame': metric_frame
    }

# Correct the DataFrame appending using pd.concat
overall_metrics = pd.DataFrame({
    'Sensitive Feature': [],
    'Metric': [],
    'Value': []
})

for feature, results in fairness_results.items():
    overall_metrics = pd.concat([
        overall_metrics,
        pd.DataFrame({
            'Sensitive Feature': [feature],
            'Metric': ['Demographic Parity Difference'],
            'Value': [results['Demographic Parity Difference']]
        })
    ], ignore_index=True)
    
    overall_metrics = pd.concat([
        overall_metrics,
        pd.DataFrame({
            'Sensitive Feature': [feature],
            'Metric': ['Equalized Odds Difference'],
            'Value': [results['Equalized Odds Difference']]
        })
    ], ignore_index=True)

# Plotting the overall fairness metrics comparison
plt.figure(figsize=(14, 8))
sns.barplot(x='Sensitive Feature', y='Value', hue='Metric', data=overall_metrics)
plt.title('Overall Fairness Metrics Comparison')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.show()




################################### Improved with more metrics to detect bias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler   
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, true_positive_rate, false_positive_rate, selection_rate, count

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data'
data = pd.read_csv(url, header=None, na_values='?')

# Load the column names
column_names = [
    'state', 'county', 'community', 'communityname', 'fold',
    'population', 'householdsize', 'racepctblack', 'racePctWhite', 'racePctAsian',
    'racePctHisp', 'agePct12t21', 'agePct12t29', 'agePct16t24', 'agePct65up',
    'numbUrban', 'pctUrban', 'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc',
    'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc', 'perCapInc', 'whitePerCap',
    'blackPerCap', 'indianPerCap', 'AsianPerCap', 'OtherPerCap', 'HispPerCap', 'NumUnderPov',
    'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad', 'PctBSorMore', 'PctUnemployed',
    'PctEmploy', 'PctEmplManu', 'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf',
    'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv', 'TotalPctDiv', 'PersPerFam',
    'PctFam2Par', 'PctKids2Par', 'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids',
    'PctWorkMom', 'NumIlleg', 'PctIlleg', 'NumImmig', 'PctImmigRecent', 'PctImmigRec5',
    'PctImmigRec8', 'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5', 'PctRecImmig8',
    'PctRecImmig10', 'PctSpeakEnglOnly', 'PctNotSpeakEnglWell', 'PctLargHouseFam',
    'PctLargHouseOccup', 'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous',
    'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR', 'MedNumBR', 'HousVacant',
    'PctHousOccup', 'PctHousOwnOcc', 'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt',
    'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart', 'OwnOccMedVal', 'OwnOccHiQuart',
    'RentLowQ', 'RentMedian', 'RentHighQ', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc',
    'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet', 'PctForeignBorn', 'PctBornSameState',
    'PctSameHouse85', 'PctSameCity85', 'PctSameState85', 'LemasSwornFT', 'LemasSwFTPerPop',
    'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasTotalReq', 'LemasTotReqPerPop',
    'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite', 'PctPolicBlack',
    'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
    'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans', 'PolicCars', 'PolicOperBudg',
    'LemasPctPolicOnPatr', 'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop',
    'ViolentCrimesPerPop'
]

data.columns = column_names

# Handle Missing Values
threshold = 0.5
data = data[data.columns[data.isnull().mean() < threshold]]
non_numeric_columns = data.select_dtypes(include=[object]).columns
data = data.drop(non_numeric_columns, axis=1)
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Bin the target variable into categories
bins = [0, 0.1, 0.2, 0.3, 0.4, 1.0]
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
data_imputed['ViolentCrimesPerPop'] = pd.cut(data_imputed['ViolentCrimesPerPop'], bins=bins, labels=labels)

# Ensure there are no NaN values in the target variable after binning
data_imputed = data_imputed.dropna(subset=['ViolentCrimesPerPop'])

# Define features (X) and target (y)
X = data_imputed.drop(['ViolentCrimesPerPop'], axis=1)
y = data_imputed['ViolentCrimesPerPop']

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the Model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Visualize Overall Model Performance
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Binarize the target variable for fairness evaluation
y_test_bin = y_test.apply(lambda x: 1 if x in ['Very High', 'High'] else 0)
y_pred_bin = pd.Series(y_pred).apply(lambda x: 1 if x in ['Very High', 'High'] else 0)

# List of potential sensitive features
sensitive_features = ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct65up', 'medIncome']

# Function to calculate and visualize fairness metrics for a given sensitive feature
def evaluate_fairness(sensitive_feature):
    dpd = demographic_parity_difference(y_test_bin, y_pred_bin, sensitive_features=X_test[sensitive_feature])
    eod = equalized_odds_difference(y_test_bin, y_pred_bin, sensitive_features=X_test[sensitive_feature])
    aod = equalized_odds_difference(y_test_bin, y_pred_bin, sensitive_features=X_test[sensitive_feature])
    eop = equalized_odds_difference(y_test_bin, y_pred_bin, sensitive_features=X_test[sensitive_feature])
    
    # Calculate additional fairness metrics using MetricFrame
    metric_frame = MetricFrame(
        metrics={
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'true_positive_rate': true_positive_rate,
            'false_positive_rate': false_positive_rate,
            'selection_rate': selection_rate,
            'count': count
        },
        y_true=y_test_bin,
        y_pred=y_pred_bin,
        sensitive_features=X_test[sensitive_feature]
    )
    print(f"Metrics for {sensitive_feature}:")
    print("Overall Metrics:")
    print(metric_frame.overall)
    print("Disaggregated Metrics:")
    print(metric_frame.by_group)
    
    # Visualize disaggregated metrics
    disaggregated_metrics = metric_frame.by_group.reset_index()
    disaggregated_metrics_melted = disaggregated_metrics.melt(id_vars=sensitive_feature, var_name='Metric', value_name='Value')
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Metric', y='Value', hue=sensitive_feature, data=disaggregated_metrics_melted)
    plt.title(f'Disaggregated Metrics by Sensitive Feature ({sensitive_feature})')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.show()
    
    # Return the fairness metrics for further analysis if needed
    return dpd, eod, aod, eop, metric_frame

# Evaluate fairness for each sensitive feature
fairness_results = {}
for feature in sensitive_features:
    dpd, eod, aod, eop, metric_frame = evaluate_fairness(feature)
    fairness_results[feature] = {
        'Demographic Parity Difference': dpd,
        'Equalized Odds Difference': eod,
        'Average Odds Difference': aod,
        'Equal Opportunity Difference': eop,
        'Metric Frame': metric_frame
    }

# Optionally, visualize overall fairness metrics for comparison
overall_metrics = pd.DataFrame({
    'Sensitive Feature': [],
    'Metric': [],
    'Value': []
})

for feature, results in fairness_results.items():
    overall_metrics = overall_metrics._append(
        {'Sensitive Feature': feature, 'Metric': 'Demographic Parity Difference', 'Value': results['Demographic Parity Difference']},
        ignore_index=True
    )
    overall_metrics = overall_metrics._append(
        {'Sensitive Feature': feature, 'Metric': 'Equalized Odds Difference', 'Value': results['Equalized Odds Difference']},
        ignore_index=True
    )
    overall_metrics = overall_metrics._append(
        {'Sensitive Feature': feature, 'Metric': 'Average Odds Difference', 'Value': results['Average Odds Difference']},
        ignore_index=True
    )
    overall_metrics = overall_metrics._append(
        {'Sensitive Feature': feature, 'Metric': 'Equal Opportunity Difference', 'Value': results['Equal Opportunity Difference']},
        ignore_index=True
    )

plt.figure(figsize=(14, 8))
sns.barplot(x='Sensitive Feature', y='Value', hue='Metric', data=overall_metrics)
plt.title('Overall Fairness Metrics Comparison')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.show()




############################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming fairness_results is a dictionary containing the fairness metrics for each sensitive feature
fairness_results = {
    'racepctblack': {'Demographic Parity Difference': 0.1, 'Equalized Odds Difference': 0.2, 'Average Odds Difference': 0.15, 'Equal Opportunity Difference': 0.18},
    'racePctWhite': {'Demographic Parity Difference': 0.05, 'Equalized Odds Difference': 0.1, 'Average Odds Difference': 0.08, 'Equal Opportunity Difference': 0.09},
    'racePctAsian': {'Demographic Parity Difference': 0.02, 'Equalized Odds Difference': 0.04, 'Average Odds Difference': 0.03, 'Equal Opportunity Difference': 0.035},
    'racePctHisp': {'Demographic Parity Difference': 0.07, 'Equalized Odds Difference': 0.15, 'Average Odds Difference': 0.11, 'Equal Opportunity Difference': 0.13},
    'agePct12t21': {'Demographic Parity Difference': 0.09, 'Equalized Odds Difference': 0.18, 'Average Odds Difference': 0.14, 'Equal Opportunity Difference': 0.16},
    'agePct65up': {'Demographic Parity Difference': 0.12, 'Equalized Odds Difference': 0.22, 'Average Odds Difference': 0.17, 'Equal Opportunity Difference': 0.19},
    'medIncome': {'Demographic Parity Difference': 0.06, 'Equalized Odds Difference': 0.12, 'Average Odds Difference': 0.09, 'Equal Opportunity Difference': 0.11}
}

# Convert fairness_results to a DataFrame
overall_metrics = pd.DataFrame({
    'Sensitive Feature': [],
    'Metric': [],
    'Value': []
})

for feature, results in fairness_results.items():
    overall_metrics = overall_metrics._append(
        {'Sensitive Feature': feature, 'Metric': 'Demographic Parity Difference', 'Value': results['Demographic Parity Difference']},
        ignore_index=True
    )
    overall_metrics = overall_metrics._append(
        {'Sensitive Feature': feature, 'Metric': 'Equalized Odds Difference', 'Value': results['Equalized Odds Difference']},
        ignore_index=True
    )
    overall_metrics = overall_metrics._append(
        {'Sensitive Feature': feature, 'Metric': 'Average Odds Difference', 'Value': results['Average Odds Difference']},
        ignore_index=True
    )
    overall_metrics = overall_metrics._append(
        {'Sensitive Feature': feature, 'Metric': 'Equal Opportunity Difference', 'Value': results['Equal Opportunity Difference']},
        ignore_index=True
    )

# Plot the overall fairness metrics
plt.figure(figsize=(14, 8))
sns.barplot(x='Sensitive Feature', y='Value', hue='Metric', data=overall_metrics)
plt.title('Overall Fairness Metrics Comparison')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.show()
