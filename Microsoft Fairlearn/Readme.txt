Link to Github repository: https://github.com/ClemensThrakl-git/Bias-Mitigation-and-Fairness-Group9
Link to Crime Dataset: https://archive.ics.uci.edu/dataset/211/communities+and+crime+unnormalized
Link to Student Dataset: https://archive.ics.uci.edu/dataset/320/student+performance


For the Fairlearn Notebooks always when opening them just run all cells, they are ment to be executed in the corresponding order that the cells are put in. 
The Notebooks already come executed with the results stored so they can be observed without having to run the notebook. But if you want to execute them yourself open them in the corresponding Steps in their labeling (Step 1 - Step 5) and execute them in that order.

Bias Detection and Mitigation Using Microsoft Fairlearn:
This part of the project focuses on bias detection and mitigation using Microsoft Fairlearn, analyzing two datasets: crime statistics and student failure rates. 
The repository is organized to ensure clear documentation and easy navigation.


Repository Structure:

Fairlearn: This folder contains all the code related to bias detection and mitigation.
	Main Code: The primary scripts used for the analysis. (Code from Start to FInish for one example run)
	Code Variations: Alternative versions of the main code, which run with different variables or parameters. (to get different information and results and insights)
	Old Code: Previous iterations of the code, maintained to document the development progress.
	Analysis of initial Tests: Contains 2 Word Documents that Contain the Results of each individual Step from the first full Testrun and document the findings and our interpretations of the resuls.
	Screenshots of some Preliminary Results: Contains Screenshots of some of the Results from the first Iteration of Code. (Results also contained within corresponding Code Notebooks)

Main Code Folder Structure:
Inside the Main Code folder, you will find two subfolders:
	Student Failures: Contains the scripts for analyzing the student failure dataset.
	Crime: Contains the scripts for analyzing the crime statistics dataset.


Within the Main Code Folder is always one Example for each Data-Set performed through 5 Steps (Data Exploration, Cleaning, Analysis, Bias Detection, Bias Mitigation). 
This is performed for one Case to see a full case example. For more different in depth results and analysis we performed multiple different approaches which are saved within the folder of "Code Variations". 
Those are mostly just different Variations in regards to bias detection and mitigation, but because of changes in the approach the code always had to be adjusted a bit.
Furthermore within the Main Code Folder there is one jupyter notebook file that contains a list of all install statements for each library that is required. Run this once in the beginning to ensure all necessary libraries are installed if you plan on running the code yourself.
Also the "Code Variations" contain same code in form of normal python code (without execution output but therefore small filesize) and as juypter notebooks (with execution output but bigger filesize)



#################################################################################################################################################################################################################



Crime Statistics: 

Data Exploration
Purpose
The purpose of this notebook is to explore the crime statistics dataset, understand its structure, visualize key features, and prepare the data for modeling. This step is crucial to identify any biases and understand the data before applying any bias mitigation techniques.

Steps
Load the Dataset:
We start by loading the crime statistics dataset and assigning appropriate column names.
View Basic Information:
Display basic information about the dataset to understand its structure and content.

Check for Missing Values:
Check for missing values to understand the completeness of the data.

Summarize Numerical and Categorical Features:
Summarize numerical features to get an idea of their distributions and central tendencies.
Summarize categorical features to understand the distribution of categorical variables.

Visualize Distributions of Numerical Features:
Visualize the distributions to see the spread and skewness of numerical features.

Explore Relationships Between Features:
Create a correlation matrix to understand the relationships between numerical features.
Generate pair plots for a subset of numerical features to visualize relationships between pairs of numerical features.

Examine Target Variable Distributions and Relationships with Other Features:
Analyze the distribution of the target variable ViolentCrimesPerPop.
Explore the relationship between the target variable and categorical features.

Handle Missing Values and Encode Categorical Features:
Fill missing numerical values and encode categorical features using one-hot encoding.

Analyze Feature Importance:
Train a Random Forest model to analyze feature importance.

Detect Outliers:
Use boxplots to identify outliers in both the target variable and the top numerical features.
Explore the relationship between the target variable and categorical features using boxplots.

Each step is designed to provide a thorough understanding of the dataset, preparing it for subsequent bias detection and mitigation steps.




Data Cleaning
In this next section, we preprocess the crime statistics dataset to prepare it for further analysis and modeling. 
This involves handling missing values, encoding categorical variables, removing outliers, normalizing numerical features, and splitting the dataset into training and testing sets.

Loading the Dataset: We load the dataset from the UCI repository and assign appropriate column names.
Initial Data Exploration: We review basic information about the dataset, including the first few rows and summary statistics.
Handling Missing Values: Missing values in numerical features are imputed using the mean value.
Encoding Categorical Variables: Non-essential categorical columns are dropped to simplify the dataset.
Outlier Removal: Outliers are removed using the Interquartile Range (IQR) method.
Normalizing Numerical Features: Numerical features are normalized to ensure equal contribution to the analysis.
Splitting the Dataset: The dataset is split into training and testing sets to evaluate model performance.






Data Analysis
In this next section, we perform Exploratory Data Analysis (EDA) and build predictive models using Linear Regression and Random Forest. We evaluate the performance of these models and analyze feature importance.

Loading the Dataset: We load the cleaned dataset for further analysis.

Exploratory Data Analysis (EDA):
Visualizations: Distribution plots, boxplots, scatter plots, and heatmap for relevant features.
Descriptive Statistics: Summary statistics for numerical features.
Correlation Analysis: Correlations of features with the target variable.
Feature Relationships with Sensitive Features: Scatter plots to understand relationships between sensitive features and the target variable.

Building and Evaluating Predictive Models:
Model Selection: Linear Regression and Random Forest.
Train-Test Split: Splitting the dataset into training and testing sets.
Model Training: Training the selected models.
Model Evaluation: Evaluating the models using RMSE, MAE, and R^2.
Feature Importance Analysis: Analyzing feature importance using Random Forest.





Bias Detection:
The next step performs a comprehensive fairness evaluation of a Random Forest model trained to predict violent crime rates and detect bias. The process includes:

Model Training: Train a Random Forest model.
Metric Computation: Calculate standard performance metrics (accuracy, precision, recall, F1 score) and fairness metrics (demographic parity, equalized odds, selection rate differences).
Visualization: Visualize metrics by sensitive groups to identify any potential biases.
Additional Metrics: Compute and visualize additional custom fairness metrics for deeper insights.





Bias Mitigation:
The Last Part Contains the Bias Mitigation (and improved steps for the bias detection - so parts of Step 4 are contained for comparrison and got optimized a bit)


The following code implements a comprehensive approach to bias detection and mitigation in a Random Forest classification model.

We focus on ViolentCrimesPerPop as the target variable and racepctblack as the sensitive feature. The notebook covers data preparation, bias detection, and various bias mitigation techniques.

1. Load the Dataset
We start by loading the cleaned dataset. The target variable ViolentCrimesPerPop is binarized based on its mean value to facilitate binary classification. The sensitive feature racepctblack is discretized into bins to allow for group-based fairness analysis.

2. Data Preparation
Binarizing the Target Variable: The target variable is transformed into a binary format, with values greater than the mean being labeled as 1 and others as 0.
Discretizing the Sensitive Feature: The racepctblack feature is divided into bins, creating categorical groups for fairness analysis.
Splitting the Dataset: The dataset is split into training and testing sets to evaluate model performance and fairness.
3. Training the Base Model
A RandomForestClassifier model is trained on the prepared dataset. Predictions are made on the test set, which will be used for initial bias detection.

4. Bias Detection
Custom Metric Functions: We define custom functions for calculating accuracy, precision, recall, and F1 score.
Compute Metrics by Group: Using Fairlearn's MetricFrame, we compute these metrics for different groups based on the sensitive feature (racepctblack_bin). This helps us understand how the model performs across different subgroups.
Visualize Metrics by Group: We plot the metrics (accuracy, precision, recall, and F1 score) for each subgroup to visualize disparities in model performance.
Fairness Metrics: We calculate fairness metrics such as Demographic Parity Difference and Equalized Odds Difference to quantify bias in the model.
5. Bias Mitigation
Several techniques are employed to mitigate bias in the model:

Reweighing
Compute Sample Weights: Adjusts sample weights to ensure that each group is represented equally.
Train Reweighed Model: A new RandomForestClassifier is trained using these sample weights.
Predict and Evaluate: The model is evaluated on the test set, and performance metrics are computed for each group.

Adversarial Debiasing
Exponentiated Gradient Reduction: This technique is used to reduce bias by iteratively adjusting the model to satisfy fairness constraints.
Train and Evaluate: The mitigated model is trained and evaluated, with performance metrics computed for each group.

Post-processing
Threshold Optimizer: This method adjusts the decision thresholds for different groups to satisfy fairness constraints.
Train and Evaluate: The post-processed model is trained and evaluated, with performance metrics computed for each group.

Comparison of Results
We compare the metrics of the original model with the reweighed, adversarially debiased, and post-processed models to understand the effectiveness of each bias mitigation technique. 
Visualizations such as bar plots and radar charts are used to compare the overall metrics and fairness metrics before and after mitigation.

Fairlearn Dashboard (attempted - code there, library didn´t work propperly)
To provide an interactive way to explore the results, the Fairlearn Dashboard is included. This tool allows users to inspect the performance and fairness of the models across different subgroups interactively.


Conclusion and Takeaway of the results here:

The bias mitigation techniques applied to the crime dataset—reweighing, adversarial debiasing, and post-processing—show varied impacts on the model's performance and fairness. 
Reweighing and post-processing provided more balanced improvements across different racial demographic bins, maintaining relatively high accuracy and precision while reducing bias. 
Adversarial debiasing, although significantly reducing bias, caused a substantial drop in recall and F1 scores, indicating a trade-off between fairness and overall model effectiveness. 
The key takeaway is that while no single mitigation technique is perfect, reweighing and post-processing offer promising methods to enhance fairness with minimal performance degradation, whereas adversarial debiasing may require careful tuning to avoid significant losses in model performance.





###############################################################################################################################


Student Dataset

Data Exploration
This notebook explores the Student Performance dataset. The dataset contains information on student achievement in secondary education in two Portuguese schools. 
This exploration involves understanding the data structure, checking for missing values, summarizing the features, visualizing distributions, exploring feature relationships, handling missing values and categorical encoding, and performing outlier detection. The steps include:

Steps:
Load the Dataset: Fetch the Student Performance dataset from the UCI repository and load it into a pandas DataFrame.
Basic Information: Display basic information about the dataset, including data types, the number of non-null values, and the first few rows of the dataset.
Missing Values: Check for any missing values in the dataset.
Summary Statistics: Provide summary statistics for numerical and categorical features.
Visualize Distributions: Visualize the distributions of numerical features using histograms.
Feature Relationships: Explore relationships between features using correlation matrices and pair plots.
Target Variable Exploration: Analyze the distribution of the target variable, G3, and its relationships with other features.
Handle Missing Values and Encode Categorical Features: Fill missing values and convert categorical features to numerical using one-hot encoding.
Correlation with Target Variable: Identify features most correlated with the target variable.
Feature Importance: Train a Random Forest model and determine feature importance.
Outlier Detection: Detect outliers in the dataset using boxplots.





Data Cleaning
This script performs comprehensive data preprocessing and cleaning on the student dataset. 
The steps include loading the dataset, handling missing values, encoding categorical variables, removing outliers, normalizing numerical features, and splitting the data into training and testing sets. 
Each step is critical for preparing the data for further analysis and machine learning modeling. Below is a detailed walkthrough of the process:

Loading the Dataset: Fetches the student dataset from the UCI repository and loads it into a pandas DataFrame.
Initial Data Exploration: Provides basic information about the dataset, including the first few rows and summary statistics for both numerical and categorical features.
Handling Missing Values: Identifies and imputes missing values in numerical features using the mean value.
Encoding Categorical Variables: Converts categorical variables to numerical form using one-hot encoding.
Outlier Removal: Identifies and removes outliers using the Interquartile Range (IQR) method.
Normalizing Numerical Features: Normalizes the numerical features to ensure all features contribute equally to the analysis.
Splitting the Dataset: Splits the dataset into training and testing sets for model evaluation.
Saving the Cleaned Dataset: Saves the cleaned and preprocessed dataset to a CSV file for further analysis.






Data Analysis
This code loads the cleaned dataset, 
performs data preprocessing, 
trains two machine learning models (Logistic Regression and Random Forest), 
evaluates the models, and analyzes feature importance using the Random Forest model. 

The steps include loading the dataset, 
binarizing the target variable, 
splitting the dataset, 
training the models, 
evaluating the models, 
performing feature importance analysis with corresponding visualizations.





Bias Detection
We use a Random Forest classifier to predict a binarized version of the target variable, G3 (student performance), and then evaluates the model's performance and fairness across different age groups. 
The fairness metrics include demographic parity difference, equalized odds difference, and various rate differences (e.g., false positive rate difference). 
Additionally, the code provides visualizations for accuracy, precision, recall, F1 score, and other metrics by age group.
This is all plotted with confusion matrixes, heatmaps and bar charts and boxplots for better visualization.




Bias Mitigation
This code demonstrates the process of detecting and mitigating bias in a dataset using various techniques, including reweighing, adversarial debiasing, and post-processing. 
The goal is to assess the fairness of a model's predictions and apply different strategies to improve fairness while maintaining model performance.

Steps:
Data Preparation: Load and preprocess the dataset, binarize the target variable, and discretize sensitive features.
Train a Base Model: Train a Random Forest classifier and evaluate its performance and fairness metrics.
Reweighing: Apply sample weights to address class imbalance and retrain the model.
Adversarial Debiasing: Use adversarial techniques to debias the model.
Post-Processing: Apply post-processing techniques to adjust model predictions for fairness.
Fairness Metrics Calculation: Calculate various fairness metrics to compare the performance and fairness of all models.
Visualization: Visualize the results using plots and heatmaps to better understand the impact of different debiasing techniques.


Conclusion and Takeaway of the results here:

in Short:
Best Overall Performance: The Reweighed Model shows the best overall performance metrics but has higher disparities in fairness metrics.
Best Fairness: The Adversarial Debiasing Model shows the best results in terms of fairness metrics but has lower overall performance metrics.
Trade-Off: There is a clear trade-off between optimizing for fairness and optimizing for performance. 
Choosing the right model depends on whether fairness or performance is prioritized in the specific context.
Base and Post-Processing Models: These models show moderate results in both performance and fairness, offering a balanced approach without being the best in either category.

in more Detail:

Fairness Metrics
Demographic Parity Difference: The adversarial debiasing model shows the lowest demographic parity difference, indicating it performs slightly better in ensuring fairness across different demographic groups.
Equalized Odds Difference: The adversarial debiasing model also performs the best in terms of equalized odds difference, followed by the base and post-processing models, while the reweighed model performs the worst.
False Positive Rate Difference: All models show similar performance with slight differences, indicating a consistent level of fairness in false positive rates.
False Negative Rate Difference: This metric is not applicable or was not observed, suggesting a need for further investigation.
Selection Rate Difference: The adversarial debiasing model shows a slightly lower difference, indicating better fairness in the selection rate compared to other models.
False Omission Rate Difference: The reweighed model performs the best with the lowest difference, while the adversarial debiasing model shows the highest difference.
True Negative Rate Difference: All models show similar performance, indicating consistent fairness in true negative rates.

Heatmaps and Visualizations
Comparison Heatmap: The comparison heatmap shows that the adversarial debiasing model generally performs better in terms of fairness metrics compared to other models.
Fairness Metrics Heatmap: This heatmap highlights that the reweighed model has the highest equalized odds difference, while the adversarial debiasing model has a higher false omission rate difference.

Reweighing Model: The reweighed model improves overall performance metrics but does not significantly enhance fairness in all aspects. It performs the best in reducing false omission rate difference but struggles with equalized odds difference.
Adversarial Debiasing Model: This model shows the most improvement in terms of fairness, particularly in demographic parity and equalized odds. However, this comes at the cost of a decrease in overall performance metrics.
Post-Processing Model: The post-processing model offers a balanced approach, maintaining similar performance metrics to the base model while slightly improving fairness metrics.
Base Model: The base model serves as a benchmark, showing reasonable performance and fairness metrics without any specific bias mitigation strategies.
