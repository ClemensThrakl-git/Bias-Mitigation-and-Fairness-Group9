# Bias-Mitigation-and-Fairness-Group9

This repository contains all the content regarding the comparison of bias detection- and mitigation techniques by Fairlearn and Aif360.
Said comparison was conducted with two distinct datasets, namely the `Student Performance Dataset` [5](#stud_ds), as well as the `Communities and Crime Dataset` [6](#crime_ds).

Link to Github repository: https://github.com/ClemensThrakl-git/Bias-Mitigation-and-Fairness-Group9
Link to Crime Dataset: https://archive.ics.uci.edu/dataset/211/communities+and+crime+unnormalized
Link to Student Dataset: https://archive.ics.uci.edu/dataset/320/student+performance


## Fairlearn

For the Fairlearn Notebooks always when opening them just run all cells, they are ment to be executed in the corresponding order that the cells are put in. 

The Notebooks already come executed with the results stored so they can be observed without having to run the notebook. But if you want to execute them yourself open them in the corresponding Steps in their labeling (Step 1 - Step 5) and execute them in that order.

Bias Detection and Mitigation Using Microsoft Fairlearn:

This part of the project focuses on bias detection and mitigation using Microsoft Fairlearn, analyzing two datasets: crime statistics and student failure rates. 
The repository is organized to ensure clear documentation and easy navigation.


Repository Structure:

- Fairlearn: This folder contains all the code related to bias detection and mitigation.
- Main Code: The primary scripts used for the analysis. (Code from Start to Finish for one example run)
- Code Variations: Alternative versions of the main code, which run with different variables or parameters. (to get different information and results and insights)
- Old Code: Previous iterations of the code, maintained to document the development progress.
- Analysis of initial Tests: Contains 2 Word Documents that Contain the Results of each individual Step from the first full Testrun and document the findings and our interpretations of the resuls.
- Screenshots of some Preliminary Results: Contains Screenshots of some of the Results from the first Iteration of Code. (Results also contained within corresponding Code Notebooks)

Main Code Folder Structure:

Inside the Main Code folder, you will find two subfolders:

- Student Failures: Contains the scripts for analyzing the student failure dataset.
- Crime: Contains the scripts for analyzing the crime statistics dataset.


Within the Main Code Folder is always one Example for each Data-Set performed through 5 Steps (Data Exploration, Cleaning, Analysis, Bias Detection, Bias Mitigation). 

This is performed for one Case to see a full case example. For more different in depth results and analysis we performed multiple different approaches which are saved within the folder of "Code Variations". 

Those are mostly just different Variations in regards to bias detection and mitigation, but because of changes in the approach the code always had to be adjusted a bit.

Furthermore within the Main Code Folder there is one jupyter notebook file that contains a list of all install statements for each library that is required. Run this once in the beginning to ensure all necessary libraries are installed if you plan on running the code yourself.

Also the "Code Variations" contain same code in form of normal python code (without execution output but therefore small filesize) and as juypter notebooks (with execution output but bigger filesize)

## AiFairness 360

This section covers all content related to the ***aif360*** framework [1](#aif360_ref). This particular readme contains the structure of this subdirectory, the changes made between the development and final versions, and the setup and execution instructions. Please note that there are two versions of approximately the same code. The earlier development versions are stored in the **archive** directory. This version shows the `exact result` found on the final presentation. The code cells in it have not been changed in any way. Conversely, anything not stored in this separate directory is the cleaned up version, allowing for easy understanding and comparison. 

### Structure

Two datasets with two protected variables were used to compare the results. Both completed comparisons can be found in two separate notebooks. Additionally, all previous versions and attempts can be found in the archive. Please note that the contents of the archive have not been re-run to preserve the exact results within the notebooks for full disclosure.

|Name|Short Description|
|--|--|
|`student_ds.ipynb`|Each bias detection & mitigation techniques used on the age and workday alcohol consumption as protected variables|
|`viol_pred_ds.ipynb`|Each bias detection & mitigation techniques used on the as protected variable|
|`archive`|Directory containing all previous version and the graphs displayed in the presentation|

For the archive, in particular the notebooks `stud_age.ipynb` and `viol_pred_eth.ipynb` contain the results used as highlights in the presentation for the aif360 part. Furthermore, some cell output in the legacy notebooks have been cleared, as the warnings displayed information which may not be shared publicly. Emphasis has been placed on the cleaned and structured final versions in the form of the `student_ds.ipynb` and `viol_pred_ds.ipynb` notebooks.

In terms of bias reduction methods, all the options available in **Fairlearn** have been chosen to allow direct comparison, rather than having to interpret similarities in different methods. These are ***Reweighing*** [2](#reweighing) for preprocessing, ***Adversarial Debiasing*** [3](#adv_deb) for inprocessing, and ***Equalised Liberal Odds*** [4](#cal_eq_odds) for postprocessing.

### Changes in the cleaned version

Essentially, the following points will differ from the presentation version:

- The `bias detection' function has been omitted, as the results did not produce an interpretable/comparable result.
- Most of the code originally written in some Python source files will be included directly in the cleaned up notebooks. The actual results should be broadly the same (apart from variations due to the training and predictions of the ml modules). The code was set up for a larger operation comparing dozens of different combinations. However, for the scope of this work, the setup introduces more complexity than is helpful. Consequently, most of the calls have been replaced with manual instantiations of the wrappers originally written to facilitate multiple instantiations.
- The plotted graphs have been greatly extended to show comparisons that were previously planned to be done manually for presentation.
- A huge expansion of the markdown cells to detail each step.


### Setup & Code Execution

This project relies on a fairly large number of external libraries. These libraries need to be installed in the target environment of your choice. Ideally, the environment should be different from the execution environment in which `Fairlearn` runs, to avoid version conflicts (e.g. setting up two different venvs). Regarding the libraries themselves, there are two options.

1. The two main notebooks contain commented ***pip install statements***. These statements can be uncommented to run the installation via pip. Please note that unlike the accompanying ***requirements.txt***, only the libraries/frameworks needed to run the notebooks are present.

2. Installing the libraries is also possible by using the ***requirements.txt*** provided and the pip install command in the console (pip install -r /***{path_of_txt_file}***/requirements.txt). For this option, please note that the requirements file contains **all** libraries that are used to additionally set up and run the Jupyter environment in a local setup (resulting in numerous additional libraries).

After a successful installation, each cell should (hopefully) be able to run on each of the two notebooks.

## Results:

Finally, this section shows a brief summary the results regarding the comparison of the techniques as well as the differences in content.

**Rq:** ***Which key differences in bias detection and mitigation capabilities between Fairlearn and AIF360 when applied to student failure and crime datasets exist?***
 
`Fairlearn`

- Fairlearn is easier applicable​
- Very well suited for bias detection​
- Yields fast results​
- Very well suited for gaining an overview​
- Good results in combination with pre and post-processing​
- In-Processing is a bit challenging
- If training data not adequately represents all sensitive groups, difficult to tweak correctly

`Aif360`

- Limited bias detection​
- More challenging  to set up and get to work propperly​
- Once set up properly, ​more mitigation ​options and  metrics​
- (Api to sklearn)

`Conclusions about the datasets`

**Rq:** ***What correlations can be drawn between biases in student failure rates and crime rates, and how do these correlations inform our understanding of socio-economic factors and systemic inequalities?​***


- Bias is generally more present in the crime dataset, particularly in relation to socio-economic factors.

- The student dataset has high biases, especially for absenteeism and previous failures (personal behaviour).

- Bias reduction techniques reduce these biases in both datasets (at least according to their own metrics).

- However, this is often at the cost of reduced performance of the machine learning modules used.

- The results highlight the trade-offs involved in addressing systemic inequalities and socio-economic disparities, and may display our understanding of the complex interplay between education and crime better.


## References

 [1] Bellamy, R. K., et al (2018).(2019). AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias. IBM Journal of Research and Development, 63(4/5), 4-1. <a name="aif360_ref"></a>

[2] Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. Knowledge and information systems, 33(1), 1-33. <a name="reweighing"></a>

[3] Zhang, B. H., Lemoine, B., & Mitchell, M. (2018, December). Mitigating unwanted biases with adversarial learning. In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society (pp. 335-340). <a name="adv_deb"></a>

[4] Pleiss, G., Raghavan, M., Wu, F., Kleinberg, J., & Weinberger, K. Q. (2017). On fairness and calibration. Advances in neural information processing systems, 30. <a name="cal_eq_odds"></a>

[5] Cortez,Paulo. (2014). Student Performance. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T. <a name="stud_ds"></a>

[6]  Redmond,Michael. (2011). Communities and Crime Unnormalized. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC8X. <a name="crime_ds"></a>
