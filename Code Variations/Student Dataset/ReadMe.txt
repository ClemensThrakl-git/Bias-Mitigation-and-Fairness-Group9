Here are some of the Versions we tested.
The naming shows what/how we tested in that specific notebook.

- 1 Feture: We only observed a single feature as sensitive feature
Combined Feature: We Looked at 2 or more features at once and applied both as a group for the mitigation
Filtered: Since we had 3 Target variables (G1-3) we filtered out 1+2 and only observed G3.
Unfiltered: We observed G3 but kept G1-2 in the data.
Raw_data: We used the data-set without performing data cleaning on it.
Iterative: As a test we always used the results of the previous mitigation calculation for the following calcuation (Debiasing calculated on the results of reweighing for example)
More Visualizations: Just more visualizations to better understand and compare the results