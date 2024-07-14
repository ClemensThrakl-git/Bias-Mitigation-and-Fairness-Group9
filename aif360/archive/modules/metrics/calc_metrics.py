from typing import Iterable

import sklearn
import pandas as pd
from aif360.metrics import ClassificationMetric
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets.structured_dataset import StructuredDataset


def calc_metr(actual: pd.Series, predicted: pd.Series, to_calc: Iterable | None = None, print_results: bool = True) -> dict:
    ''' Orchestration function to calculate the names of metrics provided by "to_calc".
    If set to None, every available metric will be calculated 
    
    params:
        actual: Actual target values
        predicted: The predictions made
        to_cal: Iterable containing the performance mterics to calculate
        print_results: Defines if the results shall be printed to the console
    
    returns:
        A dictionary containing the calculated results
    '''
    # Store all ml performance calculation functions
    avail_metrics = {'ml_perf_metrics': _calc_ml_perf_metrics}
    to_calc = {k for k in avail_metrics} if to_calc is None else to_calc
    results = {}
    
    # Calculate the results
    for met in to_calc:
        results.update(avail_metrics[met](predicted=predicted, actual=actual, print_results=print_results))

    return results


def _calc_ml_perf_metrics(predicted, actual, print_results: bool = True) -> None:
    ''' Helper to calculate all metrics concerning the machine learning performance.

    params:
        actual: Actual target values
        predicted: The predictions made
        to_cal: Iterable containing the performance mterics to calculate
        print_results: Defines if the results shall be printed to the console
    
    returns:
        A dictionary containing the calculated results
    
    '''
    if print_results:
        print(f'Precision: {sklearn.metrics.precision_score(actual, predicted, average="weighted", zero_division=True)}')
        print(f'Accuracy: {sklearn.metrics.accuracy_score(actual, predicted)}')
        print(f'F1-Score: {sklearn.metrics.f1_score(actual, predicted, average="weighted",)}')
        print(f'Recall: {sklearn.metrics.recall_score(actual, predicted, average="weighted", zero_division=True)}')

    return {
        'Precision': sklearn.metrics.precision_score(actual, predicted, average="weighted", zero_division=True),
        'Accuracy': sklearn.metrics.accuracy_score(actual, predicted),
        'F1-Score': sklearn.metrics.f1_score(actual, predicted, average="weighted"),
        'Recall': sklearn.metrics.recall_score(actual, predicted, average="weighted", zero_division=True)
    }


def calc_aif360_bin_label_metrics(dataset: StructuredDataset,
                            priviledged: dict,
                            unpriviledged: dict,
                            print_results: bool = True) -> dict:
    ''' Calculates the bias metrics as suggested by the bias360 framework.

    params:
        dataset: The dataset to evaluate
        priviledged: The priviledged group
        unpriviledged: The unpriviledged group
        print_results: Defines if the results shall be printed to the console
    
    returns:
        A dictionary containing the calculated results
    '''
    bin_label_metr = BinaryLabelDatasetMetric(dataset=dataset, 
                                            unprivileged_groups=unpriviledged,
                                            privileged_groups=priviledged)

    if print_results:
        print("Mean difference = %f" % bin_label_metr.mean_difference())
        print("Consistency = %f" % bin_label_metr.consistency())
        print("Disparate impact = %f" % bin_label_metr.disparate_impact())
    
    return {
        "Mean difference": bin_label_metr.mean_difference(),
        "Consistency": bin_label_metr.consistency(),
        "Disparate impact": bin_label_metr.disparate_impact(),
    }


def calc_aif360_bias_metrics(original_ds: StructuredDataset, 
                            mitigated: StructuredDataset,
                            priviledged: dict,
                            unpriviledged: dict,
                            print_results: bool = True) -> dict:
    ''' Calculates the bias metrics as suggested by the bias360 framework.
    
        params:
        original_ds: The base dataset before mitigation techniques were applied
        mitigated: The base dataset afer mitigation techniques were applied
        priviledged: The priviledged group
        unpriviledged: The unpriviledged group
        print_results: Defines if the results shall be printed to the console
    
    returns:
        A dictionary containing the calculated results
    '''



    classified_metric_debiasing_test = ClassificationMetric(dataset=original_ds, 
                                                    classified_dataset=mitigated,
                                                    unprivileged_groups=unpriviledged,
                                                    privileged_groups=priviledged)

    TPR = classified_metric_debiasing_test.true_positive_rate()
    TNR = classified_metric_debiasing_test.true_negative_rate()
    bal_acc_debiasing_test = 0.5*(TPR+TNR)

    if print_results:
        print("Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
        print("Balanced classification accuracy = %f" % bal_acc_debiasing_test)
        print("Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())
        print("Equal opportunity difference = %f" % classified_metric_debiasing_test.equal_opportunity_difference())
        print("Average odds difference = %f" % classified_metric_debiasing_test.average_odds_difference())
        print("Theil_index = %f" % classified_metric_debiasing_test.theil_index())
    
    return {
        "Classification accuracy": classified_metric_debiasing_test.accuracy(),
        "Balanced classification accuracy": bal_acc_debiasing_test,
        "Disparate impact": classified_metric_debiasing_test.disparate_impact(),
        "Equal opportunity difference": classified_metric_debiasing_test.equal_opportunity_difference(),
        "Average odds difference": classified_metric_debiasing_test.average_odds_difference(),
        "Theil_index": classified_metric_debiasing_test.theil_index()
    }