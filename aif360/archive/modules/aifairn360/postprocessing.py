from aif360.datasets import StandardDataset
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing


class ExecPostprocess:
    ''' Class to execute the preprocecsing phase of aif360.
    The class is set up as builder to allow method chaining.

    params:
        dataset: The dataset to work with
        transform: If true, the transform method will be called in addition, if available
        predict: If true, the predict method will be called in addition, if available

    Returns:
        self, or result dictionary, depending on the closing method
    '''

    def __init__(self, dataset: StandardDataset, transform: bool = False, predict: bool = True) -> None:
        self.ds = dataset
        self.transform = transform
        self.predict = predict

        self.ml_tracker = 0
        self.result = {}


    def exec_cal_equalized_odds(self, 
                            unpriviledged: dict,
                            priveledged: dict,
                            predicted: StandardDataset
                            ) -> object:
        ''' Method to execut e the DisarateImpactRemover on the chosen sensitive attribute. '''

        # Calculate the equailised odds
        eq_odds = CalibratedEqOddsPostprocessing(unprivileged_groups=unpriviledged,
                                       privileged_groups=priveledged)
        fitted = eq_odds.fit(self.ds, dataset_pred=predicted)

        # Store the result, as well as the instance used to calculate said result
        res = {f'eq_odds_{self.ml_tracker}': {
                'mitig_inst': eq_odds,
                'fitted': fitted,
                }
            }


        self.ml_tracker += 1
        self.result.update(res)

        return self

    
    def return_results(self) -> dict:
        ''' Sort of an alternative to "build" which just returns the stored
        results of this builder instance, rather than the whole instance
        '''
        return self.result
    

    def build(self) -> object:
        ''' Concluded the instance when called by returning self '''
        return self