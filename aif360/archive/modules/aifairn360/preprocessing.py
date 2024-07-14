
from aif360.algorithms.preprocessing import DisparateImpactRemover, OptimPreproc, Reweighing
from aif360.datasets import StandardDataset


class ExecPreprocess:
    ''' Class to execute the preprocecsing phase of aif360.
    The class is set up as builder to allow method chaining.

    params:
        dataset: The dataset instance to operate on
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


    def exec_disparate_impact_remover(self, 
                                    sensitive_attribue: str = '', 
                                    transform: bool | None = None, 
                                    predict: bool | None = None,
                                    to_predict: StandardDataset | None = None
                                    ) -> object:
        ''' Method to execut e the DisarateImpactRemover on the chosen sensitive attribute
        '''
        # Handle default values
        if transform is None:
            transform = self.transform
        if predict is None:
            predict = self.predict

        # Guard clause to see if values for potential predictions have been provided
        if predict and to_predict is None:
            raise ValueError('"Predict" is set to true, but no test values were provided')

        # Calculate the result
        direm = DisparateImpactRemover(sensitive_attribute=sensitive_attribue)
        fitted = direm.fit(self.ds)

        # Transform the dataset & predict some provided data, if configurated so
        transformed = direm.fit_transform(self.ds) if transform else None
        predicted = direm.fit_predict(to_predict) if predict else None

        # Store the result, as well as the instance used to calculate said result
        res = {f'disparate_impact_remover_{self.ml_tracker}': {
                'mitig_inst': direm,
                'fitted': fitted,
                'transformed': transformed,
                'predicted': predicted
                }
            }

        self.ml_tracker += 1
        self.result.update(res)

        return self
    
    
    def exec_optim_preproc(self, 
                            optimizer: object,
                            optim_options: dict,
                            unpriviledged_groups: dict | None = None,
                            priviledged_roups: dict | None = None,
                            transform: bool | None = None, 
                            predict: bool | None = None,
                            to_predict: StandardDataset | None = None
                            ) -> object:
        ''' Method to execut e the DisarateImpactRemover on the chosen sensitive attribute
        '''
        # Handle default values
        if transform is None:
            transform = self.transform
        if predict is None:
            predict = self.predict

        # Guard clause to see if values for potential predictions have been provided
        if predict and to_predict is None:
            raise ValueError('"Predict" is set to true, but no test values were provided')
    
        # Calculate the result
        optim = OptimPreproc(optim_options=optim_options, optimizer=optimizer, 
                             privileged_groups=priviledged_roups, unprivileged_groups=unpriviledged_groups)
        fitted = optim.fit(self.ds)

        # Transform the dataset & predict some provided data, if configurated so
        transformed = fitted.transform(self.ds) if transform else None
        predicted = fitted.predict(to_predict) if predict else None

        # Store the result, as well as the instance used to calculate said result
        res = {f'optim_preproc_{self.ml_tracker}': {
                'mitig_inst': optim,
                'fitted': fitted,
                'transformed': transformed,
                'predicted': predicted
                }
            }

        self.ml_tracker += 1
        self.result.update(res)

        return self
    

    def exec_reweighing(self, 
                        priveledged: list, 
                        unpriviledged: list, 
                        transform: bool | None = None, 
                        predict: bool | None = None,
                        to_predict: StandardDataset | None = None
                        ) -> object:
        ''' Method to execut e the DisarateImpactRemover on the chosen sensitive attribute '''
        # Handle default values
        if transform is None:
            transform = self.transform
        if predict is None:
            predict = self.predict

        # Guard clause to see if values for potential predictions have been provided
        if predict and to_predict is None:
            raise ValueError('"Predict" is set to true, but no test values were provided')

        rew = Reweighing(unprivileged_groups=unpriviledged,
                           privileged_groups=priveledged)
        fitted = rew.fit(self.ds)

        # Transform the dataset & predict some provided data, if configurated so
        transformed = fitted.transform(self.ds) if transform else None
        predicted = rew.fit_predict(to_predict) if predict else None

        # Store the result, as well as the instance used to calculate said result
        res = {f'reweighing_{self.ml_tracker}': {
                'mitig_inst': rew,
                'fitted': fitted,
                'transformed': transformed,
                'predicted': predicted
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