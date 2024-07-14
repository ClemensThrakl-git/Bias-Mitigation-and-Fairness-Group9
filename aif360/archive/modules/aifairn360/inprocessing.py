from aif360.datasets import StructuredDataset
from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


class ExecInprocess:
    ''' Class to execute the preprocecsing phase of aif360.
    The class is set up as builder to allow method chaining.

    params:
        dataset: The dataset to work with
        transform: If true, the transform method will be called in addition, if available
        predict: If true, the predict method will be called in addition, if available

    Returns:
        self, or result dictionary, depending on the closing method
    '''

    def __init__(self, dataset: StructuredDataset, transform: bool = False, predict: bool = True) -> None:
        self.ds = dataset
        self.transform = transform
        self.predict = predict

        self.ml_tracker = 0
        self.result = {}


    def exec_advesarial_debiasing(self, 
                                unpriviledged: dict,
                                priveledged: dict,
                                transform: bool | None = None, 
                                predict: bool | None = None,
                                to_predict: StructuredDataset | None = None
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
        
        # Start the tensorflow sesssion
        tf.reset_default_graph()
        sess = tf.Session()

        adv_deb = AdversarialDebiasing(privileged_groups = priveledged,
                          unprivileged_groups = unpriviledged,
                          scope_name='adv_classifier',
                          debias=True,
                          sess=sess)
        # direm = DisparateImpactRemover(sensitive_attribute=sensitive_attribue)

        fitted = adv_deb.fit(self.ds)

        # Predict the provided data, if configurated so
        transformed = None
        predicted = fitted.predict(to_predict)

        # Store the result, as well as the instance used to calculate said result
        res = {f'adveserial_debiasing_{self.ml_tracker}': {
                'mitig_inst': adv_deb,
                'fitted': fitted,
                'transformed': transformed,
                'predicted': predicted
                }
            }


        self.ml_tracker += 1
        self.result.update(res)

        # Stop the tensorflow session
        sess.close()

        return self

    
    def return_results(self) -> dict:
        ''' Sort of an alternative to "build" which just returns the stored
        results of this builder instance, rather than the whole instance
        '''
        return self.result
    

    def build(self) -> object:
        ''' Concluded the instance when called by returning self '''
        return self