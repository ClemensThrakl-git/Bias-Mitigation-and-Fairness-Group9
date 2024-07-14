import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron


class MlModelProvider:
    ''' A class that stores different ML models that can be imported on command
    Each model is set up to train a model based on the input provided, and to
    return the trained model. In addition, this class is a builder to allow fast
    method chaining to train multiple models on the same input.
    The results are stored in an attribute called "result" and are returned to the
    in the build method, either as a standalone or with all trained models.

    train_feat: Features to train on
    train_targets: Targets to predict during training.
    '''


    def __init__(self, train_feat: pd.DataFrame, train_targ: pd.DataFrame | pd.Series) -> None:
        self.train = train_feat
        self.targ = train_targ

        self.ml_tracker = 0
        self.result = {}


    def random_forest(self, 
                       already_predict: bool = False,
                       test_feat: pd.DataFrame | None = None,
                       **rf_kwawrgs
                       ) -> object:
        ''' Method to train a Random Forest. Either just the training process
        can be done, or predictions can also be made. If predictions are 
        to be made, the features of the test dataset have to be provided as well.

        params:
            already_predicts: Configurates, if predictions are already to be made
            test_feat: Features to predict, if predictions are to be made

        returns:
            The current instance (self)
        '''
        # Guard clause for predict
        if already_predict and test_feat is None:
            raise ValueError(f'Predict is set to "True", but no feature dataset was provided')
        
        # Train the model
        rf = RandomForestClassifier(**rf_kwawrgs).fit(self.train, self.targ)
        # If set, make predictions
        predicted = rf.predict(test_feat)if already_predict else None

        # Store results & instance 
        res = {f'random_forest_{self.ml_tracker}': {
            'trained_model': rf,
            'prediction': predicted
            }
        }
        # Append the results & update the tracker
        self.ml_tracker += 1
        self.result.update(res)

        return self
    

    def perceptron(self, 
                    already_predict: bool = False,
                    test_feat: pd.DataFrame | None = None,
                    **pc_kwargs) -> object:
        ''' Method to train a Perceptron. Either just the training process
        can be done, or predictions can also be made. If predictions are 
        to be made, the features of the test dataset have to be provided as well.

        params:
            already_predicts: Configurates, if predictions are already to be made
            test_feat: Features to predict, if predictions are to be made

        returns:
            The current instance (self)

        '''
        # Guard clause for predict
        if already_predict and test_feat is None:
            raise ValueError(f'Predict is set to "True", but no feature dataset was provided')
        
        # Train the model
        pc = Perceptron(**pc_kwargs).fit(self.train, self.targ)
        # If set, make predictions
        predicted = pc.predict(test_feat) if already_predict else None

        res = {f'perceptron_{self.ml_tracker}': {
            'trained_model': pc,
            'prediction': predicted
            }
        }
        # Append the results & update the tracker
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