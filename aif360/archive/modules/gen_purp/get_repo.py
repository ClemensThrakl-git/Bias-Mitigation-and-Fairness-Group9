import os

from ucimlrepo import fetch_ucirepo

import pandas as pd


def get_uci_repo(main_path: os.PathLike, data_name: str, data_id: int, data_rep: str = 'data') -> tuple:
    ''' Helper function to either get a dataset from the uci repository and store it locally, 
        or get the locally stored data, if already present.

    params:
        main_path: Current main path of the running environment
        data_name:  Name of the files to store the data in (one file for features
                    one file for targets)
        data_id: Id of the uci repo
        data_rep:   Name of the repository the data can be stored/found
                    (relative to the main path)

    returns:
        Tuple containing the features as dataframe in position 0,
        and the targets as dataframe in position 1
    
    '''
    # Check if the data is already downloaded
    data_path = os.path.join(main_path, data_rep)

    feat_name = data_name + '__features.csv'
    target_name = data_name + '__targets.csv'

    feat_path = os.path.join(data_path, feat_name)
    target_path = os.path.join(data_path, target_name)

    # Fetch dataset 
    if all([os.path.isfile(feat_path), os.path.isfile(target_path)]):
        features = pd.read_csv(feat_path)
        targets = pd.read_csv(target_path)
    else:

        communities_and_crime_unnormalized = fetch_ucirepo(id=data_id)    
        features: pd.DataFrame = communities_and_crime_unnormalized.data.features 
        targets = communities_and_crime_unnormalized.data.targets

        features.to_csv(feat_path)
        targets.to_csv(target_path)


    return features, targets