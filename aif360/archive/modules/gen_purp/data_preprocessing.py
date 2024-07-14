import pandas as pd


def preprocessing(data: tuple) -> tuple:
    ''' Helper function to preprocess a dataframe by a pre-defined procedure.
    Essentially gets the tuple with features and targets, infers more precise
    types, if possible, drops NaNs, and converts categorical data to dummies.
    (Note: This function has nothing to do with the bias preprocessing techniques)

    params:
        data: Tuple containing the features on position 0 and the targets on position 1

    returns:
        Tuple containing the preprocessed features on position 0 and the targets on position 1
    '''
    features, targets = data[0], data[1]
    features: pd.DataFrame
    # Convert data to more precise types where possible
    features = features.infer_objects()
    features = features.map(lambda x: 1 if isinstance(x, object) and x == 'yes' else 0 if isinstance(x, object) and x == 'no' else x)
    # Drop NaNs
    feat_na_dropped = features.dropna(how='any', axis=1)
    targets_na_dropped: pd.DataFrame = targets.dropna(how='any', axis=1)
    # Get the dummies
    feat_encoded = pd.get_dummies(feat_na_dropped)

    return feat_encoded, targets_na_dropped