from dataclasses import dataclass
from typing import Iterable, Callable

from aif360.datasets.structured_dataset import StructuredDataset
from aif360.datasets.binary_label_dataset import BinaryLabelDataset
from aif360.datasets.standard_dataset import StandardDataset

import pandas as pd

from sklearn.model_selection import train_test_split


@dataclass(frozen=True, slots=True)
class DatasetContainer:
    ''' This class acts as base container storing all the data inmutably
    in various useful forms. Namely, the data is stored as an dataset instance 
    from aif360, as well as a pandas DataFrame
    '''
    # All Dataset vars for aif360
    ds: StructuredDataset | BinaryLabelDataset
    ds_feat_train: StructuredDataset | BinaryLabelDataset
    ds_feat_test: StructuredDataset | BinaryLabelDataset

    # All pandas DataFrames
    df: pd.DataFrame
    df_feat_train: pd.DataFrame
    df_feat_test: pd.DataFrame
    df_targ_train: pd.DataFrame
    df_targ_test: pd.DataFrame

    # target Series
    target: pd.Series | pd.DataFrame
    

class DatasetInstantiater:
    ''' Responsible for using the input data and creating an 
    instance of the dataset container
    '''
    def __init__(self, 
                train: pd.DataFrame, 
                target: pd.DataFrame | pd.Series,
                prot_attr: str | list,
                priv_identifier: Iterable | Callable,
                priv_cls: Iterable, 
                unpriv_cls: Iterable,  
                priv_label: any = ..., 
                unpriv_label: any = ...,
                categorical_features: Iterable = [],
                feat_to_drop: str | Iterable = [], 
                train_test_split: float = 0.8) -> None:
        
        self.train = train
        self.targ = target
        self.prot = [prot_attr] if not isinstance(prot_attr, list) else prot_attr
        self.prv_cls = [priv_cls]  if priv_cls == ... else priv_cls
        self.unprv_cls = [unpriv_cls]   if unpriv_cls == ... else unpriv_cls
        self.prv_lab = priv_label
        self.unprv_lab = unpriv_label
        self.prv_ident = priv_identifier
        self.categ_f = categorical_features
        self.f_to_drop = feat_to_drop
        self.tts = train_test_split


    def inst_ds_container(self) -> DatasetContainer:
        ''' Orchestration method executing the necessary preparations
        and returning an instance of a DatasetConatiner based on the data
        provided'''

        ds = StandardDataset(
                df=self.train,
                label_name=self.prv_lab,
                favorable_classes=self.prv_cls,
                privileged_classes=self.prv_ident,
                protected_attribute_names=self.prot,
                categorical_features=self.categ_f,
                features_to_drop=self.f_to_drop
            )
        
        ds_train, ds_test = ds.split([self.tts])

        df_feat_train, df_feat_test, df_targ_train, df_targ_test = train_test_split(self.train, self.targ, train_size=self.tts)

        return DatasetContainer(ds=ds, ds_feat_train=ds_train, ds_feat_test=ds_test,
                                df=self.train, target=self.targ, df_feat_train=df_feat_train,
                                df_feat_test=df_feat_test, df_targ_train=df_targ_train,
                                df_targ_test=df_targ_test)

        