#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:32:48 2021

@author: Kajetan
"""

def preprocess_data(X):
    
    import pandas as pd
    
    from sklearn.pipeline import Pipeline
    from sklearn.base import TransformerMixin, BaseEstimator
    from sklearn.preprocessing import (StandardScaler, OneHotEncoder)
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
        
    class ColumnRemover(TransformerMixin,BaseEstimator):
    
        def __init__(self, maximum_missing):
            self.maximum_missing = maximum_missing
        
        def fit(self, X, y=None):
            X_ = X.copy()
            self.columns_to_remove = []
            for col in X_.columns:
                if (X_[col].isnull().sum()/len(X_)) >= self.maximum_missing:
                    self.columns_to_remove.append(col)
            return self
        
        def transform(self, X):
            return X.drop(self.columns_to_remove,axis=1)
        
    

    num_pipeline = Pipeline([
            ('column_remover', ColumnRemover(0.1)),
            ('imputer', SimpleImputer(strategy='median')),
            ('std_scaler', StandardScaler()) 
    ])
    
    cat_pipeline = Pipeline([
            ('column_remover', ColumnRemover(0.1)),
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('OneHot_encoder', OneHotEncoder(sparse=False,handle_unknown='ignore',drop='first'))
    ])

    preprocessor = ColumnTransformer([
            ('numerical', num_pipeline, X.select_dtypes(exclude='object').columns),
            ('categorical', cat_pipeline, X.select_dtypes('object').columns)
    ])

    return pd.DataFrame(preprocessor.fit_transform(X))

"""
# Importing dataset
df = pd.read_csv("hotel_bookings.csv")

# Reading data only for 'City' hotel
city_df = df[df['hotel'] == 'City Hotel'].reset_index(drop=True).drop('hotel',axis=1)

# Removing 'reservation_status' column to avoid 'cheating'
city_df.drop('reservation_status', axis=1, inplace=True)

# Setting up X and y
y = city_df.iloc[:,0]
X = city_df.iloc[:,1:]

test = preprocess_data(X)
"""