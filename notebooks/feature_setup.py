#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:53:30 2021

@author: Kajetan
"""
import pandas as pd
from src.features.preprocessing_pipeline import preprocess_data



from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import (StandardScaler, OneHotEncoder)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression # for testing
from sklearn.model_selection import train_test_split # for testing

import pandas as pd
# Importing dataset
df = pd.read_csv("src/data/hotel_bookings.csv")
# Reading data only for 'City' hotel
#city_df = df[df['hotel'] == 'City Hotel'].reset_index(drop=True).drop('hotel',axis=1)
# Removing 'reservation_status' column to avoid 'cheating'
#city_df.drop('reservation_status', axis=1, inplace=True)
# Setting up X and y
#y = city_df.iloc[:,0]
#X = city_df.iloc[:,1:]

#test = preprocess_data(X)
#print('finished.')
"""
# Classes



# Transformer to remove columns with equal of higher of specified % of missing columns
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
    

# Dropping columns with to many missing values
# Filling missing values
# Removing outliers 
# Standarization
        
# Importing dataset
df = pd.read_csv("src/data/hotel_bookings.csv")

# Reading data only for 'City' hotel
city_df = df[df['hotel'] == 'City Hotel'].reset_index(drop=True).drop('hotel',axis=1)

# Removing 'reservation_status' column to avoid 'cheating'
city_df.drop('reservation_status', axis=1, inplace=True)

# Setting up X and y
y = city_df.iloc[:,0]
X = city_df.iloc[:,1:]

# Pipeline for numeric variables
num_pipeline = Pipeline([
    ('column_remover', ColumnRemover(0.1)),
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()) 
])


# Pipeline for categorical variables
cat_pipeline = Pipeline([
    ('column_remover', ColumnRemover(0.1)),
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('OneHot_encoder', OneHotEncoder(sparse=False,handle_unknown='ignore'))
])


# Final column tranformer
preprocessor = ColumnTransformer([
    ('numerical', num_pipeline, X.select_dtypes(exclude='object').columns),
    ('categorical', cat_pipeline, X.select_dtypes('object').columns)
])

#num_transformed = pd.DataFrame(preprocessor.fit_transform(X))
    
# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)    

# Defyining the final pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])    
    
    
clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))

"""
# Removing outliers 
# Standarization

# Saving as a pickle


