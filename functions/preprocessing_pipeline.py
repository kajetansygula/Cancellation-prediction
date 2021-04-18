"""
Created on Sun Apr 11 16:32:48 2021

@author: Kajetan
"""


def preprocess_data(x):
    
    import pandas as pd
    
    from sklearn.pipeline import Pipeline
    from sklearn.base import TransformerMixin, BaseEstimator
    from sklearn.preprocessing import (StandardScaler, OneHotEncoder)
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
        
    class ColumnRemover(TransformerMixin, BaseEstimator):
    
        def __init__(self, maximum_missing):
            self.maximum_missing = maximum_missing
            self.columns_to_remove = []
        
        def fit(self, x):
            x_ = x.copy()
            for col in x_.columns:
                if (x_[col].isnull().sum()/len(x_)) >= self.maximum_missing:
                    self.columns_to_remove.append(col)
            return self
        
        def transform(self, x):
            return x.drop(self.columns_to_remove, axis=1)

    num_pipeline = Pipeline([
            ('column_remover', ColumnRemover(0.1)),
            ('imputer', SimpleImputer(strategy='median')),
            ('std_scaler', StandardScaler()) 
    ])
    
    cat_pipeline = Pipeline([
            ('column_remover', ColumnRemover(0.1)),
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('OneHot_encoder', OneHotEncoder(sparse=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
            ('numerical', num_pipeline, x.select_dtypes(exclude='object').columns),
            ('categorical', cat_pipeline, x.select_dtypes('object').columns)
    ])

    return pd.DataFrame(preprocessor.fit_transform(x))


def isolationforest_detection(dataframe, columns, contamination, outlier_name='Outliers'):

    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(max_samples='auto', random_state=1, contamination=contamination)

    if type(columns) is list:
        preds = clf.fit_predict(dataframe[columns])
    else:
        preds = clf.fit_predict(dataframe[columns].values.reshape(-1, 1))

    dataframe[outlier_name] = preds

    print(dataframe[outlier_name].value_counts())


test = 1