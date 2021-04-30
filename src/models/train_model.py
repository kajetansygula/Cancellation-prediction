from src.visualization.charts import roc_chart

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

import time

import warnings
warnings.filterwarnings('ignore')


def evaluate_model(X, y, algorithm, kfolds=10, roc=True):
    """
    Prints score of cross-validation, confusion matrix, classification report and optionally ROC curve.
    """

    start_time = time.clock()

    # Investigating cross validation accuracy score
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    cross_val_mean = np.mean(cross_val_score(algorithm, X, y, cv=kf, scoring="accuracy"))
    print('Average cross validation accuracy score of {}:  {}'
          .format(algorithm.__class__.__name__, round(cross_val_mean, 3)))

    print('Cross validation score execution time: {}\n'.format(str(time.clock() - start_time)))

    # Setting train/test sets
    # X_train, X_test, y_train, y_test = train_test_split(
    # X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)

    # Setting train/test sets
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fitting a model
    model = algorithm.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Printing model scores
    print('Confusion Matrix:\n {}'
          .format(confusion_matrix(y_test, y_pred)))
    print('Classification report of {}: \n{}'
          .format(algorithm.__class__.__name__, classification_report(y_test, y_pred)))

    # Plotting a ROC curve
    if roc:
        roc_chart(X_test, y_test, model)

    # Return
    return cross_val_mean


def train_model(X, y, algorithm, params=None, kfolds=10, GridSearchCV=False):

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = algorithm.fit(X_train, y_train)
    return model


