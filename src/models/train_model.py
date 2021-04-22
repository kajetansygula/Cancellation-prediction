from src.features.preprocessing import preprocess_data, isolationforest_detect

from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

# Reading dataframe
# def build_model:

df = pd.read_csv("../../data/city_dataframe.csv")

# Removing 'reservation_status' column to avoid 'cheating'
df.drop('reservation_status', axis=1, inplace=True)

# Setting up X and y
y = df.iloc[:, 0]
X = df.iloc[:, 1:]

# Preprocessing data
# Removing outliers
isolationforest_detect(X, ['adr', 'days_in_waiting_list'], 0.05, 'combined_outliers')
X.drop('combined_outliers', inplace=True, axis=1)

# Running preprocessing pipeline
X = preprocess_data(X)

"""
Models:
DecisionTreeClassifier
RandomForestClassifier
LogisticRegression
XGBClassifier
Gaussian Naive Bayes
SVC
MLPClassifier

To check:
KFold X
cross_val_score X
accuracy_score X
confusion_matrix X
GridSearch - looking for the best parameters
ROC
"""
log = LogisticRegression()


def evaluate_model(X, y, algorithm, algorithm_name='Algorithm', kfolds=10):

    # Investigating cross validation accuracy score
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    print('Average cross validation accuracy score of {}:  {}'
          .format(algorithm_name, round(np.mean(cross_val_score(algorithm, X, y, cv=kf, scoring="accuracy")),3)))

    # Setting train/test sets
    # X_train, X_test, y_train, y_test = train_test_split(
    # X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fitting the model
    model = algorithm.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Printing model scores
    print('Accuracy Score: {}\nConfusion Matrix:\n {}'
          .format(accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred)))
    print('Classification report of {}: \n{}'.format(algorithm_name, classification_report(y_test, y_pred)))

    # Plotting the ROC curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    sns.set()
    plt.figure()
    plt.plot(fpr, tpr, color='red',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (KFolds)')
    plt.legend(loc="lower right")
    plt.show()

# Testing
evaluate_model(X, y, log, 'Logistic Regression', 4)
