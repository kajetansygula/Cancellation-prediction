
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def roc_chart(X_test, y_test, model):

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    sns.set()
    plt.figure()
    plt.plot(fpr, tpr, color='red',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (KFolds)')
    plt.legend(loc="lower right")
    plt.show()


def random_forest_parameters_chart(X, y):
    """
    Found there: https://www.kaggle.com/creepykoala/study-of-tree-and-forest-algorithms
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44)

    plt.figure(figsize=(15, 10))

    # N Estimators
    plt.subplot(3, 3, 1)
    feature_param = range(1, 21)
    scores = []
    for feature in feature_param:
        clf = RandomForestClassifier(n_estimators=feature)
        clf.fit(X_train, y_train)
        scoreCV = clf.score(X_test, y_test)
        scores.append(scoreCV)
    plt.plot(scores, '.-')
    plt.axis('tight')
    # plt.xlabel('parameter')
    # plt.ylabel('score')
    plt.title('N Estimators')
    plt.grid();

    # Criterion
    plt.subplot(3, 3, 2)
    feature_param = ['gini', 'entropy']
    scores = []
    for feature in feature_param:
        clf = RandomForestClassifier(criterion=feature)
        clf.fit(X_train, y_train)
        scoreCV = clf.score(X_test, y_test)
        scores.append(scoreCV)
    plt.plot(scores, '.-')
    # plt.xlabel('parameter')
    # plt.ylabel('score')
    plt.title('Criterion')
    plt.xticks(range(len(feature_param)), feature_param)
    plt.grid();

    # Max Features
    plt.subplot(3, 3, 3)
    feature_param = ['auto', 'sqrt', 'log2', None]
    scores = []
    for feature in feature_param:
        clf = RandomForestClassifier(max_features=feature)
        clf.fit(X_train, y_train)
        scoreCV = clf.score(X_test, y_test)
        scores.append(scoreCV)
    plt.plot(scores, '.-')
    plt.axis('tight')
    # plt.xlabel('parameter')
    # plt.ylabel('score')
    plt.title('Max Features')
    plt.xticks(range(len(feature_param)), feature_param)
    plt.grid();

    # Max Depth
    plt.subplot(3, 3, 4)
    feature_param = range(1, 21)
    scores = []
    for feature in feature_param:
        clf = RandomForestClassifier(max_depth=feature)
        clf.fit(X_train, y_train)
        scoreCV = clf.score(X_test, y_test)
        scores.append(scoreCV)
    plt.plot(feature_param, scores, '.-')
    plt.axis('tight')
    # plt.xlabel('parameter')
    # plt.ylabel('score')
    plt.title('Max Depth')
    plt.grid();

    # Min Samples Split
    plt.subplot(3, 3, 5)
    feature_param = range(1, 21)
    scores = []
    for feature in feature_param:
        clf = RandomForestClassifier(min_samples_split=feature)
        clf.fit(X_train, y_train)
        scoreCV = clf.score(X_test, y_test)
        scores.append(scoreCV)
    plt.plot(feature_param, scores, '.-')
    plt.axis('tight')
    # plt.xlabel('parameter')
    # plt.ylabel('score')
    plt.title('Min Samples Split')
    plt.grid();

    # Min Weight Fraction Leaf
    plt.subplot(3, 3, 6)
    feature_param = np.linspace(0, 0.5, 10)
    scores = []
    for feature in feature_param:
        clf = RandomForestClassifier(min_weight_fraction_leaf=feature)
        clf.fit(X_train, y_train)
        scoreCV = clf.score(X_test, y_test)
        scores.append(scoreCV)
    plt.plot(feature_param, scores, '.-')
    plt.axis('tight')
    # plt.xlabel('parameter')
    # plt.ylabel('score')
    plt.title('Min Weight Fraction Leaf')
    plt.grid();

    # Max Leaf Nodes
    plt.subplot(3, 3, 7)
    feature_param = range(2, 21)
    scores = []
    for feature in feature_param:
        clf = RandomForestClassifier(max_leaf_nodes=feature)
        clf.fit(X_train, y_train)
        scoreCV = clf.score(X_test, y_test)
        scores.append(scoreCV)
    plt.plot(feature_param, scores, '.-')
    plt.axis('tight')
    # plt.xlabel('parameter')
    # plt.ylabel('score')
    plt.title('Max Leaf Nodes')
    plt.grid();

