from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
from sklearn import tree

import numpy as np
import random
def do_miss_label(y_train_labels, miss_labeled_ratio):
    y_train_copy = np.copy(y_train_labels)
    all_labels = set(y_train_copy)
    all_labels_count = len(all_labels)
    misslabeled_indices = np.random.choice(len(y_train_copy)-1,
                                           int((len(y_train_copy)-1)*miss_labeled_ratio))
    for misslabeled_index in misslabeled_indices:
        y_train_copy[misslabeled_index] = random.randrange(0,all_labels_count)
    return y_train_copy

train = pd.read_csv("wine/wine.csv")
y_train, x_train = train.Type.values, train.drop("Type", axis=1).values
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train)

from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier()
ada_clf.fit(x_train, y_train)
ada_validate_predict = ada_clf.predict(x_validate)
ada_f1_score = f1_score(y_validate, ada_validate_predict, average='weighted')
print("ada_f1_score")
print(ada_f1_score)

y_train_miss_label = do_miss_label(y_train, 0.3)

from sklearn.ensemble import AdaBoostClassifier
ada_clf_misslabeled = AdaBoostClassifier()
ada_clf_misslabeled.fit(x_train, y_train_miss_label)
ada_validate_predict_misslabeled = ada_clf_misslabeled.predict(x_validate)
ada_f1_score_misslabeled = f1_score(y_validate, ada_validate_predict_misslabeled, average='weighted')
print("ada_f1_score_misslabeled")
print(ada_f1_score_misslabeled)