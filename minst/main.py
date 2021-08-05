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

dirname = os.path.dirname(__file__)
os.makedirs(os.path.join(dirname, "../report-resources/minst"), exist_ok=True)
train = pd.read_csv(os.path.join(dirname, "../all-data/minst/train.csv"))
y_train, x_train = train.label.values, train.drop("label", axis=1).values
plt.imshow(x_train[0].reshape(28, 28), interpolation="gaussian")


plt.savefig(os.path.join(dirname, "../report-resources/minst/simple-minst-sample-1.pdf"))  


x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, random_state=1643)

# Let's first fit simple classifiers and observe their performance.

lg_clf = LogisticRegression(solver='lbfgs', max_iter=1000)
lg_clf.fit(x_train, y_train)
logistic_regression_validate_predict = lg_clf.predict(x_validate)
logistic_regression_f1_score = f1_score(
    y_validate, 
    logistic_regression_validate_predict, 
    average='weighted')
print(logistic_regression_f1_score)

decision_tree_clf = tree.DecisionTreeClassifier()
decision_tree_clf.fit(x_train, y_train)
decision_tree_validate_predict = decision_tree_clf.predict(x_validate)
decision_tree_f1_score = f1_score(
    y_validate, 
    decision_tree_validate_predict, 
    average='weighted')
print(decision_tree_f1_score)

random_forest_clf = RandomForestClassifier(n_jobs=-1)
random_forest_clf.fit(x_train, y_train)
random_forest_validate_predict = random_forest_clf.predict(x_validate)
random_forest_f1_score = f1_score(
    y_validate, 
    random_forest_validate_predict, 
    average='weighted')
print(random_forest_f1_score)

simple_comparison_result = pd.DataFrame(dict(
    logistic_regression=[logistic_regression_f1_score],
    decision_tree=[decision_tree_f1_score],
    random_forest=[random_forest_f1_score]))

with open(os.path.join(dirname, "../report-resources/minst/f1_scores.tex"), 'w') as f:
    print(simple_comparison_result.to_latex(index=False), file=f)

# **Considering the models suggested to build the ensemble:**
# rf_clf = RandomForestClassifier(n_jobs=-1)

# et_clf = ExtraTreesClassifier(n_jobs=-1)

# svm_clf = Pipeline([
#     ("standarize", StandardScaler()),
#     ("svc", SVC(verbose=2))
# ])

# hard_voting_ensemble = VotingClassifier(estimators=[
#     ("random_forest", rf_clf),
#     ("extra_trees", et_clf),
#     ("svm", svm_clf)
# ], voting="hard")

# rf_clf.fit(x_train, y_train)
# et_clf.fit(x_train, y_train)

# svm_clf.fit(x_train, y_train)

# hard_voting_ensemble.fit(x_train, y_train)

# test = pd.read_csv("../input/test.csv").values
# test_predict = hard_voting_ensemble.predict(test)

# test_predict_df = pd.DataFrame({"ImageId": range(1, len(test_predict) + 1),
#                                  "Label": test_predict})
# test_predict_df.index = test_predict_df.ImageId
# test_predict_df.drop("ImageId", axis=1, inplace=True)

# test_predict_df.head()

# test_predict_df.to_csv("MNIST_test_pred.csv", header=True)

