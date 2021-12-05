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

# getting the data

dirname = os.path.dirname(__file__)
os.makedirs(os.path.join(dirname, "../report-resources/minst"), exist_ok=True)
train = pd.read_csv(os.path.join(dirname, "../all-data/minst/train.csv"))
y_train, x_train = train.label.values, train.drop("label", axis=1).values

# plot the sample image

plt.imshow(x_train[0].reshape(28, 28), interpolation="gaussian")


plt.savefig(os.path.join(dirname, "../report-resources/minst/simple-minst-sample-1.pdf"))  


# doing the training and validation split

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train)

# fit LogisticRegression classifiers and observe their performance without mis-labeling.

lg_clf = LogisticRegression(solver='lbfgs', max_iter=1000)
lg_clf.fit(x_train, y_train)
logistic_regression_validate_predict = lg_clf.predict(x_validate)
logistic_regression_f1_score = f1_score(
    y_validate, 
    logistic_regression_validate_predict, 
    average='weighted')

print("logistic_regression_f1_score")
print(logistic_regression_f1_score)

# fit DecisionTreeClassifier classifiers and observe their performance without mis-labeling.

decision_tree_clf = tree.DecisionTreeClassifier()
decision_tree_clf.fit(x_train, y_train)
decision_tree_validate_predict = decision_tree_clf.predict(x_validate)
decision_tree_f1_score = f1_score(
    y_validate, 
    decision_tree_validate_predict, 
    average='weighted')

print("decision_tree_f1_score")
print(decision_tree_f1_score)

# fit RandomForestClassifier classifiers and observe their performance without mis-labeling.

random_forest_clf = RandomForestClassifier(n_jobs=-1)
random_forest_clf.fit(x_train, y_train)
random_forest_validate_predict = random_forest_clf.predict(x_validate)
random_forest_f1_score = f1_score(
    y_validate, 
    random_forest_validate_predict, 
    average='weighted')

print("random_forest_f1_score")
print(random_forest_f1_score)

# fit AdaBoostClassifier classifiers and observe their performance without mis-labeling.

from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier()
ada_clf.fit(x_train, y_train)
ada_validate_predict = ada_clf.predict(x_validate)
ada_f1_score = f1_score(y_validate, ada_validate_predict, average='weighted')

print("ada_f1_score")
print(ada_f1_score)

# fit XGBRegressor classifiers and observe their performance without mis-labeling.

import xgboost as xgb
xgb_clf = xgb.XGBRegressor(objective="multi:softmax", num_class=10)
xgb_clf.fit(x_train, y_train)
xgb_validate_predict = xgb_clf.predict(x_validate)
xgb_f1_score = f1_score(y_validate, xgb_validate_predict, average='weighted')

print("xgb_f1_score")
print(xgb_f1_score)

simple_comparison_result = pd.DataFrame(dict(
    logistic_regression=[logistic_regression_f1_score],
    decision_tree=[decision_tree_f1_score],
    random_forest=[random_forest_f1_score],
    ada = [ada_f1_score]
    ,
    xgb = [xgb_f1_score]
    ))

# plotting the f1 scores of all model performance without mis-labeling.

with open(os.path.join(dirname, "../report-resources/minst/f1_scores.tex"), 'w') as f:
    print(simple_comparison_result.to_latex(index=False), file=f)


from do_miss_label import do_miss_label

y_train_miss_label = do_miss_label(y_train, 0.3)
count = 0
for i in range(len(y_train)):
    if y_train[i]!=y_train_miss_label[i]:
        count+=1
print("miss labeled count")
print(count)

# fit LogisticRegression classifiers and observe their performance with mis-labeling.

lg_clf_misslabeled = LogisticRegression(solver='lbfgs', max_iter=1000)
lg_clf_misslabeled.fit(x_train, y_train_miss_label)
logistic_regression_misslabeled_validate_predict = lg_clf_misslabeled.predict(x_validate)
logistic_regression_f1_score_misslabeled = f1_score(y_validate, 
                                        logistic_regression_misslabeled_validate_predict, 
                                        average='weighted')
print("logistic_regression_f1_score_misslabeled")
print(logistic_regression_f1_score_misslabeled)

# fit DecisionTreeClassifier classifiers and observe their performance with mis-labeling.


decision_tree_clf_misslabeled = tree.DecisionTreeClassifier()
decision_tree_clf_misslabeled.fit(x_train, y_train_miss_label)
decision_tree_misslabeled_validate_predict = decision_tree_clf_misslabeled.predict(x_validate)
decision_tree_f1_score_misslabeled = f1_score(y_validate, 
                                  decision_tree_misslabeled_validate_predict, 
                                  average='weighted')

print("decision_tree_f1_score_misslabeled")
print(decision_tree_f1_score_misslabeled)

# fit RandomForestClassifier classifiers and observe their performance with mis-labeling.


random_forest_clf_misslabeled = RandomForestClassifier(n_jobs=-1)
random_forest_clf_misslabeled.fit(x_train, y_train_miss_label)
random_forest_misslabeled_validate_predict = random_forest_clf_misslabeled.predict(x_validate)
random_forest_misslabeled_f1_score = f1_score(y_validate, 
                                              random_forest_misslabeled_validate_predict, 
                                              average='weighted')

print("random_forest_misslabeled_f1_score")
print(random_forest_misslabeled_f1_score)

# fit XGBRegressor classifiers and observe their performance with mis-labeling.


import xgboost as xgb
xgb_clf_misslabeled = xgb.XGBRegressor(objective="multi:softmax", 
    num_class=10, random_state=42)
xgb_clf_misslabeled.fit(x_train, y_train_miss_label)
xgb_validate_predict_misslabeled = xgb_clf_misslabeled.predict(x_validate)
xgb_f1_score_misslabeled = f1_score(y_validate, 
    xgb_validate_predict_misslabeled, 
    average='weighted')

print("xgb_f1_score_misslabeled")
print(xgb_f1_score_misslabeled)

# fit AdaBoostClassifier classifiers and observe their performance with mis-labeling.


from sklearn.ensemble import AdaBoostClassifier
ada_clf_misslabeled = AdaBoostClassifier()
ada_clf_misslabeled.fit(x_train, y_train_miss_label)
ada_validate_predict_misslabeled = ada_clf_misslabeled.predict(x_validate)
ada_f1_score_misslabeled = f1_score(y_validate, ada_validate_predict_misslabeled, average='weighted')
print("ada_f1_score_misslabeled")
print(ada_f1_score_misslabeled)


misslabeled_comparison_result = pd.DataFrame(dict(
    logistic_regression=[logistic_regression_f1_score_misslabeled],
    decision_tree=[decision_tree_f1_score_misslabeled],
    random_forest=[random_forest_misslabeled_f1_score],
    ada = [ada_f1_score_misslabeled]
    ,
    xgb = [xgb_f1_score_misslabeled]
    ))

with open(os.path.join(dirname, "../report-resources/minst/misslabeled_f1_scores.tex"), 'w') as f:
    print(misslabeled_comparison_result.to_latex(index=False), file=f)



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

