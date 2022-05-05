# ===========================
# Import Libraries
# ===========================
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def load_dataset(filename, train_or_test):
    df = pd.read_csv(filename)
    df = df.set_index('PassengerId')
    if(train_or_test == 'train'):
        df['Transported'] = df['Transported'].astype(int)
        X = df.drop(['Transported'],axis=1)
        y = df['Transported']
        return X, y
    else:
        return df

X, y = load_dataset('../data/data_train.csv', "train")
X_TEST = load_dataset('../data/data_test.csv', "test")

index_submission = X_TEST.index

# =====================
# Split train / val 
# =====================
from sklearn.model_selection import train_test_split

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=0)

# =========================
# Modelling
# =========================
def print_scores(y_true, y_pred):
    print('ROCAUC score:',roc_auc_score(y_true, y_pred))
    print('Accuracy score:',accuracy_score(y_true, y_pred))
    print('F1 score:',f1_score(y_true, y_pred))
    print('Precision score:',precision_score(y_true, y_pred))
    print('Recall:',recall_score(y_true, y_pred))
    print("\n\n")

def run_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(model)
    print_scores(y_test, y_pred)
    return score, y_pred

def tune_model(model, param_grid, n_iter, X_train, y_train):
    grid = RandomizedSearchCV(model, param_grid, verbose=100,
        scoring='accuracy', cv=3, n_iter=n_iter)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    return best_model

def run_on_test(model, X_test, filename):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X)
    X_test =  scaler.transform(X_test)
    print(X_train.shape)
    y_train = y
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    submission = pd.DataFrame({'PassengerId': index_submission ,'Transported': y_pred.astype(bool)},
            columns=['PassengerId', 'Transported'])

    submission.to_csv(filename, index=False)

# feature selection
def KBest(X_train, y_train, X_test, k):
    fs = SelectKBest(score_func=chi2, k=k)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def normalize(X_train, X_test):
    scaler = StandardScaler()
    X_train_scl = scaler.fit_transform(X_train)
    X_test_scl = scaler.transform(X_test)
    return X_train_scl, X_test_scl

# ==========================
# ++++++++++++++++++++++++++
# 
#           MAIN 
#
# ++++++++++++++++++++++++++
# ==========================

# -----------------
# Models With KBest
# -----------------

print("LogisticRegression with KBest: \n")
X_train_fs, X_val_fs, fs = KBest(X_train, y_train, X_val, 33)
X_train_fs, X_val_fs = normalize(X_train_fs, X_val_fs)
lg = LogisticRegression()
score, y_pred = run_model(lg, X_train_fs, y_train, X_val_fs, y_val) 

print("LogisticRegression: \n")
X_train, X_val = normalize(X_train, X_val)
lg = LogisticRegression()
score, y_pred = run_model(lg, X_train, y_train, X_val, y_val) 


# SVC

# # defining parameter range
# param_grid_svc = {'C': [1, 1.5, 2, 3, 5, 10, 12, 20],
#               'gamma': [0.002, 0.003, 0.004, 0.001],
#               'kernel': ['rbf']
# }

# svc = SVC(random_state=32)
# best_svc = tune_model(svc, param_grid_svc, 10, X_train, y_train)
# score, y_pred = run_model(best_svc, X_train, y_train, X_val, y_val)


# n_estimators = [10,100, 300, 500, 800, 1200, 2000]
# max_depth = [5, 8, 15, 25, 30, 35, 40, 50, 80, 100]
# min_samples_split = [2, 5, 10, 15, 100]
# min_samples_leaf = [1, 2, 5, 10] 

# param_grid_rf = dict(n_estimators = n_estimators, 
#             max_depth = max_depth,  
#             min_samples_split = min_samples_split, 
#             min_samples_leaf = min_samples_leaf)

# rf = RandomForestClassifier(random_state = 1)
# best_rf = tune_model(rf, param_grid_rf, 10, X_train, y_train)
# score, y_pred = run_model(best_rf, X_train, y_train, X_val, y_val)

# run_on_test(best_rf, X_train, y_train, X_test, '../submissions/submission_rf.csv')

# import xgboost as xgb
# import random

# param_grid_xgb = {
#  'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
#  'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
#  'min_child_weight' : [ 1, 3, 5, 7 ],
#  'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
#  'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ]
# }

# xgb = xgb.XGBClassifier()
# best_xgb = tune_model(xgb, param_grid_xgb, 10, X_train, y_train)
# score, y_pred = run_model(best_xgb, X_train, y_train, X_val, y_val)



# =================
# Auto-sklearn
# =================
# import autosklearn.classification
# import sklearn

# automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=90)
# automl.fit(X_train, y_train)
# y_pred = automl.predict(X_val)
# print("Accuracy score", sklearn.metrics.accuracy_score(y_val, y_pred))
