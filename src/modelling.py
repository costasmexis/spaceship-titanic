# ===========================
# Import Libraries
# ===========================
import numpy as np
import pandas as pd
import random

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

# ==================
# Load datasets
# ==================

X_train = pd.read_csv('../data/X_train.csv', index_col='PassengerId')
y_train = pd.read_csv('../data/y_train.csv', index_col='PassengerId')

X_sub_train = pd.read_csv('../data/X_sub_train.csv', index_col='PassengerId')
y_sub_train = pd.read_csv('../data/y_sub_train.csv', index_col='PassengerId')

X_val = pd.read_csv('../data/X_val.csv', index_col='PassengerId')
y_val = pd.read_csv('../data/y_val.csv', index_col='PassengerId')

X_test = pd.read_csv('../data/X_test.csv', index_col='PassengerId')

index_submission = X_test.index # Keep 'PassengerId' of test set to use for submission file

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

def normalize(X_train, X_test):
    scaler = StandardScaler()
    X_train_ = X_train.copy()
    X_test_ = X_test.copy()
    X_train_[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])
    X_test_[cols_to_norm] = scaler.transform(X_test[cols_to_norm])
    return X_train, X_test

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

def run_on_test(model, X_train, y_train, X_test, filename):
    print(model)    
    X_train_, X_test_ = normalize(X_train, X_test)
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

# ==========================
# ++++++++++++++++++++++++++
# 
#           MAIN 
#
# ++++++++++++++++++++++++++
# ==========================



cols_to_norm = X_train.select_dtypes(include='float64').columns.values # The numerical features to normalize before training


def main():

    X_train_lg, X_val_lg = normalize(X_sub_train, X_val)
    lg = LogisticRegression(max_iter=10000)
    score, y_pred = run_model(lg, X_train_lg, y_sub_train.values.ravel(), X_val_lg, y_val.values.ravel()) 


    # SVC
    # defining parameter range
    param_grid_svc = {'C': [1, 1.5, 2, 3, 5, 10, 12, 20],
                  'gamma': [0.002, 0.003, 0.004, 0.001],
                  'kernel': ['rbf']
    }

    X_train_svc, X_val_svc = normalize(X_sub_train, X_val)
    svc = SVC(random_state=32)
    best_svc = tune_model(svc, param_grid_svc, 25, X_train_svc, y_sub_train.values.ravel())
    score, y_pred = run_model(best_svc, X_train_svc, y_sub_train.values.ravel(), X_val_svc, y_val.values.ravel())

    run_on_test(best_svc, X_train, y_train.values.ravel(), X_test, '../submissions/submission_new_svc.csv')


    n_estimators = [10,100, 300, 500, 800, 1200, 2000]
    max_depth = [5, 8, 15, 25, 30, 35, 40, 50, 80, 100]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10] 

    param_grid_rf = dict(n_estimators = n_estimators, 
                max_depth = max_depth,  
                min_samples_split = min_samples_split, 
                min_samples_leaf = min_samples_leaf)

    rf = RandomForestClassifier(random_state = 1)
    X_train_rf, X_val_rf = normalize(X_sub_train, X_val)
    best_rf = tune_model(rf, param_grid_rf, 25, X_train_rf, y_sub_train.values.ravel())
    score, y_pred = run_model(best_rf, X_train_rf, y_sub_train.values.ravel(), X_val_rf, y_val.values.ravel())

    run_on_test(best_rf, X_train, y_train.values.ravel(), X_test, '../submissions/submission_rf.csv')


    param_grid_xgb = {
     'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
     'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
     'min_child_weight' : [ 1, 3, 5, 7 ],
     'gamma': [ 0.0, 0.01, 0.05, 0.1, 0.2 , 0.3, 0.4 ],
     'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ],
     'n_estimators' : [10, 25, 50, 100, 150, 200, 300, 500]
    }

    xgb_model = xgb.XGBClassifier(use_label_encoder=False)
    X_train_xgb, X_val_xgb = normalize(X_sub_train, X_val)
    best_xgb = tune_model(xgb_model, param_grid_xgb, 30, X_train_xgb, y_sub_train.values.ravel())
    score, y_pred = run_model(best_xgb, X_train_xgb, y_sub_train.values.ravel(), X_val_xgb, y_val.values.ravel())

    run_on_test(best_xgb, X_train, y_train.values.ravel(), X_test, '../submissions/submission_xgb.csv')


def ann():

    import tensorflow as tf
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier

    opt = tf.keras.optimizers.SGD(learning_rate=0.01)

    model = Sequential()
    model.add(Dense(100, kernel_initializer = 'uniform', activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(100, kernel_initializer = 'uniform', activation='relu'))
    model.add(Dense(100, kernel_initializer = 'uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer = 'uniform', activation='sigmoid'))
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    # Normalize data
    # X_train_ann, X_val_ann = normalize(X_sub_train, X_val)


    # hist = model.fit(X_train_ann, y_sub_train, epochs=100, batch_size=20)
    # y_pred = model.predict(X_val_ann) > .5
    # print("Accuracy:",accuracy_score(y_val, y_pred))

    # Normalize data
    X_train_ann, X_test_ann = normalize(X_train, X_test)

    hist = model.fit(X_train_ann, y_train, epochs=80, batch_size=20)
    y_pred = model.predict(X_test_ann) > .5

    submission = pd.DataFrame({'PassengerId': index_submission ,'Transported': y_pred.reshape(-1,)},columns=['PassengerId', 'Transported'])

    submission.to_csv('../submissions/submission_ann.csv', index=False)

ann()

