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

from bayes_opt import BayesianOptimization, UtilityFunction

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb


import pickle

seed = 42 # Random seed

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

    # save the model to disk
    filename = '../models/finalized_svc.sav'
    pickle.dump(best_svc, open(filename, 'wb'))

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

    # save the model to disk
    filename = '../models/finalized_rf.sav'
    pickle.dump(best_rf, open(filename, 'wb'))

    run_on_test(best_rf, X_train, y_train.values.ravel(), X_test, '../submissions/submission_rf.csv')


    param_grid_xgb = {
     'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
     'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
     'min_child_weight' : [ 1, 3, 5, 7 ],
     'gamma': [ 0.0, 0.01, 0.05, 0.1, 0.2 , 0.3, 0.4 ],
     'reg_alpha': [0.01, 0.1, 0.5, 1, 2, 4, 10, 15, 20],
     'reg_lambda': [0.01, 0.1, 0.5, 1, 2, 4, 10, 15, 20],
     'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ],
     'n_estimators' : [10, 25, 50, 100, 150, 200, 300, 500]
    }

    xgb_model = xgb.XGBClassifier(use_label_encoder=False)
    X_train_xgb, X_val_xgb = normalize(X_sub_train, X_val)
    best_xgb = tune_model(xgb_model, param_grid_xgb, 30, X_train_xgb, y_sub_train.values.ravel())
    score, y_pred = run_model(best_xgb, X_train_xgb, y_sub_train.values.ravel(), X_val_xgb, y_val.values.ravel())
    
    # save the model to disk
    filename = '../models/finalized_xgb.sav'
    pickle.dump(best_xgb, open(filename, 'wb'))

    run_on_test(best_xgb, X_train, y_train.values.ravel(), X_test, '../submissions/submission_xgb.csv')


def ann(type):

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

    if(type=="validation"):
        # FOR VALIDATION
        # Normalize data
        X_train_ann, X_val_ann = normalize(X_sub_train, X_val)


        hist = model.fit(X_train_ann, y_sub_train, epochs=100, batch_size=50)
        y_pred = model.predict(X_val_ann) > .5
        print("Accuracy:",accuracy_score(y_val, y_pred))
    elif(type=="testing"):
        # FOR TEST SET AND SUBMISSION
        # Normalize data
        X_train_ann, X_test_ann = normalize(X_train, X_test)

        hist = model.fit(X_train_ann, y_train, epochs=80, batch_size=20)
        y_pred = model.predict(X_test_ann) > .5

        submission = pd.DataFrame({'PassengerId': index_submission ,'Transported': y_pred.reshape(-1,)},columns=['PassengerId', 'Transported'])

        submission.to_csv('../submissions/submission_ann.csv', index=False)
    else:
        print("Wrong choice... You should choose either 'validation' or 'testing'")



def mlpclassifier():
    from sklearn.neural_network import MLPClassifier

    param_grid_mlp = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
    }

    mlp = MLPClassifier(max_iter=10000)
    X_train_mlp, X_val_mlp = normalize(X_sub_train, X_val)
    best_mlp = tune_model(mlp, param_grid_mlp, 10, X_train_mlp, y_sub_train.values.ravel())
    score, y_pred = run_model(best_mlp, X_train_mlp, y_sub_train.values.ravel(), X_val_mlp, y_val.values.ravel())

    # run_on_test(best_mlp, X_train, y_train.values.ravel(), X_test, '../submissions/submission_mlp.csv')



def stacking():
    
    from sklearn.ensemble import StackingClassifier
    from sklearn.neighbors import KNeighborsClassifier  

    # load the model from disk
    svc = pickle.load(open('../models/finalized_svc.sav', 'rb'))
    rf = pickle.load(open('../models/finalized_rf.sav', 'rb'))
    xgb = pickle.load(open('../models/finalized_xgb.sav', 'rb'))


    # Create Base Learners
    base_learners = [
                 ('rf_1', svc),
                 ('rf_2', rf)
                ]

    # Initialize Stacking Classifier with the Meta Learner
    clf = StackingClassifier(estimators=base_learners, final_estimator=xgb)

    X_train_clf, X_val_clf = normalize(X_sub_train, X_val)
    score, y_pred = run_model(clf, X_train_clf, y_sub_train.values.ravel(), X_val_clf, y_val.values.ravel())

    run_on_test(clf, X_train, y_train.values.ravel(), X_test, '../submissions/submission_stack.csv')


def my_bayesian():

    def black_box_function(C, gamma):
        # C: SVC hyper parameter to optimize for.
        model = SVC(C = C, gamma=gamma)
        model.fit(X_train_scaled.values, y_sub_train.values.ravel())
        # y_score = model.decision_function(X_val_scaled.values)
        # f = roc_auc_score(y_val.values.ravel(), y_score)
        return model.score(X_val_scaled, y_val)

    # Set range of C to optimize for.
    # bayes_opt requires this to be a dictionary.

    param_grid_xgb = {
     'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
     'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
     'min_child_weight' : [ 1, 3, 5, 7 ],
     'gamma': [ 0.0, 0.01, 0.05, 0.1, 0.2 , 0.3, 0.4 ],
     'reg_alpha': [0.01, 0.1, 0.5, 1, 2, 4, 10, 15, 20],
     'reg_lambda': [0.01, 0.1, 0.5, 1, 2, 4, 10, 15, 20],
     'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ],
     'n_estimators' : [10, 25, 50, 100, 150, 200, 300, 500]
    }

    # Create a BayesianOptimization optimizer,
    # and optimize the given black_box_function.

    X_train_scaled, X_val_scaled = normalize(X_sub_train, X_val)

    optimizer = BayesianOptimization(f = black_box_function, pbounds=param_grid_svc, verbose = 20, random_state = 4)

    optimizer.maximize(init_points = 25, n_iter = 25)
    print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

    best_model_svc_bayesian = SVC(C=optimizer.max['params']['C'], gamma=optimizer.max['params']['gamma'])

    run_on_test(best_model_svc_bayesian, X_train, y_train.values.ravel(), X_test, '../submissions/submission_svc_bayesian.csv')

import warnings
warnings.filterwarnings('ignore')

def xgb_bayesian():

    def xgbc_cv(min_child_weight,colsample_bytree,gamma):
    
        estimator_function = xgb.XGBClassifier(max_depth=int(5.0525),
                                               colsample_bytree= colsample_bytree,
                                               gamma=gamma,
                                               min_child_weight= int(min_child_weight),
                                               learning_rate= 0.2612,
                                               n_estimators= int(75.5942),
                                               reg_alpha = 0.9925,
                                               nthread = -1,
                                               objective='binary:logistic'
                                               )
        # Fit the estimator
        estimator_function.fit(X_train_scaled.values, y_sub_train.values.ravel())
        
        # calculate out-of-the-box roc_score using validation set 1
        probs = estimator_function.predict_proba(X_val_scaled)
        probs = probs[:,1]
        val1_roc = roc_auc_score(y_val.values.ravel(),probs)
        
        # return the mean validation score to be maximized 
        return val1_roc


    gp_params = {"alpha": 1e-10}

    hyperparameter_space = {
    'min_child_weight': (1, 20),
    'colsample_bytree': (0.1, 1),
    'gamma' : (0,10)
    }

    X_train_scaled, X_val_scaled = normalize(X_sub_train, X_val)

    xgbcBO = BayesianOptimization(f = xgbc_cv, 
                                 pbounds =  hyperparameter_space,
                                 random_state = 32,
                                 verbose = 10)

    # Finally we call .maximize method of the optimizer with the appropriate arguments

    xgbcBO.maximize(init_points=3,n_iter=10,acq='ucb', kappa= 3, **gp_params)

    print("Best result: {}; f(x) = {}.".format(xgbcBO.max["params"], xgbcBO.max["target"]))

def my_xgb():


    # Initially trying an untuned classifier to test the performance
    import datetime
    import warnings
    warnings.filterwarnings('ignore')

    import xgboost as xgb
    start = datetime.datetime.now()

    X_train_scaled, X_val_scaled = normalize(X_sub_train, X_val)

    # Instantiate the classifier
    xgbc = xgb.XGBClassifier(nthread=3)
    xgbc.fit(X_train_scaled, y_sub_train, eval_metric='logloss')

    end = datetime.datetime.now()
    process_time = end - start
    print("Training XGB classifier took: " + str(process_time.seconds/60) + " minutes.")

    probs = xgbc.predict_proba(X_train_scaled)
    probs = probs[:,1]

    print(probs)

    print("Untuned XGB classifier roc score on sub train set: " + str(roc_auc_score(y_sub_train,probs)))    
    print("Untuned XGB classifier accuracy on sub train set: " + str(accuracy_score(y_sub_train, probs>.5)))    
    print()

    # calculate out-of-the-box roc_score using validation set 1
    y_pred_val = xgbc.predict(X_val_scaled)
    print("Untuned XGB classifier accuracy on val set: " + str(accuracy_score(y_val, y_pred_val)))    

    def xgbc_cv(max_depth,learning_rate,n_estimators,reg_alpha):
        
        estimator_function = xgb.XGBClassifier(max_depth=int(max_depth),
                                               learning_rate= learning_rate,
                                               n_estimators= int(n_estimators),
                                               reg_alpha = reg_alpha,
                                               nthread = -1,
                                               objective='binary:logistic',
                                               seed = seed)
        # Fit the estimator
        estimator_function.fit(X_train_scaled, y_sub_train, eval_metric='logloss')

        # calculate out-of-the-box roc_score using validation set 1
        probs = estimator_function.predict_proba(X_val_scaled)
        probs = probs[:,1]
        val_roc = roc_auc_score(y_val, probs)

        y_pred_val = estimator_function.predict(X_val_scaled)
        
        # return the validation score to be maximized 
        # return val_roc
        return accuracy_score(y_val, y_pred_val)

    # alpha is a parameter for the gaussian process
    # Note that this is itself a hyperparemter that can be optimized.
    gp_params = {"alpha": 1e-10}

    # We create the BayesianOptimization objects using the functions that utilize
    # the respective classifiers and return cross-validated scores to be optimized.

    # We create the bayes_opt object and pass the function to be maximized
    # together with the parameters names and their bounds.
    # Note the syntax of bayes_opt package: bounds of hyperparameters are passed as two-tuples

    hyperparameter_space = {
        'max_depth': (5, 20),
        'learning_rate': (0, 1),
        'n_estimators' : (10,100),
        'reg_alpha': (0,1)
    }

    xgbcBO = BayesianOptimization(f = xgbc_cv, 
                                 pbounds =  hyperparameter_space,
                                 random_state = seed,
                                 verbose = 10)

    # Finally we call .maximize method of the optimizer with the appropriate arguments
    # kappa is a measure of 'aggressiveness' of the bayesian optimization process
    # The algorithm will randomly choose 3 points to establish a 'prior', then will perform 
    # 10 interations to maximize the value of estimator function
    xgbcBO.maximize(init_points=3,n_iter=10,acq='ucb', kappa= 3, **gp_params)
    print("Best result: {}; f(x) = {}.".format(xgbcBO.max["params"], xgbcBO.max["target"]))
