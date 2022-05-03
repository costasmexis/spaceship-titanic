# ===========================
# Import Libraries
# ===========================
import numpy as np
import pandas as pd

train = pd.read_csv('../data/data_train.csv')

train['Transported'] = train['Transported'].astype(int)

# Set index PassengerId
train = train.set_index('PassengerId')

# ===========================
# Split TRAIN/VAL sets
# ===========================
X = train.drop(['Transported'],axis=1)
y = train['Transported']

from sklearn.model_selection import train_test_split

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.3,random_state=0)

# ===========================
# Normalize data
# ===========================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# =========================
# Modelling
# =========================
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

def run_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return score, y_pred
    
def print_scores(y_true, y_pred):
    print('ROCAUC score:',roc_auc_score(y_true, y_pred))
    print('Accuracy score:',accuracy_score(y_true, y_pred))
    print('F1 score:',f1_score(y_true, y_pred))
    print('Precision score:',precision_score(y_true, y_pred))
    print('Recall:',recall_score(y_true, y_pred))
    print("\n\n")

 
# LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

model = LogisticRegression()
score, y_pred = run_model(model, X_train, y_train, X_val, y_val) 
print_scores(y_val, y_pred) 

# SVC
from sklearn.svm import SVC

# # defining parameter range
# param_grid = {'C': [1, 1.5, 2, 3, 5, 10, 12, 20],
#               'gamma': [0.002, 0.003, 0.004, 0.001],
#               'kernel': ['rbf']
# }

# grid = RandomizedSearchCV(SVC(random_state=32), param_grid, verbose = 100, 
#     scoring='accuracy', cv=3, n_iter=10)
 
# # fitting the model for grid search
# grid.fit(X_train, y_train)
# best_model = grid.best_estimator_
# print(best_model)

# grid_predictions = grid.predict(X_val)
# print_scores(y_val, grid_predictions)

# best_svc = SVC(C=20, gamma=0.004, random_state=32)
# best_svc.fit(X_train, y_train)
# print(best_svc)

# y_pred = best_svc.predict(X_val)
# print_scores(y_val, y_pred)


# from sklearn.ensemble import RandomForestClassifier

# n_estimators = [10,100, 300, 500, 800, 1200, 2000]
# max_depth = [5, 8, 15, 25, 30, 35, 40, 50, 80, 100]
# min_samples_split = [2, 5, 10, 15, 100]
# min_samples_leaf = [1, 2, 5, 10] 

# hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
#               min_samples_split = min_samples_split, 
#              min_samples_leaf = min_samples_leaf)

# forest = RandomForestClassifier(random_state = 1)

# gridF = RandomizedSearchCV(forest, hyperF, cv = 3, verbose = 100, 
#                       n_jobs = -1, n_iter=25)
# bestF = gridF.fit(X_train, y_train)
# best_model = gridF.best_estimator_
# print(best_model)

# grid_predictions = best_model.predict(X_val)
# print_scores(y_val, grid_predictions)

import xgboost as xgb
import random

params = {
 'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
 'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
 'min_child_weight' : [ 1, 3, 5, 7 ],
 'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ]
}

model = xgb.XGBClassifier()

# fitting the model for grid search
grid = RandomizedSearchCV(model, params, verbose = 100, 
    scoring='accuracy', cv=3, n_iter=10)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print(best_model)

grid_predictions = best_model.predict(X_val)
print_scores(y_val, grid_predictions)








