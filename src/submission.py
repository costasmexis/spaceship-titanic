# =============================================================================
# Import Libraries
# =============================================================================
import numpy as np
import pandas as pd


train = pd.read_csv('data_train.csv')
test = pd.read_csv('data_test.csv')

train['Transported'] = train['Transported'].astype(int)

# Set index PassengerId
train = train.set_index('PassengerId')
test = test.set_index('PassengerId')

# ===========================
# Split TRAIN/VAL sets
# ===========================
X_train = train.drop(['Transported'],axis=1)
y_train = train['Transported']

X_test = test

# ===========================
# Normalize data
# ===========================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# Modelling
# =========================

# LogisticRegression

# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# from sklearn.svm import SVC

# model = SVC(C=20, gamma=0.004, random_state=32)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

import xgboost as xgb
import random
from sklearn.model_selection import RandomizedSearchCV


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
    scoring='accuracy', cv=3, n_iter=25)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print(best_model)

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

submission = pd.DataFrame({'PassengerId':test.index ,'Transported': y_pred.astype(bool)},
        columns=['PassengerId', 'Transported'])

submission.to_csv("submission.csv",index=False)

