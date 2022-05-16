import pandas as pd
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from ssklearn.utils import class_weight
from xgboost import XGBClassifier
import pickle


# Load the csv file
train = pd.read_csv("ins_train.csv")
test = pd.read_csv("ins_test.csv")

print(train.head())
del train["Region_Code"]
del train["Annual_Premium"]
del train["Policy_Sales_Channel"]
del train["Vintage"]
print(train)
del test["Region_Code"]
del test["Annual_Premium"]
del test["Policy_Sales_Channel"]
del test["Vintage"]

le = preprocessing.LabelEncoder()

train.Gender = le.fit_transform(train.Gender)
train.Vehicle_Age = le.fit_transform(train.Vehicle_Age)
train.Vehicle_Damage = le.fit_transform(train.Vehicle_Damage)

test.Gender = le.fit_transform(test.Gender)
test.Vehicle_Age = le.fit_transform(test.Vehicle_Age)
test.Vehicle_Damage = le.fit_transform(test.Vehicle_Damage)

y = train.Response
X = train.drop(['Response', 'id'], axis = 1)
X_test = test.drop(['id'], axis = 1)
X.shape, y.shape, X_test.shape

X_train, X_val, y_train, y_val = train_test_split (X, y, random_state=1, test_size=0.10, stratify=y)
X_train.shape, X_val.shape, y_train.shape, y_val.shape

class_weights = dict(zip(np.unique(y_train), class_weight.compute_class_weight(class_weight='balanced', classes =np.unique(y_train), y= y_train)))

model = XGBClassifier(learning_rate=1, n_estimators=2000, max_depth=40, min_child_weight=40,
                      gamma=0.4,nthread=10, subsample=0.8, colsample_bytree=.8,
                      objective= 'binary:logistic',scale_pos_weight=10,seed=29, class_weight=class_weights)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
print(model.score(X_val, y_val))

pred = model.predict(X_test)
pred
print(train)
print(test)
pickle.dump(model, open("pima.pickle", "wb"))
