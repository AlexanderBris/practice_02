# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:21:01 2023

@author: Alexander
"""

# подгрузка стандартных библиотек
import os
import numpy as np
import pandas as pd

# функция чтения данных из файла
def read_data(path, filename):
    return pd.read_csv(os.path.join(path, filename))

# загружаем данные в переменную df
cur_dir = os.getcwd()
# файлы лежат в папке с основным скриптом
df = read_data(cur_dir, 'train.csv')
# проверка
df.head()

def load_dataset(label_dict):
    train_X = read_data(cur_dir, 'train.csv').values[:,:-2]
    train_y = read_data(cur_dir, 'train.csv')['Activity']
    train_y = train_y.map(label_dict).values
    test_X = read_data(cur_dir, 'test.csv').values[:,:-2]
    test_y = read_data(cur_dir, 'test.csv')
    test_y = test_y['Activity'].map(label_dict).values
    return(train_X, train_y, test_X, test_y)    

label_dict = {'WALKING':0, 'WALKING_UPSTAIRS':1, 'WALKING_DOWNSTAIRS':2, 'SITTING':3, 'STANDING':4, 'LAYING':5}

# создаем списки для данных
train_X, train_y, test_X, test_y = load_dataset(label_dict)

# берем модели из sklearn

#------------------------------------------------------------------
#------------------------------------------------------------------

from sklearn import tree
dt_model = tree.DecisionTreeClassifier()
dt_model.fit(train_X, train_y)
err_tree_train = np.mean(train_y != dt_model.predict(train_X))
err_tree_test  = np.mean(test_y  != dt_model.predict(test_X))
print("DecisionTreeClassifier")
print(err_tree_train, err_tree_test)
#------------------------------------------------------------------
#------------------------------------------------------------------

from sklearn import ensemble
n_estimators = 100 # 50000
gbt_model = ensemble.GradientBoostingClassifier(n_estimators = n_estimators, max_depth = 1)
gbt_model.fit(train_X, train_y)
gbt_model.staged_predict(train_X)

err_gbt_train = []
for y_train_pred in gbt_model.staged_predict(train_X):
    err_gbt_train.append(np.mean(y_train_pred != train_y))
    
err_gbt_test = []
for y_test_pred in gbt_model.staged_predict(test_X):
    err_gbt_test.append(np.mean(y_test_pred != test_y))
print("GradientBoostingClassifier")
print(min(err_gbt_test), np.argmin(err_gbt_test) + 1)

# строим график
import matplotlib.pyplot as plt
plt.figure(figsize = (8, 6))
plt.plot(range(1, n_estimators + 1), err_gbt_train, 'r', label = 'Train error')
plt.plot(range(1, n_estimators + 1), err_gbt_test,  'b', label = 'Test error')
plt.legend(loc = 1)
plt.axhline(y = min(err_gbt_test), color = 'gray')
plt.xlabel('Number of Trees')
plt.ylabel('Error')



















