# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:18:54 2024

@author: user
"""
# Importing Libraries
import numpy as np                  # working with array 
import pandas as pd                 # import data set
import matplotlib.pyplot as plt     # for visualization
import seaborn as sns 


df = pd.read_csv('insurance.csv')

#Kategorik verileri dönüştürmek
#Başka kodlayıcılar, Label/OHE/Ordinal ?
['sex','children', 'smoker', 'region']
df_encode = pd.get_dummies(data = df, prefix = 'OHE', prefix_sep='_',
               columns = ['sex','children', 'smoker', 'region'],
               drop_first =True,
              dtype='int8')


from sklearn.model_selection import train_test_split
X = df_encode.drop('charges',axis=1) #
y = df_encode['charges'] # 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=23)


# Scikit Learn module
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train) #


R2 = regressor.score(X_test, y_test)
print('R2 = '+ str(R2))


y_pred = regressor.predict(X_test)


# Metrikler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
R2= r2_score(y_test, y_pred)
mse_ = mean_squared_error(y_test, y_pred)
mape= mean_absolute_percentage_error(y_test, y_pred)
print('R square obtain for scikit learn library is :',R2)
print('MSE',mse_)
print('MAPE',mape)

#bias variance tradeoff nedir
"""
from mlxtend.evaluate import bias_variance_decomp
loss, bias, var = bias_variance_decomp(regressor, X_train.values, y_train.values, X_test.values, y_test.values, loss='mse', num_rounds=200, random_seed=1)
"""

#Yellowbrick oldukça kullanışlı bir görselleştirme aracı
#Diğerleri?: matplotlib, seaborn, pltly vs..
from yellowbrick.regressor import PredictionError, ResidualsPlot

visualizer = PredictionError(regressor).fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.poof()
