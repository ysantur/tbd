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


#Katehorik verileri dönüştürdük
['sex','children', 'smoker', 'region']
df_encode = pd.get_dummies(data = df, prefix = 'OHE', prefix_sep='_',
               columns = ['sex', 'smoker', 'region'],
               drop_first =True,
              dtype='int8')


from sklearn.model_selection import train_test_split
X = df_encode.drop('charges',axis=1) # Independet variable
y = df_encode['charges'] # dependent variable

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=23)


# En iyi sınıflandırıcıyı seçelim, AutoML'e giriş

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

Rs = {"LinearRegression()":0,"GradientBoostingRegressor()":0,"RandomForestRegressor()":0}

for k in Rs.keys():
    regressor= eval(k)
    regressor.fit(X_train,y_train) # Note: x_0 =1 is no need to add, sklearn will take care of it.

    R2 = regressor.score(X_test, y_test)
    Rs[k] = R2    
    
#En iyi skorları bar/ver grafik olarak gösterme

#plt.barh(range(len(Rs)), list(Rs.values()), align='center')
#plt.yticks(range(len(Rs)), list(Rs.keys()))
plt.barh(*zip(*Rs.items()))

plt.show()

print("En iyi sınıflandırıcı:{}, skor={}".format(max(Rs, key=Rs.get), max(Rs.values())))

