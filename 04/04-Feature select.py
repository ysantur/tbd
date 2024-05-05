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

#Kategorik veriler
['sex','children', 'smoker', 'region']
df_encode = pd.get_dummies(data = df, prefix = 'OHE', prefix_sep='_',
               columns = ['sex','children', 'smoker', 'region'],
               drop_first =True,
              dtype='int8')


from sklearn.model_selection import train_test_split
X = df_encode.drop('charges',axis=1) # Independet variable
y = df_encode['charges'] # dependent variable


corr = df.loc[:,["age","bmi","children","charges"]].corr()
matrix = np.triu(corr)


sns.heatmap(corr, cmap = 'Wistia', annot= True) #mask=matrix 
plt.show()

#bmi sildik ve doğruluk yükseldi ?
X = df_encode.drop('bmi',axis=1) # Independet variable

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=23)


# Scikit Learn module

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
Rs = {"LinearRegression()":0,"GradientBoostingRegressor()":0,"RandomForestRegressor()":0, "LogisticRegression()":0}

for k,v in Rs.items():
    regressor= eval(k)
    regressor = GradientBoostingRegressor()
    regressor.fit(X_train,y_train) # Note: x_0 =1 is no need to add, sklearn will take care of it.


    R2 = regressor.score(X_test, y_test)
    Rs[k] = R2    
    


print("En iyi sınıflandırıcı:{}, skor={}".format(max(Rs, key=Rs.get), max(Rs.values())))

