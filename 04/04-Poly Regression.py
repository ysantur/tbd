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


df = pd.read_csv('MLR-verileri.csv', sep=";")
from sklearn.preprocessing import PolynomialFeatures



from sklearn.model_selection import train_test_split
X = df.drop('Satis',axis=1) # Independet variable
y = df['Satis'] # dependent variable


poly=PolynomialFeatures(degree=4)

X= poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=23)


# Scikit Learn module

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
Rs = {"LinearRegression()":0,"GradientBoostingRegressor()":0,"RandomForestRegressor()":0}

for k,v in Rs.items():
    regressor= eval(k)
    regressor.fit(X_train,y_train) # Note: x_0 =1 is no need to add, sklearn will take care of it.


    R2 = regressor.score(X_test, y_test)
    Rs[k] = R2    
    


print("En iyi sınıflandırıcı:{}, skor={}".format(max(Rs, key=Rs.get), max(Rs.values())))


