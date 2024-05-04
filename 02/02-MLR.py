# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:18:54 2024

@author: user
"""
# Importing Libraries
import numpy as np                  # working with array 
import pandas as pd                 # import data set
import matplotlib.pyplot as plt     # for visualization

df = pd.read_csv('MLR-verileri.csv', sep=";") #Excelden oluşturdğumuz için sep kullandık


X = df.iloc[:, :-1] # Bağımsızlar
y = df.iloc[:, -1]  # Bağımlı



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Training the Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

R2 = regressor.score(X_test, y_test)
print('R2 = '+ str(R2))

#Modeli kullanarak yeni değer tahmin etmek
print(regressor.predict([[60,20, 19]]))

