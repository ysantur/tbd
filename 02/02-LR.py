# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:18:54 2024

@author: user
"""
import numpy as np                  # working with array 
import pandas as pd                 # import data set
import matplotlib.pyplot as plt     # for visualization

df = pd.read_csv('LR-verileri.csv', sep=";") #Excelden oluşturdğumuz için sep kullandık


X = df.loc[:, "Aylar"].values.reshape(-1, 1) # Bağımsız 
y = df.loc[:, "Satis"]  # Bağımlı


#Train-Test split ayırma 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Training the Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

R2 = regressor.score(X_test, y_test)
print('R2 = '+ str(R2))


#Modeli kullanarak yeni değer tahmin etmek
print(regressor.predict([[20]]))

plt.style.use('seaborn')
plt.scatter(X_test, y_test, color = 'red', marker = 'o', s = 35, alpha = 0.5,
          label = 'Test data')
plt.plot(X_train, regressor.predict(X_train), color = 'blue', label='Model Plot')
plt.title('Predicted Values vs Inputs')
plt.xlabel('Inputs')
plt.ylabel('Predicted Values')
plt.legend(loc = 'upper left')
plt.show()

