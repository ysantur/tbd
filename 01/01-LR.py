# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:25:00 2024

@author: user
"""

# Kütüphaneler
import numpy as np                  #  
import pandas as pd                 # 
import matplotlib.pyplot as plt     #
import random

# Eğitim veri seti
Training_Dataset = pd.read_csv("train.csv")
Training_Dataset = Training_Dataset.dropna()


X_train = np.array(Training_Dataset.iloc[:, 0]).reshape(-1,1) # Bağımsız değişkenler
y_train = np.array(Training_Dataset.iloc[:, 1]).reshape(-1,1)   # Bağımlı değişken (Hedef)



# Test veri seti
Testing_Dataset = pd.read_csv("test.csv")
Testing_Dataset = Testing_Dataset.dropna()
X_test = np.array(Testing_Dataset.iloc[:, 0]).reshape(-1,1) #Bağımsız
y_test = np.array(Testing_Dataset.iloc[:, 1]).reshape(-1,1) #Bağımlı


from sklearn.linear_model import LinearRegression  #Modelimiz
regressor = LinearRegression()
regressor.fit(X_train, y_train)



R2 = regressor.score(X_test, y_test)  #Performans metriği
print('R2 = '+ str(R2))

plt.style.use('seaborn')
plt.scatter(X_train, y_train, color = 'red', marker = 'o', s = 35, alpha = 0.5, label = 'Test data')
plt.scatter(y_test, regressor.predict(X_test), color = 'blue', marker = 'o', s = 35, alpha = 0.5, label = 'Test data')
#plt.plot(y_test, regressor.predict(X_test), color = 'blue', label='Model Plot')
plt.title('Predicted Values vs Inputs')
plt.xlabel('Inputs')
plt.ylabel('Predicted Values')
plt.legend(loc = 'upper left')
plt.show()

