# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:25:00 2024

@author: user
""" 

#Kütüphaneler
import numpy as np                  # 
import pandas as pd                 #
import matplotlib.pyplot as plt     # 
import random



#İlk örnekten farklı olarak sıralı dizi oluşturup random (0-1) arası gürültü ekledik 
X_train = np.array([i for i in range(100)]).reshape(-1,1) # Bağımsız değişken
y_train = np.array([i+random.random() for i in range(100)]).reshape(-1,1)   # Bağımlı değişken (gürültü eklendi)


#Test verileri, reshape numpy vector yapmak için
X_test = np.array(np.array([i for i in range(100,125)]) ).reshape(-1,1)   #
y_test = np.array(np.array([i+random.random() for i  in range(100,125)]) ).reshape(-1,1)     


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#metrics
bias = (regressor.predict(X_test) - y_test).mean()
var = y_test.var()


R2 = regressor.score(X_test, y_test)
print('R2 = '+ str(R2))


plt.style.use('seaborn')
plt.scatter(X_train, y_train, color = 'red', marker = 'o', s = 35, alpha = .5,
          label = 'Train data')
plt.plot(y_test, regressor.predict(X_test), color = 'blue', label='Model Plot')
plt.title('Predicted Values vs Inputs')
plt.xlabel('Inputs')
plt.ylabel('Predicted Values')
plt.legend(loc = 'upper left')
plt.show()

#Formül katsayıları
"""
Formül = x*regressor.coef_ + regressor.intercept_
regressor.predict([[1000]])
"""

"""
from mlxtend.evaluate import bias_variance_decomp
loss, bias, var = bias_variance_decomp(regressor, X_train, y_train, X_test, y_test, loss='mse', num_rounds=200, random_seed=1)
"""