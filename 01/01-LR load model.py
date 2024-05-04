# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:25:00 2024

@author: user
"""

# Kütüphaneler                # 

# Kaydedilmiş modeli yüklemek

import joblib
# Load
regressor = joblib.load("model.pkl")

print(regressor.predict([[10]]))

print(regressor.predict([[100]]))
print(regressor.predict([[1000]]))
