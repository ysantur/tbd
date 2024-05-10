# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 09:51:35 2023

@author: ysant
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:58:13 2021

@author: santury
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart.csv')

df.sample(10)




counts = df['target'].value_counts()
fig, ax = plt.subplots()
ax.pie(counts, autopct='%1.1f%%')
ax.legend(labels=['0 (Normal)', '1 (Suspect)'], title='Dağılım',loc='lower right')
ax.set_title(" dağılım")
plt.show()
plt.close()



corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

ax, fig = plt.subplots(figsize=(15,15))
sns.heatmap(corr, vmin=-1, cmap='RdYlBu', annot=True, mask=mask)
plt.show()




X = df.drop(['target'], axis=1)
y = df['target']



#Korelasyon
from yellowbrick.target import FeatureCorrelation
visualizer = FeatureCorrelation(labels=X.columns)

visualizer.fit(X, y)        # Fit the data to the visualizer
visualizer.show() 


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,shuffle=False, test_size=0.25)




from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



#GridSearch daha önceki örneklerde AutoML'e girişe benzer olarak, bir modeldeki en iyi hyperparameters bulur
#Overfitting ???

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

params = {"criterion":("gini", "entropy"), 
          "splitter":("best", "random"), 
          "max_depth":(list(range(2, 5))), 
          "min_samples_split":[2, 3, 4], 
          "min_samples_leaf":list(range(2, 15))
          }

tree_clf = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree_clf, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=5)
tree_cv.fit(X_train, y_train)
best_params = tree_cv.best_params_
print(f'Best_params: {best_params}')

tree_clf = DecisionTreeClassifier(**best_params)
tree_clf.fit(X_train, y_train)

tree_pred=tree_clf.predict(X_test)


report = pd.DataFrame(classification_report(y_test, tree_pred, output_dict=True))

print(report)
print(f"Confusion Matrix: \n {confusion_matrix(y_test, tree_pred)}\n")



