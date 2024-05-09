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
ax.set_title(" Dağılım")
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
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,shuffle=True, test_size=0.2)




from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



#RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

rf_clf = KNeighborsClassifier()
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)

rf_report = pd.DataFrame(classification_report(y_test, rf_pred, output_dict=True))

print(rf_report)
print(f"Confusion Matrix: \n {confusion_matrix(y_test, rf_pred)}\n")



# ROC Curves

from sklearn.metrics import roc_curve

tpr_rf, fpr_rf, thresh_rf = roc_curve(y_test, rf_clf.predict_proba(X_test)[:, 1], pos_label = 1)

plt.plot(tpr_rf, fpr_rf, linestyle = "solid", color = "green", label = "RF")



plt.title('Receiver Operator Characteristics (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')

plt.legend(loc = 'best')
plt.savefig('ROC', dpi = 300)
plt.show()
plt.close()

#AUC
from sklearn import metrics
auc = metrics.roc_auc_score(y_test, rf_clf.predict(X_test))
print(auc)

#from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
#f1_score(y_test, p4, average='weighted')