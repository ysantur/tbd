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

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=0,shuffle=False, test_size=0.25)


from keras.models import Sequential
from keras.layers import Dense,Dropout, LSTM, GRU
classifier = Sequential()

# katmanlar, nöron sayısı, aktivasyon fonksiyonları, dropout

l1=classifier.add(GRU(units = 100, input_shape=(X.shape[1],1), return_sequences=True, name="layer1"))
classifier.add(Dropout(0.25))

classifier.add(GRU(units = 50, return_sequences=True, name="layer2"))
classifier.add(Dropout(0.25))

classifier.add(GRU(units = 20, return_sequences=True, name="layer3"))
classifier.add(Dropout(0.25))


# Adding the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the ANN | means applying SGD on the whole ANN
classifier.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['accuracy'])

history=classifier.fit(x_train, y_train, batch_size = 50, epochs = 100, validation_split=0.1)

score, acc = classifier.evaluate(x_train, y_train,
                            batch_size=10)
print('Train score:', score)
print('Train accuracy:', acc)

# Sınıflandırma kriteri oluşturma
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

print('*'*20)
score, acc = classifier.evaluate(x_test, y_test,batch_size=10)

print('Test score:', score)
print('Test accuracy:', acc)

from sklearn.metrics import confusion_matrix

"""
#cm = confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
"""

# Plot Acc
plt.plot(history.history['accuracy'], color="green")
plt.plot(history.history['val_accuracy'], color="red")
plt.title('Plot History: Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Val Accuracy'], loc='upper left')
plt.show()

# Plot Loss
plt.plot(history.history['loss'], color="green")
plt.plot(history.history['val_loss'], color="red")
plt.title('Plot History: Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Loss', 'Val Loss'], loc='upper left')
plt.show()

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()

#history.model.get_weights()