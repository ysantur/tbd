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


sns.lmplot(x='bmi',y='charges',data=df,aspect=2,height=6)
plt.xlabel('Boby Mass Index$(kg/m^2)$: as Independent variable')
plt.ylabel('Insurance Charges: as Dependent variable')
plt.title('Charge Vs BMI');
plt.show()



#Heatmap - Korelasyon
corr = df.loc[:,["age","bmi","children","charges"]].corr()
matrix = np.triu(corr)

sns.heatmap(corr, cmap = 'Wistia', annot= True) #mask=matrix 
plt.show()

#pair plot ve maske
sns.pairplot(df,
                 markers="+",
                 kind='reg',
                 diag_kind="kde",
                 plot_kws={'line_kws':{'color':'#aec6cf'},
                           'scatter_kws': {'alpha': 0.5,
                                           'color': '#82ad32'}},
               corner=True,
                 diag_kws= {'color': '#82ad32'})
plt.show()

#Yoğunluk
sns.set_style('whitegrid')

sns.distplot(df['charges'], kde = True, color ='red', bins = 30)
plt.show()


#Violin
sns.violinplot(x='sex', y='charges',data=df,palette='Wistia')
plt.show()

#Box
sns.boxplot(x='children', y='charges',hue='sex',data=df,palette='rainbow')
plt.show()


#Scatter
sns.scatterplot(x='age',y='charges',data=df,palette='magma',hue='smoker')
plt.show()

sns.scatterplot(x='bmi',y='charges',data=df,palette='viridis',hue='smoker')
plt.show()

#Alan 
sns.jointplot(x ='bmi', y ='charges', data = df, kind ='kde')
plt.show()

#Req plot
sns.regplot(data=df, x="bmi", y="charges",color="red")
plt.show()