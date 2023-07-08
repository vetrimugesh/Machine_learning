#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


data = pd.read_csv('Salary_dataset.csv')
data_binary = data[['YearsExperience', 'Salary']]
 
data_binary.columns = ['Experience', 'Increment']

data_binary.shape


# In[3]:


sns.lmplot(x ="Experience", y ="Increment", data = data_binary, order = 2, ci = None)
plt.show()


data_binary = data_binary.fillna(method='ffill')

X = np.array(data_binary['Experience']).reshape(-1, 1)
y = np.array(data_binary['Increment']).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

regr = LinearRegression()
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))


# In[4]:


increment = float(input('enter your experience '))
prediction = regr.predict([[increment]])
print(prediction[0])
y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='blue')
plt.plot(X_test, y_pred, color ='green')
 
plt.show()


# In[ ]:




