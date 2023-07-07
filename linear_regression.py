#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd  
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Salary_Data.csv')
data.notnull()

X = data.iloc[:, 0].values.reshape(-1, 1)  
Y = data.iloc[:, 1].values.reshape(-1, 1) 
linear_regressor = LinearRegression()  
linear_regressor.fit(X, Y)  
Y_pred = linear_regressor.predict(X)  

plt.scatter(X, Y, color= 'orange')
plt.plot(X, Y_pred, color='red')
plt.show()


# In[ ]:




