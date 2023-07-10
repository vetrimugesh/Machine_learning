#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# In[2]:


data = pd.read_csv('gender.csv')


# In[3]:


data


# In[4]:


x= data.iloc[:,1:4] 
y= data.iloc[:,0]
y


# In[5]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


# In[6]:


st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)

model = LogisticRegression()
model.fit(x_train, y_train)


# In[7]:


accuracy = model.score(x_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))

