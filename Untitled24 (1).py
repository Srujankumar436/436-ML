#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np


# In[3]:


cars = pd.read_csv(r"C:\Users\MRUH\Downloads\Cars.csv")
cars.head()


# In[4]:


cars.head()


# In[5]:


cars


# In[6]:


sns.displot(cars)


plt.title("Distribution Plot")
plt.show()


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt


cars = sns.load_dataset('mpg')  

sns.boxplot(x=cars['mpg'])  


plt.title("Distribution of MPG")

plt.show()


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt

cars = sns.load_dataset('mpg')

sns.scatterplot(x=cars['horsepower'], y=cars['mpg'])  

plt.title("Horsepower vs MPG")


plt.show()


# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt


cars = sns.load_dataset('mpg')  

sns.boxplot(x=cars['mpg'])  


plt.title("Distribution of MPG")

plt.show()


# In[ ]:




