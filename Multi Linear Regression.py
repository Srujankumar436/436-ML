#!/usr/bin/env python
# coding: utf-8

# ***Assumption in mmulti linear regression***
# - linearity:The relationship between the predictors and the response in linear
# - Independence:Observations are independent of each other
# - Homoscedasticity:The residuals exhibit constant variance at all levels of the predictor
# - Normal Distribution of errors:The residuals of the model are normally distributed
# - No mutlilinearity:The independent variables should not be too highly correlated with each other

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


cars = pd.DataFrame(cars,columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# ***description***
# - MPG:milege of car(mile per gallon)
# - VOL:volume of car(size)
# - SP:Top speed of the car(mile per hour)
# - HP:horse power of car
# - WT:weight of car

# ***EDA***

# In[6]:


cars.info()


# In[7]:


#check for missing values
cars.isna().sum()


# ***OSERVATIONS***
# - There are no missing values
# - There ae 81 observations and the datatyppes of the coulumns are valid

# In[ ]:





# In[ ]:




