#!/usr/bin/env python
# coding: utf-8

# Multilinear regression, commonly referred to as multiple linear regression, is a statistical technique that models the relationship between  
# and a response variable by fitting a linear equation to observed data. Essentially, it extends the simple linear regression model to incorporate  
# providing a way to evaluate how multiple factors impact the outcome.
# 
# ### Assumptions in Multilinear Regression  
# 1. *Linearity*: The relationship between the predictors and the response is linear.  
# 2. *Independence*: Observations are independent of each other.  
# 3. *Homoscedasticity*: The residuals (differences between observed and predicted values) exhibit constant variance at all levels of the predictors.  
# 4. *Normal Distribution of Errors*: The residuals of the model are normally distributed.  
# 5. *No multicollinearity*: The independent variables should not be too highly correlated with each other.  
# 
# Violations of these assumptions may lead to inefficiency in the regression parameters and unreliable predictions.
# 
# The general formula for multiple linear regression is:  
# 
# y
# =
# 𝛽
# 0
# +
# 𝛽
# 1
# 𝑋
# 1
# +
# 𝛽
# 2
# 𝑋
# 2
# +
# ⋯
# +
# 𝛽
# 𝑛
# 𝑋
# 𝑛
# +
# 𝜖
# Y=β 
# 0
# ​
#  +β 
# 1
# ​
#  X 
# 1
# ​
#  +β 
# 2
# ​
#  X 
# 2
# ​
#  +⋯+β 
# n
# ​
#  X 
# n
# ​
#  +ϵ

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


cars = pd.DataFrame(cars,columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# #### Description of columns
# - MPG: Milege of the car
# - VOL: Volume of the car (size)
# - SP: Top speed of the car (miles per hour)
# - WT: weight of the car

# #### EDA

# In[4]:


cars.info()


# In[5]:


cars.isna().sum()


# **Observations**
# 
# * No missing values         
# * There are 81 observations                   
# * Data sets are relevant and valid                   

# In[6]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')  # Remove x label for the boxplot

# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust Layout
plt.tight_layout()
plt.show()


# In[7]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')  # Remove x label for the boxplot

# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust Layout
plt.tight_layout()
plt.show()


# In[8]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')  # Remove x label for the boxplot

# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust Layout
plt.tight_layout()
plt.show()


# In[9]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')  # Remove x label for the boxplot

# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust Layout
plt.tight_layout()
plt.show()


# In[10]:


# Create a figure with two subplots (one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')  # Remove x label for the boxplot

# Creating a histogram in the same x-axis
sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust Layout
plt.tight_layout()
plt.show()


# **Observations from boxplot and histograms**
# 
# * There are some extreme values (outliers) observed towards the right tail of SP and HP distributions.
# * In VOL and WT columns, a few outliers are observed in both tails of their distributions.
# * The extreme values of cars data may have come from the specially designed nature of cars.
# * As this is multi-dimensional data, the outliers with respect to spatial dimensions may have to be considered while building the regression model.

# In[11]:


cars[cars.duplicated()]


# In[12]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[13]:


cars.corr()


# 
# 

# In[15]:


model = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
model.summary()


# In[ ]:




