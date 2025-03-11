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


# **Observations from correlation plots and Coefficients**
# 
# * Between x and y, all the x variables are showing moderate to high correlation strengths, highest being between HP and MPG.
# * Therefore, this dataset qualifies for building a multiple linear regression model to predict MPG.
# * Among x columns (x1, x2, x3, and x4), some very high correlation strengths are observed between SP vs HP, VOL vs WT.
# * The high correlation among x columns is not desirable as it might lead to multicollinearity problems.

# In[14]:


model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
model1.summary()


# **Observations from model summary**
# - The R-squared and abjusted R-squared values are good and about 75% of varibility in y is explained by x columns
# - The probability value with respect to F-statistic is close to zero, indicating that all or someof x columns are significant
# - The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves, which need to be further explored
# 

# In[15]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[16]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[17]:


from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))


# **Checking for multicollinearlity x-columns using  VIF method**

# In[18]:


cars.head()


# In[19]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# **OBSERVATIONS**
# * The ideal range of VIF values shall be b/w 0 to 10. However slightly higher values can be tolearted
# * As seen from the higher VIF values for VOL and WT , it is clear that they are prone to multicollinearity properties
# * Hence is decided to drop one of column to overcome the multilinerality
# * it is decided to drop WT and retain VOL column in futher models

# In[20]:


cars1=cars.drop("WT",axis=1)
cars.head()


# In[21]:


import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()
model2.summary()


# **Performance metrics for model2**

# In[22]:


df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[23]:


pred_y2 = model2.predict(cars.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[24]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))


# #### Oservations of model2 summary()
# - The adjusted R-squared value improved slightly to 0.76
# - All the p-values for model parameters are less than 5% hence they are significant
# - Therefore the HP,VOL,SP columns are finalized as the significant predictor for the MPG
# - There is no improvement in MSE Value

# In[25]:


cars1.shape


# In[26]:


k = 3
n = 81
leverage_cutoff = 3*((k+1)/n)
leverage_cutoff


# In[27]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model1,alpha=0.5)
y=[i for i in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y, 'r+')
plt.show()


# **Obsevations**
# * from the above plot,it is evident that data points 65,70,76,78,79,80 are the influencers
# * as their H levarage values are high and size is higher

# In[36]:


cars2=cars1[cars1.index.isin([65,70,76,78,79,80])]


# In[34]:


cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)
cars2


# **Build Model3 on cars2 dataset**

# In[37]:


model3=smf.ols('MPG~VOL+SP+HP',data = cars2).fit()


# In[38]:


model3.summary()


# ***Performance Metrics for model3***

# In[39]:


df3=pd.DataFrame()
df3["actual_y3"]=cars2["MPG"]
df3.head()


# In[45]:


pred_y3 = model2.predict(cars.iloc[:,0:4])
df3["pred_y3"] = pred_y3
df3.head()


# In[46]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"], df3["pred_y3"])
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))


# 
# #### Comparison of models
# 
# | Metric          | Model 1 | Model 2 | Model 3 |
# |----------------|---------|---------|---------|
# | R-squared      | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE           | 18.89   | 18.91   | 8.68    |
# | RMSE          | 4.34    | 4.34    | 2.94    |

# -**From the above comparision table it is observed that model3 is the best among all with superior performances**

# ***Check the validity of model assumption***

# In[47]:


model3.resid


# In[48]:


model3.fittedvalues


# In[51]:


import statsmodels.api as sm
qqplott = sm.qqplot(model3.resid,line='q')
plt.title("Normal Q-Q plot of residuals")


# In[54]:


sns.displot(model3.resid, kde = True)


# In[55]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[59]:


plt.figure(figsize=(6,4))
plt.scatter(get_standardized_values(model3.fittedvalues),
            get_standardized_values(model3.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[ ]:





# In[ ]:




