#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[2]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[3]:


Univ.info()


# In[4]:


Univ.isna().sum()


# In[5]:


Univ.describe()


# ***Standardization of data***

# In[6]:


Univ1 = Univ.iloc[:,1:]


# In[7]:


Univ1


# In[8]:


cols = Univ1.columns


# In[14]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df


# In[20]:


from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[19]:


clusters_new.labels_


# In[22]:


set(clusters_new.labels_)


# In[24]:


Univ['clusterid_new']=clusters_new.labels_
Univ


# In[25]:


Univ[Univ['clusterid_new']==1]


# In[28]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# In[ ]:





# #### Observations:
# - Cluster 2 appears to be the top rated universities cluster as the cut off score,Top10,SFratio parameter mean values high.
# - Cluster 1 appears to occupy the middle level rated universities,
# - Cluster 0 comes as the lower level rated universities 
# 

# In[30]:


Univ[Univ['clusterid_new']==0]


# #### Finding optimal k value using elbow plot

# In[32]:


wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(scaled_Univ_df)
    # kmeans.fit(Univ1)
    wcss.append(kmeans.inertia_)

print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:




