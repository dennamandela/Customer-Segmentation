#!/usr/bin/env python
# coding: utf-8

# ### Import Library

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ### Loading Data

# In[8]:


dataset = pd.read_csv("Dataset Customer.csv")
dataset


# ### Visualisasi Data

# In[10]:


plt.scatter(x=dataset["Satisfaction"],y=dataset["Loyalty"])
plt.show()


# ### KMeans Clustering

# In[11]:


kmeans = KMeans(n_clusters=3).fit(dataset)


# In[12]:


clusters = kmeans.fit_predict(dataset)


# In[13]:


clusters


# In[14]:


dataset_sudah_tercluster = dataset.copy()
dataset_sudah_tercluster["Cluster"] = clusters


# In[15]:


dataset_sudah_tercluster


# In[16]:


plt.scatter(x=dataset_sudah_tercluster["Satisfaction"],y=dataset_sudah_tercluster["Loyalty"],
            c=dataset_sudah_tercluster["Cluster"],cmap="tab10")
plt.show()

