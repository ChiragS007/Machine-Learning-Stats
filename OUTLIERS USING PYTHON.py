#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset= [11,10,12,14,12,15,14,13,15,102,12,14,17,19,107,10,13,12,14,12,108,12,11,14,13,15,10,15,12,10,14,13,15,10]


# In[3]:


plt.hist(dataset)


# In[4]:


#Z-score 
outliers=[]
def detect_outliers(data):
    threshold=3 ## 3rd standard deviation
    mean=np.mean(data)
    std=np.std(data)
    
    for i in data:
        z_score=(i-mean)/std
        if np.abs(z_score)>threshold:
            outliers.append(i)
    return outliers


# In[5]:


detect_outliers(dataset)


# ##IQR 
# 1. Sort the data
# 2. Calculate Q1(25%) and Q3(75%)
# 3. IQR(Q3-Q1)
# 4. Find the lower Fence(q1-1.5(iqr))
# 5. Find the Upper Fence(q3+1.5(iqr))

# In[6]:


##sort
dataset=sorted(dataset)


# In[7]:


dataset


# In[8]:


q1,q3=np.percentile(dataset,[25,75])
print(q1,q3)


# In[9]:


iqr=q3-q1
print(iqr)


# In[10]:


##Find the lower fence and upper fence

lower_fence=q1-(1.5*iqr)
upper_fence=q3+(1.5*iqr)
print(lower_fence,upper_fence)


# In[11]:


import seaborn as sns


# In[12]:


sns.boxplot(dataset)


# In[ ]:




