#!/usr/bin/env python
# coding: utf-8

# MEAN
# MEDIAN
# MODE

# In[45]:


ages=[23,24,32,45,12,43,67,45,32,56,32,120]


# In[46]:


import numpy as np
print(np.mean(ages))
print(np.median(ages))


# In[47]:


import statistics
statistics.mode(ages)


# In[48]:


import seaborn as sns
sns.boxplot(ages)


# 5 NUMBER SUMMARY

# In[49]:


import numpy as np
q1,q3=np.percentile(ages,[25,75])


# In[50]:


print(q1,q3)


# In[51]:


IQR=q3-q1
lower_fence=q1-1.5*(IQR)
higher_fence=q3+1.5*(IQR)
print(lower_fence,higher_fence)


#   MEASURE OF DISPERSION
#   1. VARIANCE
#   2. STANDARD DEVIATION
#   

# In[52]:


statistics.variance(ages)


# In[53]:


np.var(ages,axis=0)


# In[54]:


def variance(data):
       n=len(ages)
       ## mean of the data
       mean=sum(data)/n
       ## variance
       deviation=[(x-mean)** 2 for x in data]
       variance=sum(deviation)/n
       return variance


# In[55]:


variance(ages)


# In[56]:


def variance(data):
       n=len(ages)
       ## mean of the data
       mean=sum(data)/n
       ## variance
       deviation=[(x-mean)** 2 for x in data]
       variance=sum(deviation)/(n-1)
       return variance


# In[57]:


variance(ages)


# In[58]:


statistics.pvariance(ages)


# In[59]:


import math
math.sqrt(statistics.pvariance(ages))


# HISTOGRAMS AND PDF
# 

# In[62]:


import seaborn as sns
sns.histplot(ages,kde=True)


# In[64]:


df=sns.load_dataset('iris')


# In[ ]:


df.head()


# CHECK WHETHER DISTRIBUTION IS NORMAL DISTRIBUTION

# In[65]:


### if you want to check whether the distribution is gaussian or normal
## Q-Q Plot

import matplotlib.pyplot as plt
import scipy.stats as stat
import pylab
def plot_data(sample):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    sns.histplot(sample)
    pt.subplot(1,2,2)
    stat.probplot(sample.dist='norm',plot=pylab)
    plt.show()


# In[ ]:




