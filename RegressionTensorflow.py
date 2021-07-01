# ## Simple Regression Example

# In[80]:
import pandas as pd
import numpy as np

x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10) #create data in for X axis utilizing random numbers added to a matrix


# In[81]:


x_data


# In[82]:


y_label = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5, 10) #create data in for Y axis utilizing random numbers added to a matrix


# In[83]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[84]:


plt.plot(x_data, y_label, '*') #creates the scatterplot graph


# # y = mx + b 
# This is the function which implements the equation

# In[85]:


np.random.rand(2)


# In[86]:


m = tf.Variable(0.27)
b = tf.Variable(0.67)


# In[87]:


error = 0

for x,y in zip(x_data, y_label):
    
    y_hat = m * x + b
    
    error += (y - y_hat)**2


# In[88]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

train = optimizer.minimize(error)


# In[89]:


init = tf.global_variables_initializer()


# In[90]:


with tf.Session() as sess:
    
    sess.run(init)
    
    training_steps = 100
    
    for i in range(training_steps):
        
        sess.run(train)
        
    final_slope, final_intercept = sess.run([m, b])


# In[91]:


x_test = np.linspace(-1, 11, 10)

#y = mx + b
y_pred_plot = final_slope * x_test + final_intercept

plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')
