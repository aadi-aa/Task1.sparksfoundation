#!/usr/bin/env python
# coding: utf-8

# # GRIP JANUARY''22  THE SPARKS FOUNDATION
# 
#   **DATA SCIENCE AND BUSSINESS ANALYTICS INTERN **
#  
#  Name -- Aditya kumar
#  
#  Task 1 Prediction using supervised machine learning
#   

# # Linear Regression with Python Scikit Learn

# In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables.

# # Simple Linear Regression

# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# Description :: To predict the percentage of an student based on the no of study hours.

# # IMPORTING ALL THE REQUIRED LIBRARY

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  


# # IMPORT DATASET

# In[2]:


url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# In[3]:


s_data.shape


# In[4]:


s_data.columns


# In[5]:


s_data.describe()


# # CHECKING FOR NULL VALUES

# In[6]:


s_data.isnull().sum()


# # SCORE DISTRIBUTION PLOT

# In[7]:


s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# # Preparing the data

# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[8]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values


# Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[9]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# # Training the Algorithm

# In[10]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 


# # PLOTTING THE REGRESSION LINE

# In[11]:


line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# # MAKING PREDICTION

# Now that we have trained our algorithm, it's time to make some predictions.

# In[12]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# # COMPARISON BETWEEN ACTUAL AND PREDICTED

# In[13]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# # Predicting score of student based on based on hour studied

# Here our aim is to predict the score of student if he/she studies for 9.25 hours

# In[14]:


hours = [[9.25]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # Evaluating the model

# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.

# In[15]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




