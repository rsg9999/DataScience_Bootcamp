#!/usr/bin/env python
# coding: utf-8

# In[12]:


#Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 101)


# In[13]:


data = pd.read_csv("/Users/ramshararngoyal/Downloads/train.csv")


# In[14]:


# Dimensions of training data
data.shape


# In[15]:


# Print first few rows of data
data.head()


# In[16]:


# drop id, timestamp and country columns
data = data.drop(columns=['id', 'timestamp','country'])


# In[17]:


# Explore columns
data.columns


# In[18]:


# replace NANs in hours_per_week with median value of the column  
data.loc[data['hours_per_week'].isna(), 'hours_per_week'] = data['hours_per_week'].median()
data.loc[data['telecommute_days_per_week'].isna(), 'telecommute_days_per_week'] = data['telecommute_days_per_week'].median()


# In[19]:


#Handling null values in categorical columns
data = data.dropna()


# In[20]:


data.info()


# In[21]:


# create another copy of dataset and append encoded features to it
data_train = data.copy()
data_train.head()


# In[22]:


# select categorical features
cat_cols = [c for c in data_train.columns if data_train[c].dtype == 'object' 
            and c not in ['is_manager', 'certifications']]
cat_data = data_train[cat_cols]
cat_cols


# In[23]:


#Encoding binary variables
binary_cols = ['is_manager', 'certifications']
for c in binary_cols:
    data_train[c] = data_train[c].replace(to_replace=['Yes'], value=1)
    data_train[c] = data_train[c].replace(to_replace=['No'], value=0)


# In[24]:


final_data = pd.get_dummies(data_train, columns=cat_cols, drop_first= True)
final_data.shape


# In[25]:


final_data.columns


# In[26]:


final_data


# In[27]:


y = final_data['salary']
X = final_data.drop(columns=['salary'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("Training Set Dimensions:", X_train.shape)
print("Validation Set Dimensions:", X_test.shape)


# In[28]:


# select numerical features
num_cols = ['job_years','hours_per_week','telecommute_days_per_week']
num_cols


# In[29]:


# Apply standard scaling on numeric data 
scaler = StandardScaler()
scaler.fit(X_train[num_cols])
X_train[num_cols] = scaler.transform(X_train[num_cols])


# In[31]:


reg=LinearRegression()
reg.fit(X_train, y_train)


# In[32]:


reg.coef_


# In[33]:


mean_absolute_error(y_train,reg.predict(X_train))


# In[34]:


mean_squared_error(y_train,reg.predict(X_train))**0.5


# In[35]:


#Q1 and Q2
X_test[num_cols] = scaler.transform(X_test[num_cols])
y_pred = reg.predict(X_test)
print(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test,y_pred)**0.5)


# In[36]:


X_test.describe()


# In[37]:


#Q3
ridge = Ridge(alpha=1)
ridge.fit(X_train,y_train)
y_pred = ridge.predict(X_test)
print(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test,y_pred)**0.5)

plt.scatter(np.arange(len(np.sort(y_test))),np.sort(y_test), label='true')
plt.scatter(np.arange(len(np.sort(y_pred))),np.sort(y_pred), label = 'pred')
plt.legend()


# In[38]:


ridge.coef_


# In[39]:


lasso = Lasso(alpha=1)
lasso.fit(X_train,y_train)
y_pred = lasso.predict(X_test)
print(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test,y_pred)**0.5)

plt.scatter(np.arange(len(np.sort(y_test))),np.sort(y_test))
plt.scatter(np.arange(len(np.sort(y_pred))),np.sort(y_pred))


# In[40]:


lasso.coef_


# In[41]:


# train Decision Tree regression model
decisiontree = DecisionTreeRegressor(max_depth = 10, min_samples_split = 5)
decisiontree.fit(X_train, y_train)

#evaluating train error
mean_absolute_error(y_train,decisiontree.predict(X_train))


# In[42]:


#evaluating test error
mean_absolute_error(y_test,decisiontree.predict(X_test))


# In[43]:


max_depth_list = [2,3,4,5,6,7,8,9,10,11,12,20]
train_error = []
test_error =[]

for md in max_depth_list:

    decisiontree = DecisionTreeRegressor(max_depth = md, min_samples_split = 2)
    decisiontree.fit(X_train, y_train)
    train_error.append(mean_absolute_error(y_train,decisiontree.predict(X_train)))
    test_error.append(mean_absolute_error(y_test,decisiontree.predict(X_test)))

plt.plot(max_depth_list,train_error,label = 'train error')
plt.plot(max_depth_list,test_error,label = 'test error')
plt.legend()


# In[ ]:




