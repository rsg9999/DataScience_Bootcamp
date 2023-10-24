#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Define two custom numpy arrays, say A and B. Generate two new numpy arrays by stacking A and B vertically and horizontally.

import numpy as np

# Create custom NumPy arrays A and B
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

# Stack A and B vertically
vertical_stack = np.vstack((A, B))

# Stack A and B horizontally
horizontal_stack = np.hstack((A, B))

print("Vertically stacked array:")
print(vertical_stack)

print("Horizontally stacked array:")
print(horizontal_stack)




# In[5]:


#Find common elements between A and B. [Hint : Intersection of two sets]

import numpy as np

# Create custom NumPy arrays A and B
A = np.array([1, 2, 3, 4, 5])
B = np.array([3, 4, 5, 6, 7])

# Find the common elements between A and B
common_elements = np.intersect1d(A, B)

print("Common elements between A and B:", common_elements)


# In[6]:


Extract all numbers from A which are within a specific range. eg between 5 and 10. [Hint: np.where() might be useful or boolean masks]

import numpy as np

# Create a custom NumPy array A
A = np.array([2, 6, 8, 3, 10, 12, 7, 9, 4])

# Define the specific range (between 5 and 10)
lower_bound = 5
upper_bound = 10

# Create a boolean mask to identify elements within the range
within_range_mask = (A >= lower_bound) & (A <= upper_bound)

# Extract elements from A that are within the specified range
elements_within_range = A[within_range_mask]

print("Elements within the range [5, 10]:", elements_within_range)


# In[7]:


#IFilter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0
#url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
#iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])


import numpy as np

# Load the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

# Define the conditions
condition = (iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)

# Apply the conditions to filter the rows
filtered_rows = iris_2d[condition]

# Now filtered_rows contains the rows of iris_2d that satisfy the conditions
print(filtered_rows)



# In[8]:


# From df filter the 'Manufacturer', 'Model' and 'Type' for every 20th row starting from 1st (row 0).

import pandas as pd

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# Select every 20th row starting from 1st (row 0)
selected_rows = df.iloc[0::20, :]

# Filter the 'Manufacturer', 'Model' and 'Type' columns
result = selected_rows[['Manufacturer', 'Model', 'Type']]

# Now result contains the filtered data
print(result)


# In[22]:


#2. Replace missing values in Min.Price and Max.Price columns with their respective mean.

import pandas as pd

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# Calculate the mean of Min.Price and Max.Price columns
mean_min_price = df['Min.Price'].mean()
mean_max_price = df['Max.Price'].mean()

# Replace missing values in Min.Price and Max.Price columns with their respective means
df['Min.Price'].fillna(value=mean_min_price, inplace=True)
df['Max.Price'].fillna(value=mean_max_price, inplace=True)

# Now df has the missing values in Min.Price and Max.Price columns replaced with their respective means

df


# In[23]:


#3. How to get the rows of a dataframe with row sum > 100?

import pandas as pd
import numpy as np

# Create the DataFrame
df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))

# Calculate the sum of each row
row_sums = df.sum(axis=1)

# Use boolean indexing to filter rows where the sum is greater than 100
filtered_rows = df[row_sums > 100]

# Now filtered_rows contains the rows of df where the sum of values is greater than 100
print(filtered_rows)


# In[ ]:




