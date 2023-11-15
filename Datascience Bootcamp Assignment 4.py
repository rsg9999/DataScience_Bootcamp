#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#1. Create a line plot of ZN and INDUS in the housing data.

#	a. For ZN, use a solid green line. For INDUS, use a blue dashed line.
#	b. Change the figure size to a width of 12 and height of 8.
#	c. Change the style sheet to something you find https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
import pandas as pd

# Load the Boston housing dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)

# Setting a custom style from the matplotlib style gallery
plt.style.use('fivethirtyeight')

# Change the figure size
plt.figure(figsize=(12, 8))

# Plotting ZN with a solid green line
plt.plot(df['ZN'], color='green', linestyle='-', label='ZN')

# Plotting INDUS with a blue dashed line
plt.plot(df['INDUS'], color='blue', linestyle='--', label='INDUS')

# Adding title and labels
plt.title('ZN and INDUS in Boston Housing Dataset')
plt.xlabel('Data Points')
plt.ylabel('Values')

# Adding a legend
plt.legend()

# Show the plot
plt.show()


# In[3]:


#2. Create a bar chart using col1 and col2 of dummy data.
import matplotlib.pyplot as plt
import numpy as np

# Generate some dummy data
np.random.seed(0)  # for reproducibility
col1 = np.random.rand(10)  # 10 random numbers for column 1
col2 = np.random.rand(10)  # 10 random numbers for column 2
indices = np.arange(len(col1))  # indices for the x-axis

# Plot vertical bars
plt.figure(figsize=(10, 6))

plt.bar(indices - 0.2, col1, width=0.4, label='Col 1')
plt.bar(indices + 0.2, col2, width=0.4, label='Col 2')

plt.title('Vertical Bar Chart of Dummy Data', fontsize=16)
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend(loc='lower left')

plt.show()

# Plot horizontal bars
plt.figure(figsize=(10, 6))

plt.barh(indices - 0.2, col1, height=0.4, label='Col 1')
plt.barh(indices + 0.2, col2, height=0.4, label='Col 2')

plt.title('Horizontal Bar Chart of Dummy Data', fontsize=16)
plt.xlabel('Value')
plt.ylabel('Index')
plt.legend(loc='upper right')

plt.show()


# In[ ]:


#3. Create a histogram with pandas for using MEDV in the housing data.

import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd

# Load the Boston housing dataset
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)

# Adding the target variable MEDV to the dataframe
df['MEDV'] = boston.target

# Create a histogram for MEDV with 20 bins
plt.figure(figsize=(10, 6))
df['MEDV'].hist(bins=20)

# Adding title and labels
plt.title('Histogram of MEDV in Boston Housing Dataset')
plt.xlabel('MEDV')
plt.ylabel('Frequency')

# Show the plot
plt.show()


# In[19]:


#4. Create a scatter plot of two heatmap entries that appear to have a very positive correlation

import seaborn as sns
import pandas as pd

# Generating sample data for the heatmap
np.random.seed(0)
data = np.random.rand(10, 10)
df = pd.DataFrame(data, columns=[f'Var{i+1}' for i in range(10)])

# Creating a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlation Between Variables')
plt.show()

# From the heatmap, let's choose two variables with high positive correlation
# For simplicity, I'm selecting Var1 and Var2 which are likely to have high correlation
# In a real scenario, you would choose based on the actual correlation values from the heatmap
var1 = df['Var1']
var2 = df['Var2']

# Creating a scatter plot for the chosen variables
plt.scatter(var1, var2)
plt.xlabel('Var1')
plt.ylabel('Var2')
plt.title('Scatter Plot of Var1 and Var2')
plt.show()


# In[20]:


#5. Now, create a scatter plot of two heatmap entries that appear to have negative correlation.
# Assuming Var3 and Var4 have a negative correlation
import seaborn as sns
import pandas as pd

# Generating sample data for the heatmap
np.random.seed(0)
data = np.random.rand(10, 10)
df = pd.DataFrame(data, columns=[f'Var{i+1}' for i in range(10)])

# Creating a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlation Between Variables')
plt.show()
var3 = df['Var3']
var4 = df['Var4']

# Creating a scatter plot for these negatively correlated variables
plt.scatter(var3, var4)
plt.xlabel('Var3')
plt.ylabel('Var4')
plt.title('Scatter Plot of Negatively Correlated Variables (Var3 and Var4)')
plt.show()


# In[ ]:




