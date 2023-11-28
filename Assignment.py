#!/usr/bin/env python
# coding: utf-8

# In[115]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.metrics import pairwise_distances
from sklearn import cluster, datasets, preprocessing, metrics
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

glass = pd.read_csv('/Users/ramshararngoyal/Downloads/glass.csv')
glass.head()


# In[116]:


glass.Type.value_counts().sort_index()


# In[117]:


glass['household'] = glass.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})
glass.household.value_counts()


# In[118]:


glass.sort_values( by = 'Al', inplace=True)
X= np.array(glass.Al).reshape(-1,1)
y = glass.household


# In[119]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X,y)
pred = logreg.predict(X)
logreg.coef_, logreg.intercept_


# In[120]:


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:





# In[121]:


# Function to evaluate model at different thresholds
def evaluate_threshold(threshold, y_test, y_pred_prob):
    y_pred_custom = (y_pred_prob >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_custom)
    precision = precision_score(y_test, y_pred_custom)
    recall = recall_score(y_test, y_pred_custom)
    return accuracy, precision, recall


# In[122]:


# Plotting results for binary classification
plt.figure(figsize=(10, 6))
accuracy_scores, precision_scores, recall_scores = zip(*results)
plt.plot(thresholds, accuracy_scores, label='Accuracy')
plt.plot(thresholds, precision_scores, label='Precision')
plt.plot(thresholds, recall_scores, label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Model Performance at Different Thresholds')
plt.legend()
plt.grid(True)
plt.show()


# In[123]:


# Fit a Binary Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)





# In[124]:


thresholds = np.arange(0.1, 1, 0.1)
y_pred_prob = logreg.predict_proba(X_test_scaled)[:, 1]
results = [evaluate_threshold(t, y_test, y_pred_prob) for t in thresholds]


# In[125]:


# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[126]:


# Check out the dataset and our target values
df = pd.read_csv('/Users/ramshararngoyal/Downloads/iris.csv')
print(df['Name'].value_counts())
df.head(5)


# In[127]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[128]:


cols = df.columns[:-1]
sns.pairplot(df[cols])


# In[129]:


X_scaled = preprocessing.MinMaxScaler().fit_transform(df[cols])


# In[130]:


pd.DataFrame(X_scaled, columns=cols).describe()


# In[131]:


from sklearn.metrics import silhouette_score
k_values = range(2, 11)
ssd = []
silhouette_scores = [] # Initializing the list

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    ssd.append(kmeans.inertia_)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    
# Plot the silhouette scores for each K
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Values of K')
plt.show()
    
    

# Plot the SSD for each K
plt.plot(k_values, ssd, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal K')
plt.show()



# In[132]:


labels = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_


# In[133]:


df['label'] = labels
df.head()


# In[134]:


cols = df.columns[:-2]
sns.pairplot(df, x_vars=cols, y_vars= cols, hue='label')


# In[135]:


sns.pairplot(df, x_vars=cols, y_vars= cols, hue='Name')


# In[ ]:





# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('/Users/ramshararngoyal/Downloads/nutrients.csv')
df.head(5)
cols = df.columns[:-1]
sns.pairplot(df[cols])
X_scaled = preprocessing.MinMaxScaler().fit_transform(df[cols])
pd.DataFrame(X_scaled, columns=cols).describe()
from sklearn.metrics import silhouette_score
k_values = range(2, 11)
ssd2 = []
silhouette_scores2 = [] # Initializing the list

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    ssd2.append(kmeans.inertia_)
    labels = kmeans.labels_
    silhouette_scores2.append(silhouette_score(X_scaled, labels))

print(silhouette_scores2)   
# Plot the silhouette scores for each K
plt.plot(k_values, silhouette_scores2, marker='o')
plt.xlabel('Number of clusters2 (K)')
plt.ylabel('Silhouette Score2')
plt.title('Silhouette Scores for Different Values of K2')
plt.show()
    
    

# Plot the SSD for each K
plt.plot(k_values, ssd2, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal K')
plt.show()


# In[ ]:





# In[ ]:




