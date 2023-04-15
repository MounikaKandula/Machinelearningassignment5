#!/usr/bin/env python
# coding: utf-8

# # Question1:
# Principal Component Analysis 
# a. Apply PCA on CC dataset.
# b.	Apply k-means algorithm on the PCA result and report your observation if the silhouette score has.
#     improved or not?
# c.	Perform Scaling+PCA+K-Means and report performance.
# 

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv('/Users/mounikakandula/Downloads/datasets/CC GENERAL.csv')


# Droping rows with missing values
data.dropna(inplace=True)

# Standardizing the features
scaler = StandardScaler()
X = scaler.fit_transform(data.drop('CUST_ID', axis=1))


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

principalData = pd.DataFrame(data = X_pca, columns = ['PCA 1', 'PCA 2'])
finalData = pd.concat([principalData, data[['TENURE']]], axis = 1)
finalData.head()


# In[2]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=2, random_state=42)

#Applying KMeans to the PCA result
kmeans.fit(X_pca)
y_kmeans = kmeans.predict(X_pca)

silhouette_score(X_pca, y_kmeans)


# In[3]:


# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop('CUST_ID', axis=1))

# Applying PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply k-means to PCA
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_pca)
y_kmeans = kmeans.predict(X_pca)

# Evaluating the performance
silhouette_score(X_pca, y_kmeans)


# # Question 2:
# 
# Use pd_speech_features.csv 
# a. Perform Scaling
# b.	Apply PCA (k=3)
# c.	Use SVM to report performance
# 

# In[4]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

#loading the dataset into dataframes
dataSet = pd.read_csv('/Users/mounikakandula/Downloads/datasets/pd_speech_features.csv')

#Separating the features(X) from target variable(y)
X = dataSet.iloc[:, 1:-1].values
y = dataSet.iloc[:, -1].values

# applying scaling to the features
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[5]:


from sklearn.decomposition import PCA

# applying PCA to reduce the dimensionality of the dataset
pca = PCA(n_components=3)
X = pca.fit_transform(X)
principalData = pd.DataFrame(data = X, columns = ['PCA 1', 'PCA 2', 'PCA 3'])

finalDf = pd.concat([principalData, dataSet[['class']]], axis = 1)
finalDf.head()


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# splitting the training and testing sets using the train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# using SVM model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# calculating the performance by calculating accuracy, precision, recall and f1
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: ", acc)
print("Precision: ", prec)
print("Recall: ", rec)
print("F1 score: ", f1)


# # Question:3
# Apply Linear Discriminant Analysis (LDA) on Iris.csv dataset to reduce dimensionality of data tok=2.

# In[7]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# loading the iris dataset from CSV file
dataSet = pd.read_csv('/Users/mounikakandula/Downloads/datasets/Iris.csv')

# separating the features and target variables
X = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values

# standardizing the features
sc = StandardScaler()
X = sc.fit_transform(X)

# applying LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)
import matplotlib.pyplot as plt

# visualizing the reduced-dimensional data
plt.figure(figsize=(10, 8))
for species in set(y):
    plt.scatter(X_lda[y == species, 0], X_lda[y == species, 1], label=species)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()

