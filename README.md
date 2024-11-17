# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**: Import necessary libraries (`pandas`, `KMeans`, `StandardScaler`, `matplotlib`).

2. **Load Data**: Load the dataset using `pandas.read_csv()`.

3. **Select Features**: Extract features: 'Annual Income (k$)' and 'Spending Score (1-100)'.

4. **Scale Data**: Standardize the features using `StandardScaler`.

5. **Determine Optimal K**: Use the Elbow Method (plot WCSS) to find the optimal number of clusters.

6. **K-Means Clustering**: Perform K-Means clustering with `K=5` (optimal clusters).

7. **Assign Cluster Labels**: Add the cluster labels to the dataset.

8. **Visualize Clusters**: Create a scatter plot of the clusters using `Annual Income` and `Spending Score`.
## Program:
```
/*
Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: SUBHASHINI.B  
RegisterNumber:  2122233040211
*/
# Import necessary libraries  
import pandas as pd  
from sklearn.cluster import KMeans  
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt  
  
# Load the Mall Customers dataset  
df = pd.read_csv(r'C:\Users\admin\Downloads\CustomerData.csv')  
  
# Select relevant features for clustering (Annual Income and Spending Score)  
features = df[['Annual Income (k$)', 'Spending Score (1-100)']]  
  
# Scale the data using StandardScaler  
scaler = StandardScaler()  
scaled_features = scaler.fit_transform(features)  
  
# Determine the optimal number of clusters (K) using the Elbow Method  
wcss = []  
for i in range(1, 11):  
 kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  
 kmeans.fit(scaled_features)  
 wcss.append(kmeans.inertia_)  
  
plt.plot(range(1, 11), wcss)  
plt.title('Elbow Method')  
plt.xlabel('Number of Clusters')  
plt.ylabel('WCSS')  
plt.show()  
  
# Perform K-Means clustering with the optimal number of clusters (K=5)  
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)  
kmeans.fit(scaled_features)  
  
# Predict the cluster labels for each customer  
labels = kmeans.labels_  
  
# Add the cluster labels to the original dataset  
df['Cluster'] = labels  
  
# Visualize the clusters using a scatter plot  
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='viridis')  
plt.title('Customer Segmentation using K-Means Clustering')  
plt.xlabel('Annual Income (k$)')  
plt.ylabel('Spending Score (1-100)')  
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![image](https://github.com/user-attachments/assets/2c569b13-ee94-4663-8166-d320b8af945d)
![image](https://github.com/user-attachments/assets/180dee3a-491f-441f-a181-d8ee46c10661)


## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
