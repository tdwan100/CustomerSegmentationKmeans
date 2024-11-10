# Customer Segmentation using K-Means Clustering

This project implements customer segmentation on a retail dataset using the K-Means clustering algorithm. The segmentation is based on customer `Annual Income` and `Spending Score`, and aims to identify groups of customers with similar spending behaviors. By visualizing these clusters, businesses can better target different customer groups.

## Project Overview

### Purpose
The goal is to segment customers for targeted marketing based on their income and spending patterns. Clustering helps identify distinct customer profiles, making it easier to create marketing strategies that cater to specific groups.

### Dataset
The dataset (`Mall_Customers.csv`) includes 200 entries with the following columns:
- `CustomerID`: Unique identifier for each customer
- `Gender`: Gender of the customer
- `Age`: Age of the customer
- `Annual Income (k$)`: Annual income in thousands of dollars
- `Spending Score (1-100)`: Score assigned by the mall based on customer spending behavior

## Getting Started

### Prerequisites
Make sure to install the following libraries:
```python
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Code
1. **Load Data**  
   Load the data from `Mall_Customers.csv` using `pandas`.
   
2. **Data Preprocessing**  
   We select `Annual Income (k$)` and `Spending Score (1-100)` as the features to cluster.

3. **Elbow Method for Optimal Clusters**  
   Using the Within-Cluster Sum of Squares (WCSS) approach, we plot the Elbow Curve to determine the optimal number of clusters.

4. **K-Means Clustering**  
   We train the K-Means model with the optimal number of clusters (k=5) as determined from the elbow plot.

5. **Cluster Visualization**  
   We visualize the clusters and their centroids to observe distinct customer groups.

## Code Walkthrough

### Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
```

### Data Loading
```python
data = pd.read_csv('Mall_Customers.csv')
```

### Elbow Method
```python
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
```

### K-Means Model Training
```python
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
y = kmeans.fit_predict(X)
```

### Cluster Visualization
```python
plt.figure(figsize=(10,10))
plt.scatter(X[y == 0, 0], X[y == 0, 1], s=50, c='green', label='Cluster 1')
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=50, c='pink', label='Cluster 2')
plt.scatter(X[y == 2, 0], X[y == 2, 1], s=50, c='blue', label='Cluster 3')
plt.scatter(X[y == 3, 0], X[y == 3, 1], s=50, c='yellow', label='Cluster 4')
plt.scatter(X[y == 4, 0], X[y == 4, 1], s=50, c='brown', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')
plt.title('Customer Clusters')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
```

## Results

The K-Means model segmented customers into five clusters based on their income and spending score. By analyzing the clusters, we can identify customer groups such as high-spending individuals with high income or low-spending individuals with low income.

## License
This project is licensed under the MIT License.

---
