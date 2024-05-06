# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Loading the dataset
ds = pd.read_excel('./Projects/Online Retail.xlsx')
ds.head()

# Data cleaning
print(ds.info()) # Display information about the dataset
print(ds.shape) # Display the shape of the dataset
print(ds.isnull().sum()) # Check for missing values
ds = ds.dropna() # Drop rows with missing values
print(ds.info())  # Display updated information after dropping missing values
print(ds.shape) # Display the shape of the dataset after dropping missing values

# Data processing
ds['CustomerID'] = ds['CustomerID'].astype(str) # Convert CustomerID to string
ds['Amount'] = ds['Quantity']*ds['UnitPrice'] # Calculate the total amount spent by customer
rfm_ds_n = ds.groupby('CustomerID')['Amount'].sum() # Group by CustomerID and calculate total amount
rfm_ds_n.reset_index() # Reset index
rfm_ds_n.columns = ['CustomerID', 'Amount'] # Rename columns
print(rfm_ds_n)

rfm_ds_f = ds.groupby('CustomerID')['InvoiceNo'].count()  # Group by CustomerID and count number of invoices
rfm_ds_f = rfm_ds_f.reset_index() # Rest Index
rfm_ds_f.columns = ['CustomerID','Frequency'] # Rename columns
print(rfm_ds_f)

ds['InvoiceDate'] = pd.to_datetime(ds['InvoiceDate'],format='%d-%m-%Y %H:%M')  # Convert InvoiceDate to datetime
max_date = max(ds['InvoiceDate']) # Find the maximum date
ds['Diff'] = max_date - ds['InvoiceDate'] # Calculate the difference from the maximum date
rfm_ds_p = ds.groupby('CustomerID')['Diff'].min() # Group by CustomerID and find minimum difference
rfm_ds_p = rfm_ds_p.reset_index() # Reset index
rfm_ds_p.columns = ['CustomerID', 'Diff'] #Rename columns
rfm_ds_p['Diff'] = rfm_ds_p['Diff'].dt.days # Convert timedelta to days
print(rfm_ds_p)

# Merge all three RFM metrics into one dataframe
rfm_ds_final = pd.merge(rfm_ds_n, rfm_ds_f, on='CustomerID',how='inner') 
rfm_ds_final = pd.merge(rfm_ds_final, rfm_ds_p, on='CustomerID', how='inner')
rfm_ds_final.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency'] # Rename columns
print(rfm_ds_final)

#Outliers removal
Q1 = rfm_ds_final.Amount.quantile(0.05) # 5th percentile of Amount
Q3 = rfm_ds_final.Amount.quantile(0.95) # 95th percentile of Amount
IQR = Q3-Q1  # Interquartile range
rfm_ds_final = rfm_ds_final[(rfm_ds_final.Amount > Q1 - 1.5*IQR) & (rfm_ds_final.Amount < Q3 + 1.5*IQR)]  # Remove outliers in Amount

Q1 = rfm_ds_final.Recency.quantile(0.05) # 5th percentile of Recency
Q3 = rfm_ds_final.Recency.quantile(0.95) # 95th percentile of Recency
IQR = Q3-Q1 # Interquartile range
rfm_ds_final = rfm_ds_final[(rfm_ds_final.Recency > Q1 - 1.5*IQR) & (rfm_ds_final.Recency < Q3 + 1.5*IQR)] # Remove outliers in Recency

Q1 = rfm_ds_final.Frequency.quantile(0.05) # 5th percentile of Recency
Q3 = rfm_ds_final.Frequency.quantile(0.95) # 95th percentile of Frequency
IQR = Q3-Q1 # Interquartile range
rfm_ds_final = rfm_ds_final[(rfm_ds_final.Frequency > Q1 - 1.5*IQR) & (rfm_ds_final.Frequency < Q3 + 1.5*IQR)] # Remove outliers in Frequency

print(rfm_ds_final.shape)  # Display the shape of the dataset after removing outliers

# Scaling the data
X = rfm_ds_final[['Amount', 'Frequency', 'Recency']] # Selecting features for scaling
scaler = MinMaxScaler() # Initialize MinMaxScaler
rfm_ds_scaled = scaler.fit_transform(X) # Scale the data

rfm_ds_scaled = pd.DataFrame(rfm_ds_scaled) # Convert scaled data to DataFrame
rfm_ds_scaled.columns = ['Amount', 'Frequency','Recency']  # Rename columns
rfm_ds_scaled.head()

# Model creation using KMeans
kmeans = KMeans(n_clusters= 3,max_iter= 50) # Initialize KMeans with 3 clusters
kmeans.fit(rfm_ds_scaled)  # Fit KMeans to the scaled data
lbs = kmeans.labels_ # Get cluster labels
print(kmeans.labels_) # Display cluster labels

# Elbow method to find optimal number of clusters (within-cluster sum of squares)
wss =[] # List to store within-cluster sum of squares
range_n_clusters = [2, 3, 4, 5, 6, 7, 8] # Range of clusters to try
for num_clusters in range_n_clusters: 
    kmeans = KMeans(n_clusters= num_clusters, max_iter= 50) # Initialize KMeans with varying number of clusters
    kmeans.fit(rfm_ds_scaled) # Fit KMeans to the scaled data
    wss.append(kmeans.inertia_) # Append within-cluster sum of squares to the list

plt.plot(wss) # Plot within-cluster sum of squares

# Silhouette score to evaluate clustering quality
range_n_clusters = [2, 3, 4, 5, 6, 7, 8] # Range of clusters to try
for num_clusters in range_n_clusters: 
    kmeans = KMeans(n_clusters= num_clusters, max_iter= 50) # Initialize KMeans with varying number of clusters
    kmeans.fit(rfm_ds_scaled) # Fit KMeans to the scaled data
    cluster_labels = kmeans.labels_ # Get cluster labels
    silhouette_avg = silhouette_score(rfm_ds_scaled, cluster_labels) # Calculate silhouette score
    print('For n_clusters(0), the silhouette score is {1}'.format(num_clusters, silhouette_avg)) # Print silhouette score

#kmeans = KMeans(n_clusters= 3,max_iter= 50)
#kmeans.fit(rfm_ds_scaled)
#print(kmeans.labels_)

# Assigning cluster labels to the original dataframe
rfm_ds_final['ClusterID'] = lbs # Add ClusterID column to the original dataframe
rfm_ds_final.head()

# Visualizing clusters using boxplots
sns.boxplot(x= 'ClusterID', y= 'Amount', data= rfm_ds_final) # Boxplot of Amount by ClusterID

sns.boxplot(x= 'ClusterID', y= 'Frequency', data= rfm_ds_final) # Boxplot of Frequency by ClusterID

sns.boxplot(x= 'ClusterID', y= 'Recency', data= rfm_ds_final) # Boxplot of Recency by ClusterID



