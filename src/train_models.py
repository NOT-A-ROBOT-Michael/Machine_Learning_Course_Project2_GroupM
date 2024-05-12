"begin"
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score
import pandas as pd

df = pd.DataFrame()
"""function to create the model"""
def getClusters(df):

    #model creation
    kmeans = KMeans(n_clusters= 3,max_iter= 50)
    kmeans.fit(df)
    lbs = kmeans.labels_

    "elbow-method"
    #appendin inertia
    #wss
    wss =[]
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters= num_clusters, max_iter= 50)
        kmeans.fit(df)
        wss.append(kmeans.inertia_)
    #silhouette score
    n_cluster=0
    silhouette_no=0
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters= num_clusters, max_iter= 50)
        kmeans.fit(df)
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(df, cluster_labels)
        print('For n_clusters{0}, the silhouette score is {1}'.format(num_clusters, silhouette_avg))
        if silhouette_avg>silhouette_no:
            silhouette_no=silhouette_avg
            n_cluster=num_clusters
    return n_cluster

"returns the final data-model"
def create_finalmodel(df):
    # base model
    df_scaled = df
    # final_model labels
    final_model = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=3, random_state=42)
    )
    # Predict class labels
    cluster = final_model.fit_predict(df_scaled)

    df_scaled['Cluster'] = cluster
    return df_scaled
"end"