# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import streamlit as st

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Task 1: Apply K-Means clustering and visualize the clusters
def kmeans_clustering():
    st.title("K-Means Clustering on Iris Dataset")
    st.write("### Apply K-Means clustering to the Iris dataset and visualize the clusters using a scatter plot of two features.")

    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(data.iloc[:, :-1])

    fig, ax = plt.subplots()
    sns.scatterplot(x=data['sepal length (cm)'], y=data['sepal width (cm)'], hue=data['cluster'], palette='viridis', ax=ax)
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('K-Means Clustering of Iris Dataset')
    st.pyplot(fig)

# Task 2: Use the Elbow Method and Silhouette Score to determine the optimal number of clusters
def optimal_clusters():
    st.title("Choosing the Optimal Number of Clusters")
    st.write("### Use the Elbow Method and Silhouette Score to determine the optimal number of clusters for the Iris dataset.")

    # Elbow Method
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42)
        kmeanModel.fit(data.iloc[:, :-1])
        distortions.append(kmeanModel.inertia_)

    fig, ax = plt.subplots()
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method showing the optimal k')
    st.pyplot(fig)

    # Silhouette Score
    st.write("### Silhouette Scores for different number of clusters")
    silhouette_avg = []
    for k in K[1:]:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data.iloc[:, :-1])
        cluster_labels = kmeans.labels_
        silhouette_avg.append(silhouette_score(data.iloc[:, :-1], cluster_labels))

    fig, ax = plt.subplots()
    plt.plot(K[1:], silhouette_avg, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette scores for different number of clusters')
    st.pyplot(fig)

# Task 3: Use PCA to reduce the Iris dataset to two dimensions and visualize the clusters
def pca_visualization():
    st.title("Cluster Visualization with PCA")
    st.write("### Use Principal Component Analysis (PCA) to reduce the Iris dataset to two dimensions and visualize the clusters obtained from K-Means clustering in the PCA-reduced space.")

    pca = PCA(n_components=2)
    components = pca.fit_transform(data.iloc[:, :-1])

    kmeans = KMeans(n_clusters=3, random_state=42)
    data['pca_cluster'] = kmeans.fit_predict(data.iloc[:, :-1])

    fig, ax = plt.subplots()
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=data['pca_cluster'], palette='viridis', ax=ax)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Visualization of K-Means Clustering')
    st.pyplot(fig)

# Task 4: Implement hierarchical clustering and plot a dendrogram
def hierarchical_clustering():
    st.title("Hierarchical Clustering: Dendrogram")
    st.write("### Implement hierarchical clustering using the Iris dataset and plot a dendrogram to visualize the clustering process.")

    linked = linkage(data.iloc[:, :-1], method='ward')

    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(linked, orientation='top', labels=iris.target_names[data['species']], distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram for Hierarchical Clustering')
    st.pyplot(fig)

# Task 5: Compare the performance of K-Means and Agglomerative Hierarchical Clustering
def compare_clustering_algorithms():
    st.title("Comparing Clustering Algorithms")
    st.write("### Compare the performance of K-Means and Agglomerative Hierarchical Clustering on the Iris dataset.")

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_clusters = kmeans.fit_predict(data.iloc[:, :-1])

    agglomerative = AgglomerativeClustering(n_clusters=3)
    agglomerative_clusters = agglomerative.fit_predict(data.iloc[:, :-1])

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    sns.scatterplot(x=data['sepal length (cm)'], y=data['sepal width (cm)'], hue=kmeans_clusters, palette='viridis', ax=ax[0])
    ax[0].set_title('K-Means Clustering')
    sns.scatterplot(x=data['sepal length (cm)'], y=data['sepal width (cm)'], hue=agglomerative_clusters, palette='viridis', ax=ax[1])
    ax[1].set_title('Agglomerative Clustering')
    st.pyplot(fig)

    st.write("### K-Means Clustering vs Agglomerative Clustering")
    st.write("K-Means Clustering strengths: Efficient, well-defined clusters, sensitive to initial centroids.")
    st.write("Agglomerative Clustering strengths: Can form complex clusters, no need to specify number of clusters, sensitive to noise and outliers.")

# Create the Streamlit sidebar for navigation
st.sidebar.title("Clustering Tasks")
task = st.sidebar.radio("Select Task", ('K-Means Clustering', 'Optimal Clusters', 'PCA Visualization', 'Hierarchical Clustering', 'Compare Clustering Algorithms'))

if task == 'K-Means Clustering':
    kmeans_clustering()
elif task == 'Optimal Clusters':
    optimal_clusters()
elif task == 'PCA Visualization':
    pca_visualization()
elif task == 'Hierarchical Clustering':
    hierarchical_clustering()
elif task == 'Compare Clustering Algorithms':
    compare_clustering_algorithms()
