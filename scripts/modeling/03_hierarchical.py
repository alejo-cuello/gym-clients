import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Define paths
INPUT_FILE = os.path.join("data", "04-processed", "client_features.csv")
OUTPUT_FILE = os.path.join("data", "04-processed", "client_features.csv")
OUTPUT_FILE_DENDOGRAM = os.path.join("data", "05-clustering", "dendogram.png")

def apply_hierarchical_clustering(features_df, scaled_data, n_clusters):
    """
    Applies the final clustering based on the dendrogram analysis.
    """
    # Initialize the model with the number of clusters identified
    # No spaces around '=' for keyword arguments in function calls
    model = AgglomerativeClustering(
        n_clusters=n_clusters, 
        linkage='ward'
    )
    
    # Fit and predict to get the cluster labels
    # Labels are assigned to each client in the same order as input
    cluster_labels = model.fit_predict(scaled_data)
        
    return cluster_labels

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Clustering")
    parser.add_argument('--pca', action='store_true', help='Apply PCA before clustering')
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    features = df[['average_of_days_per_routine','routines_count','gender_encoded','months_diff']]
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply PCA if requested
    if args.pca:
        print("Applying PCA...")
        pca = PCA(n_components=3)
        scaled_features = pca.fit_transform(scaled_features)
        for i, variance in enumerate(pca.explained_variance_ratio_.cumsum()):
            print(f"Cumulative variance of PCA column {i+1}: {variance:.4f}")

    # After analyzing the dendogram, I decide to split into 3 clusters
    col_name = 'hierarchical_cluster'
    if args.pca:
        col_name += '_pca'
    
    df[col_name] = apply_hierarchical_clustering(features, scaled_features, 3)

    # Save results
    print(f"Saving results to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # # Print summary
    print("Cluster distribution:")
    print(df[col_name].value_counts().sort_index())

    # # Compute linkage matrix
    # print("Computing linkage matrix...")
    # linked = linkage(scaled_features, method='ward')

    # # Plot and save dendrogram
    # plt.figure(figsize=(12, 8))
    # dendrogram(linked)

    # plt.title('Hierarchical Clustering Dendrogram (Gym Clients)')
    # plt.xlabel('Client Index or (Cluster Size)')
    # plt.ylabel('Distance (Ward)')

    # Save the figure to a file
    # Ensure no spaces around '=' for keyword arguments 
    # plt.savefig(OUTPUT_FILE_DENDOGRAM, dpi=300, bbox_inches='tight')
    # print(f"Dendrogram saved successfully to {OUTPUT_FILE_DENDOGRAM}")

if __name__ == "__main__":
    main()
