import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse
import os

# Define paths
INPUT_FILE = os.path.join("data", "04-processed", "client_features.csv")
OUTPUT_FILE = os.path.join("data", "04-processed", "client_features.csv")
OUTPUT_FILE_K_DISTANCE = os.path.join("data", "05-clustering", "k_distance.png")

def find_optimal_epsilon(scaled_data, k=4):
    """
    Plots the k-distance graph to help identify the best epsilon value.
    """
    # Calculate the distance to the k-th nearest neighbor for each point
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(scaled_data)
    distances, indices = neighbors_fit.kneighbors(scaled_data)

    # Sort the distances in ascending order
    # Following PEP 8: single space around binary operators
    distances = np.sort(distances[:, k-1], axis=0)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title(f'K-Distance Plot (k={k})')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel(f'{k}-th Nearest Neighbor Distance (Epsilon)')
    plt.grid(True)
    plt.savefig(OUTPUT_FILE_K_DISTANCE, dpi=300, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser(description="DBSCAN Clustering")
    parser.add_argument('--pca', action='store_true', help='Apply PCA before clustering')
    parser.add_argument('--epsilon', action='store_true', help='Get epsilon values instead of clustering')
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    features = df[['average_of_days_per_routine','routines_count','gender_encoded','tenure_months','recency_months']]
    
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

    if args.epsilon:
        find_optimal_epsilon(scaled_features, 2*len(features.columns))
        # Without PCA: 1.2
        # With PCA: 0.6
    else:
        # Apply DBSCAN
        print("Applying DBSCAN clustering...")
        # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        # min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        dbscan = DBSCAN(eps=0.6, min_samples=2*len(features.columns))
        clusters = dbscan.fit_predict(scaled_features)
        
        # Add clusters to original dataframe
        col_name = 'dbscan_cluster'
        if args.pca:
            col_name += '_pca'
        df[col_name] = clusters
        
        # Save results
        print(f"Saving results to {OUTPUT_FILE}...")
        df.to_csv(OUTPUT_FILE, index=False)
        
        # Print summary
        print("Cluster distribution (-1 indicates noise):")
        print(df[col_name].value_counts().sort_index())

if __name__ == "__main__":
    main()
