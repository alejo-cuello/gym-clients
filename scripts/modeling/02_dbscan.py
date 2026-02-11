import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
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
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    features = df[['average_of_days_per_routine','routines_count','gender_encoded','months_diff']]
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)    

    # Used to find the best epsilon value
    # find_optimal_epsilon(scaled_features, 2*len(features.columns))

    # Apply DBSCAN
    print("Applying DBSCAN clustering...")
    # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    dbscan = DBSCAN(eps=0.75, min_samples=2*len(features.columns))
    clusters = dbscan.fit_predict(scaled_features)
    
    # Add clusters to original dataframe
    df['dbscan_cluster'] = clusters
    
    # Save results
    print(f"Saving results to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Print summary
    print("Cluster distribution (-1 indicates noise):")
    print(df['dbscan_cluster'].value_counts().sort_index())

if __name__ == "__main__":
    main()
