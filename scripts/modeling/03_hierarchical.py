import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Define paths
INPUT_FILE = os.path.join("data", "04-processed", "client_features.csv")
# OUTPUT_FILE = os.path.join("data", "05-clustering", "dendogram.png")
OUTPUT_FILE = os.path.join("data", "05-clustering", "client_features_hierarchical.csv")

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
    
    # Add the labels back to the original dataframe for business analysis
    features_df['cluster_id'] = cluster_labels
    
    return features_df

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    
    print("Preprocessing data...")
    features = df[['average_of_days_per_routine', 'routines_count']].copy()
    
    # Encode gender: M=0, F=1 (if there are other values, they will need handling, but assuming binary for now based on glimpse)
    features['gender_encoded'] = df['gender'].map({'M': 0, 'F': 1})
    # Fill NaN if any (just in case)
    features['gender_encoded'] = features['gender_encoded'].fillna(-1) 
    
    # Months diff: get the difference between the first and last month.
    df['last_month'] = pd.to_datetime(df['last_month'])
    df['cohort_month'] = pd.to_datetime(df['cohort_month'])
    features['months_diff'] = (df['last_month'].dt.year - df['cohort_month'].dt.year) * 12 + (df['last_month'].dt.month - df['cohort_month'].dt.month)
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # After analyzing the dendogram, I decide to split into 3 clusters
    df = apply_hierarchical_clustering(features, scaled_features, 3)

    # Save results
    print(f"Saving results to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Print summary
    print("Cluster distribution:")
    print(df['cluster_id'].value_counts().sort_index())

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
    # plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    # print(f"Dendrogram saved successfully to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
