import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import os

# Define paths
INPUT_FILE = os.path.join("data", "04-processed", "client_features.csv")
OUTPUT_FILE = os.path.join("data", "04-processed", "client_features_hierarchical.csv")

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    
    # Preprocessing
    print("Preprocessing data...")
    features = df[['average_of_days_per_routine', 'routines_count']].copy()
    
    # Encode gender
    features['gender_encoded'] = df['gender'].map({'M': 0, 'F': 1})
    features['gender_encoded'] = features['gender_encoded'].fillna(-1)
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply Agglomerative Clustering
    print("Applying Agglomerative Clustering...")
    agg_clustering = AgglomerativeClustering(n_clusters=4)
    clusters = agg_clustering.fit_predict(scaled_features)
    
    # Add clusters to original dataframe
    df['hierarchical_cluster'] = clusters
    
    # Save results
    print(f"Saving results to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Print summary
    print("Cluster distribution:")
    print(df['hierarchical_cluster'].value_counts().sort_index())

if __name__ == "__main__":
    main()
