import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Define paths
INPUT_FILE = os.path.join("data", "04-processed", "client_features.csv")
OUTPUT_FILE = os.path.join("data", "04-processed", "client_features_kmeans.csv")

def main():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    
    # Preprocessing
    print("Preprocessing data...")
    features = df[['average_of_days_per_routine', 'routines_count']].copy()
    
    # Encode gender: M=0, F=1 (if there are other values, they will need handling, but assuming binary for now based on glimpse)
    features['gender_encoded'] = df['gender'].map({'M': 0, 'F': 1})
    # Fill NaN if any (just in case)
    features['gender_encoded'] = features['gender_encoded'].fillna(-1) 
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply K-Means
    print("Applying K-Means clustering...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Add clusters to original dataframe
    df['kmeans_cluster'] = clusters
    
    # Save results
    print(f"Saving results to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Print summary
    print("Cluster distribution:")
    print(df['kmeans_cluster'].value_counts().sort_index())

if __name__ == "__main__":
    main()
