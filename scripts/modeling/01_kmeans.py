import pandas as pd
import argparse
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Initialize the parser
parser = argparse.ArgumentParser(description="KMeans Clustering")

# Define the argument for clusters
parser.add_argument('--clusters', type=int, default=3, help='Number of clusters')
parser.add_argument('--pca', action='store_true', help='Apply PCA before clustering')
# To indicate the number of clusters, you should run: 
#   py scripts/modeling/01_kmeans.py --clusters 5

# Parse the arguments
args = parser.parse_args()

# Define paths
INPUT_FILE = os.path.join("data", "04-processed", "client_features.csv")
OUTPUT_FILE = os.path.join("data", "04-processed", "client_features.csv")

def main():
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

    # Apply K-Means
    print("Applying K-Means clustering...")

    # After analysis of Silhouette scores, I won't set the best n_cluster value automatically 
    # best_k = 2
    # max_silhouette = -1
    
    # for k in range(2, 11):
    #     km = KMeans(n_clusters=k, random_state=42, n_init=10)
    #     labels = km.fit_predict(scaled_features)
    #     score = silhouette_score(scaled_features, labels)
        
    #     print(f"Silhouette Score for k={k}: {score:.4f}")
    #     if score > max_silhouette:
    #         max_silhouette = score
    #         best_k = k

    # Results:
    #     Silhouette Score for k=2: 0.4172
    #     Silhouette Score for k=3: 0.4414
    #     Silhouette Score for k=4: 0.4978
    #     Silhouette Score for k=5: 0.5046
    #     Silhouette Score for k=6: 0.5273
    #     Silhouette Score for k=7: 0.5362
    #     Silhouette Score for k=8: 0.5523
    #     Silhouette Score for k=9: 0.5493
    #     Silhouette Score for k=10: 0.5453

    # print(f"Optimal number of clusters: {best_k} (Silhouette Score: {max_silhouette:.4f})")
    # kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)

    kmeans = KMeans(n_clusters=args.clusters, random_state=42, n_init=10)
    
    col_name = f'kmeans_cluster_{args.clusters}'
    if args.pca:
        col_name += '_pca'

    df[col_name] = kmeans.fit_predict(scaled_features)
    
    # Save results
    print(f"Saving results to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Print summary
    print("Cluster distribution:")
    print(df[col_name].value_counts().sort_index())

if __name__ == "__main__":
    main()
