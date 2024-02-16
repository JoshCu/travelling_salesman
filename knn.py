import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Example adjacency matrix (replace with your actual matrix)
def load_file(filename):
    with open(filename, "r") as file:
        file_content = [[int(weight) for weight in line.strip().split()] for line in file]

    #file_content = file_content[:9]

    matrix = np.zeros((len(file_content), len(file_content)))
    for i, row in enumerate(file_content):
        for j, weight in enumerate(row[:-1]):
            matrix[i, j] = weight
    # mirror over the diagonal
    matrix = np.maximum(matrix, matrix.T)
    return matrix

def evaluate_solution(matrix, path):
    start_nodes = path[:-1]
    end_nodes = path[1:]
    return matrix[start_nodes, end_nodes].sum()

def inertia(path = "Size1000.graph"):
    adj_matrix = load_file(path)
    # Use Multidimensional Scaling (MDS) to derive coordinates for nodes
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    positions = mds.fit_transform(adj_matrix)

    inertia = []
    for k in range(1, 15):  # Adjust the range of k as needed
        kmeans = KMeans(n_clusters=k, random_state=42).fit(positions)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 15), inertia, 'bo-')  # Adjust range of k as used above
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

def kmeans(path = "Size1000.graph"):
    adj_matrix = load_file(path)
    # Use Multidimensional Scaling (MDS) to derive coordinates for nodes
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    positions = mds.fit_transform(adj_matrix)
    # Apply k-means clustering
    k = 100  # Number of clusters; adjust based on your problem
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(positions)

    # Plotting the nodes and their clusters for visualization
    plt.scatter(positions[:, 0], positions[:, 1], c=clusters, cmap='viridis', marker='o')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
    plt.title('k-Means Clustering of Nodes')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    #plt.show()
    return clusters

def silouhettes(path = "Size1000.graph"):
    adj_matrix = load_file(path)
    # Use Multidimensional Scaling (MDS) to derive coordinates for nodes
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    positions = mds.fit_transform(adj_matrix)

    silhouette_scores = []
    for k in range(2, 15):  # Silhouette score can't be computed with only one cluster
        kmeans = KMeans(n_clusters=k, random_state=42).fit(positions)
        score = silhouette_score(positions, kmeans.labels_)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 4))
    plt.plot(range(2, 15), silhouette_scores, 'bo-')  # Adjust the range of k as needed
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal k')
    plt.show()

def calculate_width(clusters, adj_matrix):
    # calculate the furthest distance between clusters
    num_clusters = np.max(clusters) + 1
    furthest_distance = 0

    for num in range(num_clusters):
        cluster = np.where(clusters==num)[0]
        starts = [[i] * len(cluster) for i in cluster]
        ends = [cluster] * len(cluster)
        starts = np.array(starts).flatten()
        ends = np.array(ends).flatten()
        furthest_distance += adj_matrix[starts, ends].max()
    return furthest_distance/num_clusters

def weighted_nearest_neighbor(weighted_matrix, real_matrix):
    n = weighted_matrix.shape[0]
    path = [0]
    visited = set([0])
    for _ in range(n - 1):
        current = path[-1]
        best_next = None
        best_score = float("inf")
        for i in range(n):
            if i not in visited:
                if weighted_matrix[current, i] < best_score:
                    best_score = weighted_matrix[current, i]
                    best_next = i
        path.append(best_next)
        visited.add(best_next)
    path.append(0)
    score = evaluate_solution(real_matrix, path)
    return np.array(path), score

def generate_clustered_distances(clusters, adj_matrix, magic_constant):
    weighted_matrix = adj_matrix.copy()

    num_clusters = np.max(clusters) + 1
    # add the magic constant to every edge that is not within the same cluster
    for i in range(num_clusters):
        for j in range(num_clusters):
            if i != j:
                cluster_i = np.where(clusters == i)[0]
                cluster_j = np.where(clusters == j)[0]
                # Create a meshgrid-like index for each pair of clusters
                ixgrid = np.ix_(cluster_i, cluster_j)
                # Add the magic constant to the weights between clusters i and j
                weighted_matrix[ixgrid] += magic_constant


    return weighted_matrix
    

if __name__ == "__main__":
    #inertia()
    #kmeans()
    #silouhettes()
    adj_matrix = load_file("Size1000.graph")
    print(adj_matrix.max())
    clusters = kmeans("Size1000.graph")
    magic_constant = calculate_width(clusters, adj_matrix)
    print(magic_constant)
    overall_best_path = None
    overall_best_score = np.iinfo(np.int32).max

    for i in range(1,30):
        mc = magic_constant / i
        weighted_adj_matrix = generate_clustered_distances(clusters, adj_matrix, mc)
        best_path, score = weighted_nearest_neighbor(weighted_adj_matrix, adj_matrix)                                    
        #print(clusters)
        print(f"mc {mc} divided by {i}")
        print(f"best path: {best_path}")
        print(f"score: {score}")
        if score < overall_best_score:
            overall_best_path = best_path
            overall_best_score = score


    print(f"Overall best path: {overall_best_path}")
    print(f"Overall best score: {overall_best_score}")