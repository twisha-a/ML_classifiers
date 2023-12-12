import argparse
class KMeans:
    def __init__(self, initial_centroids, distance_metric='euclidean'):
        self.centroids = list(initial_centroids)
        self.distance_metric = distance_metric

    def euclidean_distance(self, point1, point2):
        return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5

    def manhattan_distance(self, point1, point2):
        return sum(abs(p1 - p2) for p1, p2 in zip(point1, point2))

    def assign_points_to_clusters(self, data):
        clusters = {i: [] for i in range(len(self.centroids))}  # Initialize clusters for all centroids
        for point, label in data:
            if self.distance_metric == 'euclidean':
                distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            elif self.distance_metric == 'manhattan':
                distances = [self.manhattan_distance(point, centroid) for centroid in self.centroids]
            else:
                raise ValueError("Invalid distance metric. Choose 'euclidean' or 'manhattan'.")

            closest_centroid_index = distances.index(min(distances))
            clusters[closest_centroid_index].append(label)
        return clusters

    def compute_new_centroids(self, clusters, data):
        new_centroids = []
        for i in range(len(self.centroids)):
            cluster_points = [point for point, label in data if label in clusters[i]]
            if cluster_points:  # Check if there are points in the cluster
                centroid = tuple(sum(coords) / len(coords) for coords in zip(*cluster_points))
            else:
                centroid = self.centroids[i]  # Keep the original centroid if no points are assigned
            new_centroids.append(centroid)
        return new_centroids

    def fit(self, data):
        previous_centroids = [(None, None)] * len(self.centroids)

        while previous_centroids != self.centroids:
            clusters = self.assign_points_to_clusters(data)
            previous_centroids = self.centroids[:]
            self.centroids = self.compute_new_centroids(clusters, data)

        return self.centroids, clusters


def read_data_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            point = tuple(map(int, parts[:-1]))  # Convert all but the last part to integers
            label = parts[-1]
            data.append((point, label))
    return data

def parse_centroids(centroid_args):
    centroids = []
    for arg in centroid_args:
        point = tuple(map(int, arg.split(',')))
        centroids.append(point)
    return centroids

def main():
    parser = argparse.ArgumentParser(description='KMeans Clustering')
    parser.add_argument('-train', type=str, required=True, help='Path to the training data file')
    parser.add_argument('-d', type=str, choices=['e2', 'manh'], required=True, help='Distance metric ("e2" for Euclidean, "manh" for Manhattan)')
    parser.add_argument('centroids', nargs='+', help='List of initial centroids (format: x,y)')
    args = parser.parse_args()

    distance_metric = 'euclidean' if args.d == 'e2' else 'manhattan'
    initial_centroids = parse_centroids(args.centroids)
    data_from_file = read_data_from_file(args.train)

    kmeans = KMeans(initial_centroids, distance_metric)
    final_centroids, clusters = kmeans.fit(data_from_file)

    print(f"Final Centroids: {final_centroids}")
    print({f'C{i+1}': labels for i, labels in enumerate(clusters.values())})

if __name__ == "__main__":
    main()