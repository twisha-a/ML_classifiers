import argparse
import math
def euclidean_dist(x1, y1):
    if len(x1) == len(y1):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, y1)))
    else:
        raise ValueError("Vectors x1 and y1 have different dimensions")

class knn:
    def __init__(self, line_dict, k=3):
        self.k = k
        self.train = line_dict

    def _predict(self, x):
        # Calculate distances between x and all points in the training data
        distances = []
        for key, value in self.train.items():
            dist = euclidean_dist(key, x)
            distances.append((dist, value['label']))
            print(f"Appended ({dist}, {value['label']}) to distances")

        # Sort the distances and select the k nearest neighbors
        k_nearest = sorted(distances, key=lambda x: x[0])[:self.k]
        label_counts = {}
        for _, label in k_nearest:
            label_counts[label] = label_counts.get(label, 0) + 1

        # Find the label with the maximum count
        predicted_label = max(label_counts, key=label_counts.get)
        return predicted_label
        
        # # Print the distances and labels for the k-nearest neighbors
        # for dist, label in k_nearest:
        #     print(f'x={x}, key={key}, label={label}, dist={dist}')
    def predict_test_set(self, test_set):
        predictions = []
        for features, actual_label in test_set:
            predicted_label = self._predict(features)
            predictions.append((actual_label, predicted_label))
        return predictions

def calculate_precision_recall(predictions):
    true_positives = {}
    false_positives = {}
    false_negatives = {}
    for actual, predicted in predictions:
        if actual == predicted:
            true_positives[predicted] = true_positives.get(predicted, 0) + 1
        else:
            false_positives[predicted] = false_positives.get(predicted, 0) + 1
            false_negatives[actual] = false_negatives.get(actual, 0) + 1

    precision = {}
    recall = {}
    for label in set(true_positives.keys()).union(false_positives.keys()):
        tp = true_positives.get(label, 0)
        fp = false_positives.get(label, 0)
        fn = false_negatives.get(label, 0)
        precision[label] = f"{tp}/{tp+fp}" if tp+fp > 0 else "undefined"
        recall[label] = f"{tp}/{tp+fn}" if tp+fn > 0 else "undefined"
    
    return precision, recall
# Load training data
def load_data(filename):
    data = {}
    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split(',')
            features = tuple(int(num) for num in parts[:-1])
            label = parts[-1]
            data[features] = {'label': label}
    return data

def load_test_data(filename):
    test_set = []
    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split(',')
            features = [int(num) for num in parts[:-1]]
            actual_label = parts[-1]
            test_set.append((features, actual_label))
    return test_set

def main(train_file, test_file, k, verbose):
    # Load training data
    training_data = load_data(train_file)

    # Create a KNN classifier instance
    knn_classifier = knn(training_data, k=k)

    # Load test data
    test_set = load_test_data(test_file)

    # Predict for the test set
    predictions = knn_classifier.predict_test_set(test_set)

    # Display results
    for actual, predicted in predictions:
        print(f"want={actual} got={predicted}")

    # Calculate and display precision and recall
    precision, recall = calculate_precision_recall(predictions)
    for label in set([actual for actual, _ in predictions]):
        print(f"Label={label} Precision={precision.get(label, '0/0')} Recall={recall.get(label, '0/0')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run k-Nearest Neighbors Classifier")
    parser.add_argument('-train', type=str, required=True, help='Path to the training file')
    parser.add_argument('-test', type=str, required=True, help='Path to the testing file')
    parser.add_argument('-K', type=int, default=3, help='Number of neighbors (default 3)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    main(args.train, args.test, args.K, args.verbose)
