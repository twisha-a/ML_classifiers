#rough naive bayes
import numpy as np
import argparse
import csv

class NaiveBayes:
    def __init__(self, csv_reader):
        self.data = list(csv_reader)
        self.pure_prob = {}
        self.conditional_prob = {}
        print("Initialized NaiveBayes with data of size:", len(self.data))

    def calculate_pure_probabilities(self):
        print("Calculating pure probabilities...")
        for row in self.data:
            label = row[-1]
            self.pure_prob[label] = self.pure_prob.get(label, 0) + 1

        total_rows = len(self.data)
        for label in self.pure_prob:
            self.pure_prob[label] = self.pure_prob[label] / total_rows

        # print("Pure probabilities:", self.pure_prob)
        for label in self.pure_prob:
            print(f"P(C={label}) = {self.pure_prob[label]} ")


    def calculate_conditional_probabilities(self, correction=0):
        self.c_prob = {}
        c_count = {}

        # First pass to count occurrences
        for row in self.data:
            c = row[-1]
            c_count[c] = c_count.get(c, 0) + 1
            for j, a in enumerate(row[:-1]):
                # Increment count of (feature, value, class)
                key = (j, a, c)
                self.c_prob[key] = self.c_prob.get(key, 0) + 1

        distinct_classifications = len(set(c_count.keys()))

        # Second pass to calculate probabilities with correction
        for key in self.c_prob.keys():
            a_label, a, c = key
            self.c_prob[key] = (self.c_prob[key] + correction) / (c_count[c] + correction * distinct_classifications)
            print(f"P(A{a_label}={a} | C={c}) = {self.c_prob[key]}")

        # print("Conditional probabilities:", self.c_prob)

    
    def predict(self, test_instance):
        probabilities = self.calculate_class_probabilities(test_instance)
        best_label = max(probabilities, key=probabilities.get)
        return best_label

    def get_predictions(self, test_set):
        return [self.predict(instance) for instance in test_set]

    def calculate_class_probabilities(self, test_instance):
        probabilities = {}
        for class_value in self.pure_prob:
            probabilities[class_value] = self.pure_prob[class_value]
            for i, feature_value in enumerate(test_instance):
                if (i, feature_value, class_value) in self.conditional_prob:
                    probabilities[class_value] *= self.conditional_prob[(i, feature_value, class_value)]
                else:
                    # Handle the case where the feature_value-class combination is not in the training set
                    probabilities[class_value] = 0
        return probabilities

    def calculate_precision_recall(self, y_true, y_pred):
        classes = set(y_true)
        precision = {}
        recall = {}

        for cls in classes:
            true_positives = sum((y_pred[i] == cls and y_true[i] == cls) for i in range(len(y_true)))
            false_positives = sum((y_pred[i] == cls and y_true[i] != cls) for i in range(len(y_true)))
            false_negatives = sum((y_pred[i] != cls and y_true[i] == cls) for i in range(len(y_true)))

            precision[cls] = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
            recall[cls] = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

        return precision, recall

    def display_evaluation_metrics(self, y_true, y_pred):
        precision, recall = self.calculate_precision_recall(y_true, y_pred)
        for label in precision:
            print(f"Label={label} Precision={precision[label]} Recall={recall[label]}")

#data
def run_naive_bayes(train_file, test_file, laplacian_correction, verbose):
    # Load training data
    with open(train_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        naive_bayes = NaiveBayes(csv_reader)
    
    if verbose:
        print(f"Training Naive Bayes with correction={laplacian_correction}")

    # Train the classifier
    naive_bayes.calculate_pure_probabilities()
    naive_bayes.calculate_conditional_probabilities(correction=laplacian_correction)

    # Load test data
    test_set = []
    y_true = []
    with open(test_file, mode='r') as csv_file:
        test_csv_reader = csv.reader(csv_file)
        for row in test_csv_reader:
            test_set.append(row[:-1])  # Extract features
            y_true.append(row[-1])     # Extract label

    # Get predictions for the test set
    predictions = naive_bayes.get_predictions(test_set)

    if verbose:
        naive_bayes.display_evaluation_metrics(y_true, predictions)

def main():
    parser = argparse.ArgumentParser(description="Run Naive Bayes Classifier")
    parser.add_argument('-train', type=str, required=True, help='Path to the training file')
    parser.add_argument('-test', type=str, required=True, help='Path to the testing file')
    parser.add_argument('-C', type=int, default=0, help='Laplacian correction value')
    parser.add_argument('-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    run_naive_bayes(args.train, args.test, args.C, args.v)

if __name__ == "__main__":
    main()
