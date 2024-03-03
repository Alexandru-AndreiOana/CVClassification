# Optimize bisecting k-means(Bert features) and run best configuration
import numpy as np
import pandas as pd
import os
import optuna
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import silhouette_score, classification_report
from collections import Counter

from Utils import Utils, K, get_embeddings  # K for constant values

# Extract the features (X_data)
utils = Utils()
n_trials = 100  # number of runs for hyperparameter search

# Load the data
X_data = utils.load_resumes()
Y_data = utils.load_categories()

X_train, X_test, Y_train, Y_test = utils.split_data(X_data, Y_data)
X_train = get_embeddings(X_train)


def objective(trial):
    # Configure hyperparameter ranges and data types
    n_clusters = trial.suggest_int('n_clusters', 10, 30)
    bisecting_strategy = trial.suggest_categorical('bisecting_strategy', ['biggest_inertia', 'largest_cluster'])

    # Initialize BisectingKMeans with the suggested hyperparameters
    model = BisectingKMeans(
        n_clusters=n_clusters,
        bisecting_strategy=bisecting_strategy,
        random_state=K.rand_seed
    )

    # Fit the model
    model.fit(X_train)
    labels = model.labels_

    score = silhouette_score(X_train, labels)

    results_df = pd.DataFrame({
        'Trial': [trial.number],
        'n_clusters': [n_clusters],
        'bisecting_strategy': [bisecting_strategy],
        'Silhouette score': [score]
    })

    # Write results to a CSV file
    results_df.to_csv('bisecting_kmeans_bert_optimization.csv', mode='a',
                      header=not os.path.exists('bisecting_kmeans_bert_optimization.csv'), index=False)

    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials)

best_params = study.best_trial.params
print("Best score: ", study.best_trial.value)
print("Best params: ", best_params)

# Train with the best performing hyperparameter configuration
print("Training phase")

best_n_clusters = best_params['n_clusters']
best_bisecting_strategy = best_params['bisecting_strategy']

model = BisectingKMeans(
    n_clusters=best_n_clusters,
    bisecting_strategy=best_bisecting_strategy,
    random_state=K.rand_seed)

model.fit(X_train)
labels = model.labels_

cluster_counts = Counter(labels)
print("Number of clusters: ", len(cluster_counts))

for cluster, count in cluster_counts.items():
    if count > 1:
        print(f"Cluster {cluster}: {count} datapoints")

cluster_to_label = {}
for cluster in np.unique(labels):
    # Find indices of data points in the current cluster
    indices = np.where(labels == cluster)[0]
    # Find the most common label in these indices
    most_common_label = Counter(Y_train[indices]).most_common(1)[0][0]
    cluster_to_label[cluster] = most_common_label

# --------------
print("Testing phase")

X_test = get_embeddings(X_test)
cluster_predictions = model.predict(X_test)
unique_predicted = len(set(cluster_predictions))
print("Number of clusters predicted: ", unique_predicted)

# Assign predicted labels to the proper cluster
predicted_labels = np.array([cluster_to_label[cluster] for cluster in cluster_predictions])

# Compute accuracy metrics on the test data and save the classification report
report = classification_report(Y_test, predicted_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('bisectKMeans_bert_test_report.csv', index=True)
