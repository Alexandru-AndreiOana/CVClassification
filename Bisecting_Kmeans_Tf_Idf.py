import os
import optuna
import pandas as pd
from sklearn.cluster import BisectingKMeans

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import classification_report, calinski_harabasz_score, silhouette_score

from Utils import Utils
from Utils import K # constant values
from collections import Counter

utils = Utils()
n_trials = 100  # number of runs for hyperparameter search

# Load the data
X_data = utils.load_resumes()
Y_data = utils.load_categories()

X_train, X_test, Y_train, Y_test = utils.split_data(X_data, Y_data)

# Extract features with TFIdf
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)


def objective(trial):
    # Configure hyperparameters ranges and data types
    n_clusters = trial.suggest_int('n_clusters', 10, 30)
    bisecting_strategy = trial.suggest_categorical('bisecting_strategy', ['biggest_inertia', 'largest_cluster'])

    # Initialize BisectingKMeans with the trial's hyperparameters
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
    results_df.to_csv('bisecting_kmeans_optim.csv', mode='a',
                      header=not os.path.exists('bisecting_kmeans_optim.csv'), index=False)

    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials)

best_params = study.best_trial.params
print("Best score: ", study.best_trial.value)
print("Best params: ", best_params)

# Test the best performing configuration
best_n_clusters = best_params['n_clusters']
best_bisecting_strategy = best_params['bisecting_strategy']

model = BisectingKMeans(
        n_clusters=best_n_clusters,
        bisecting_strategy=best_bisecting_strategy,
        random_state=K.rand_seed
    )

# Transform the data using the Bisecting K-Means model
X_birch_transformed = model.fit_transform(X_train)
labels = model.labels_

# Count the number of samples in each unique cluster and save for visualization
cluster_counts = Counter(labels)

cluster_data = [{'Cluster': cluster, 'Count': count} for cluster, count in cluster_counts.items()]
cluster_df = pd.DataFrame(cluster_data)

cluster_df.to_csv("cluster_counts_k_means.csv", index=False)

cluster_to_label_mapping = {}
for cluster in np.unique(labels):
    # Find indices of data points in the current cluster
    indices = np.where(labels == cluster)[0]
    # Find the most common label in these indices
    most_common_label = Counter(Y_train[indices]).most_common(1)[0][0]
    cluster_to_label_mapping[cluster] = most_common_label

X_test = vectorizer.transform(X_test)
cluster_predictions = model.predict(X_test)
unique_predicted = len(set(cluster_predictions))
print("Number of clusters predicted: ", unique_predicted)

# Assign predicted labels to the proper cluster
predicted_labels = np.array([cluster_to_label_mapping[cluster] for cluster in cluster_predictions])

# Compute accuracy metrics on the test data and save the classification report
report = classification_report(Y_test, predicted_labels, output_dict=True)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classification_report_kmeans_tf_idf.csv', index=True)