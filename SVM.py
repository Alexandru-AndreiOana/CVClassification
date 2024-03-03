import os
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from Utils import Utils
from Utils import K # holds constant values

# Constants to be used for automatic searching
n_trials = 400
n_folds = 10

utils = Utils()
# Load the data
X_data = utils.load_resumes()
Y_data = utils.load_categories()

# Split the data into training and validation sets with stratified sampling
X_train, X_test, Y_train, Y_test = utils.split_data(X_data, Y_data)

def objective(trial):
    C = trial.suggest_float('C', 1e-2, 1e5, log=True)
    Gamma = trial.suggest_float('Gamma', 1e-5, 1e5, log=True)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svc', SVC(C=C, gamma=Gamma, random_state=K.rand_seed))
    ])

    scores = cross_val_score(pipeline,
                             X_train,
                             Y_train,
                             cv=n_folds,
                             scoring='f1_weighted')

    results_df = pd.DataFrame({
        'Trial': [trial.number],
        'C': [C],
        'Gamma': [Gamma],
        'Mean_Cross_Val_Score': [scores.mean()]
    })

    # Write results to a CSV file
    results_df.to_csv('svm_optimization_results.csv', mode='a',
                      header=not os.path.exists('svm_optimization_results.csv'), index=False)
    return scores.mean()


# Run 400 optimization trials and choose the hyperparameter configuration
# that maximizes the mean f1 weighted score after 10-fold cross validation
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials)

best_params = study.best_trial.params
print("Best score: ", study.best_trial.value)
print("Best params: ", best_params)

# Train the SVM classifier with the best parameters from optimization
C = best_params['C']
Gamma = best_params['Gamma']

# Create a pipeline for tfidf preprocessing and SVM configuration
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svc', SVC(C=C, gamma=Gamma, random_state=K.rand_seed))
])

pipeline.fit(X_train, Y_train)

# Infer the labels for the test set
Y_pred = pipeline.predict(X_test)

# Get the evaluation metrics statistics for the prediction
report = classification_report(Y_test, Y_pred, output_dict=True)

# Convert the report to a DataFrame
report_df = pd.DataFrame(report).transpose()

report_df.to_csv('classification_report.csv', index=True)