
This project aims to compare multiple Machine Learning approaches for matching a candidate's CV to the role they applied for. 
For this, I used the following models:
* Bisecting K-Means (Unsupervised ML)
* SVM (Supervised ML)
* XLNet end-to-end, fine-tuned on this dataset (State of the Art Deep Learning)

Additionally, both Bert embeddings and Tf-Idf vectorizer were used as preprocessing techniques.
The work focused on both classification performance as well as ease of use / sensitivity to hyperparameters.

### Dataset
The dataset comprises approximately 2400 resumes, which are labelled according to the position the candidate is applying for. There are 24 unique candidate positions.
The dataset contains the candidate’s resume as a string and also the candidate’s resume as the html file.
For this project, only the string version was used. At the time of writing, the dataset is available at: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset.

## Conclusion
* XLNet achieved the greatest performance from the models tested, followed by SVM and Bisecting K-Means
* For this dataset, Bisecting K-Means was particularly challenging to fit because standard Unsupervised Learning evaluation metrics (Silhouette, CH etc.) were not well correlated with
prediction evaluation metrics (F1, accuracy). Thus, the fitted models would be well "clustered", but that would prove to be a poor indicator of prediction performance.

### Technical Report containing in depth documentation of the experiments
