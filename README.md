## CV Classification 
This repo compares multiple Machine Learning approaches for matching a candidate's CV to the role they applied for. 
For this, I used the following models:
* XLNet fine-tuned on this dataset (Masked Language Model)
* Bisecting K-Means (Unsupervised ML)
* SVM (Supervised ML)

For preprocessing techniques, I tested both Bert embeddings and Tf-Idf features.
The work focuses on classification performance and on the reliability of training, meaning how sensitive the model’s performance is to the choice of hyperparameters.

### Dataset
The dataset comprises approximately 2400 resumes, which are labelled according to the position the candidate is applying for. There are 24 unique candidate positions.
The dataset contains the candidate’s resume as a string and also the candidate’s resume as the html file.
For this project, only the string version was used. At the time of writing, the dataset is available at: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset.

## Conclusion
* XLNet achieved the greatest performance from the models tested, followed by SVM and Bisecting K-Means
* For this dataset, Bisecting K-Means was particularly challenging to fit because standard Unsupervised Learning evaluation metrics (Silhouette, CH etc.) were not well correlated with
prediction evaluation metrics (F1, accuracy). Thus, the fitted models would be well "clustered", but that would prove to be a poor indicator of prediction performance.

### Technical Report containing in depth documentation of the experiments
https://github.com/Alexandru-AndreiOana/CVClassification/blob/master/TechnicalReport.pdf 
