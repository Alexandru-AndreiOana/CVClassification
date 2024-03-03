import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Download nltk stopwords packages (only need to run once)
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# constant values
class K:
    test_size = 0.2
    rand_seed = 1


class Utils:
    def __init__(self, csv_file_name='Resume.csv'):
        self.df = pd.read_csv(csv_file_name)

    def load_resumes(self):
        # get the training data
        resumes = self.df['Resume_str'].values
        return resumes

    def load_categories(self):
        # get the training labels
        categories = self.df['Category'].values
        return categories

    @staticmethod
    def split_data(X_data, Y_data, test_size=K.test_size, random_state=K.rand_seed):
        x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data,
                                                            test_size=test_size,
                                                            random_state=random_state,
                                                            stratify=Y_data)
        return x_train, x_test, y_train, y_test


def remove_stopwords(corpus):
    word_tokens = word_tokenize(corpus)
    tokenized_corpus = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(tokenized_corpus)


# Process a corpus of text into embeddings from Bert pretrained model
def get_embeddings(corpus):
    corpus = [remove_stopwords(document) for document in corpus]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    from transformers import BertTokenizer, BertModel
    # Load Bert tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load Bert model
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    # Configure the tokenizer
    tokenized_inputs = tokenizer(
        list(corpus),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"  # Return PyTorch tensors
    )

    print("Tokenized corpus done -> Getting embeddings")

    # Use TensorDataset and DataLoader to split data in batches
    dataset = TensorDataset(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'])
    loader = DataLoader(dataset, batch_size=256)

    corpus_embeddings = []

    # Process data in batches to avoid memory overload
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing Batches"):
            input_ids, attention_mask = batch
            # Move tensors to correct device
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

            # Average the token embeddings to reduce the embedding shape
            mean_embeddings = embeddings.mean(dim=1)
            corpus_embeddings.append(mean_embeddings.cpu())

    # Concatenate embeddings
    corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
    embeddings = corpus_embeddings.numpy()

    print(embeddings.shape)
    print("Embeddings Computed -> Done")

    return embeddings
