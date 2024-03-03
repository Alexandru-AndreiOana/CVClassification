# End-to-End approach

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder


# Useful mainly for reducing computational drain / dimensionality
# before tokenizing with XLNet
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)


# Read CVs and candidate positions
df = pd.read_csv('/kaggle/input/resume-dataset/Resume/Resume.csv')

rand_seed = 1

X_data = df['Resume_str'].values
Y_data = df['Category'].values

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Encode labels
encoder = LabelEncoder()
Y_data = encoder.fit_transform(Y_data)

# 64 - 16 - 20 (train - validation - test)
# Split data into training and testing subsets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data,
                                                    test_size=0.2,
                                                    random_state=rand_seed,
                                                    stratify=Y_data)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                  test_size=0.2,
                                                  random_state=rand_seed,
                                                  stratify=Y_train)


print("Training data shape ", X_train.shape)
print("Validation data shape", X_val.shape)
print("Testing data shape ", X_test.shape)

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import XLNetTokenizer, XLNetForSequenceClassification


def tokenize_data(data):
    return tokenizer(data, truncation=True, padding=True, max_length=512, return_tensors='pt')


tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=len(np.unique(Y_data)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Preprocess and tokenize data
X_train = [remove_stopwords(text) for text in X_train]
X_val = [remove_stopwords(text) for text in X_val]
X_test = [remove_stopwords(text) for text in X_test]

print("Stopwords removed")

X_train = tokenize_data(X_train)
X_test = tokenize_data(X_test)
X_val = tokenize_data(X_val)

Y_train = torch.tensor(Y_train)
Y_test = torch.tensor(Y_test)
Y_val = torch.tensor(Y_val)

print("Tokenization done")

# Split data in batches
batch_size = 12  # Choose max value that does not cause out of memory issues

train_data = TensorDataset(X_train['input_ids'], X_train['attention_mask'], Y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

val_data = TensorDataset(X_val['input_ids'], X_val['attention_mask'], Y_val)
val_loader = DataLoader(val_data, batch_size=batch_size)

test_data = TensorDataset(X_test['input_ids'], X_test['attention_mask'], Y_test)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Train the model
model.train()
model.to(device)  # Move the model to the correct device
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

epoch_losses = []
val_losses = []

epochs = 10

for epoch in range(epochs):
    total_loss = 0
    model.train()

    for batch in train_loader:
        b_input_ids, b_attention_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_attention_mask = b_attention_mask.to(device)
        b_labels = b_labels.to(device)

        optimizer.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    epoch_losses.append(avg_loss)

    # Validation step
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            b_input_ids, b_attention_mask, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_attention_mask = b_attention_mask.to(device)
            b_labels = b_labels.to(device)

            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss}, Validation Loss: {avg_val_loss}")

    if len(val_losses) > 1 and avg_val_loss > val_losses[-1]:
        print("Training early stopped.")
        val_losses.append(avg_val_loss)
        break

    val_losses.append(avg_val_loss)

plt.figure()
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.show()

print("Training completed")

# Evaluate the model
model.eval()
total_predictions = []
total_true_labels = []

with torch.no_grad():
    for batch in test_loader:
        b_input_ids, b_attention_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_attention_mask = b_attention_mask.to(device)
        b_labels = b_labels.to(device)

        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        total_predictions.extend(predictions.cpu().numpy())
        total_true_labels.extend(b_labels.cpu().numpy())

print(classification_report(total_true_labels, total_predictions))