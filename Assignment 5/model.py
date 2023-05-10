import numpy as np
import random
import torch
import os
import spacy
from torchtext.vocab import GloVe, FastText
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import csv

pos_set = []
neg_set = []
with open("./Train dataset.csv", encoding='utf-8') as csvf:
    data = csv.DictReader(csvf)

    for rows in data:

        # Removing punctuations
        chars_to_remove = ['¡', '§', '…','‘', '’', '¿', '«', '»', '¨', '%', '-', '“', '”', '--', '`', '~', '<', '>', '*', '{', '}', '^', '=', '_', '[', ']', '|', '- ', '/', '<br />']
        
        review = rows['review'].replace('<br />', " ", -1)
        for char in chars_to_remove:
            review = review.replace(char, " ", -1)

        if rows['sentiment'] == 'positive':
            pos_set.append(review)
        else:
            neg_set.append(review)

def set_seed(seed = 42):
    '''
        For Reproducibility: Sets the seed of the entire notebook.
    '''

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    # Sets a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(1)

from torchtext.data import get_tokenizer

# Downloads GloVe and FastText
global_vectors = GloVe(name='840B', dim=300)

# ----------- Text Preprocessing -----------
nlp = spacy.load("en_core_web_md")

data_set = []
vocab = []

chars_to_remove = ['--', '`', '~', '<', '>', '*', '{', '}', '^', '=', '_', '[', ']', '|', '- ', '.', ',', '<br />']

tokenizer = get_tokenizer("basic_english")

for line in pos_set:

    # Tokenizes the input text into words
    tokens = tokenizer(line)

    data_set.append((tokens, 1))
    # Adds the extracted words to a list
    vocab.extend(tokens)


print("--- Positive Finished ---")

for line in neg_set:

    # Tokenizes the input text into words
    tokens = tokenizer(line)

    data_set.append((tokens, 0))
    # Adds the extracted words to a list
    vocab.extend(tokens)

test_set = []
with open("./E0334 Assignment2 Test Dataset.csv", encoding='utf-8') as csvf:
    data = csv.DictReader(csvf)

    for rows in data:

        # Removing punctuations
        chars_to_remove = ['¡', '§', '…','‘', '’', '¿', '«', '»', '¨', '“', '”', '--', '`', '~', '<', '>', '*', '{', '}', '^', '=', '_', '[', ']', '|', '- ', '<br />']
        
        review = rows['review'].replace('<br />', " ", -1)
        for char in chars_to_remove:
            review = review.replace(char, " ", -1)

        tokens = tokenizer(review)

        if rows['sentiment'] == 'positive':
            test_set.append((tokens, 1))
        else:
            test_set.append((tokens, 0))

#len(set(vocab))
def sort_key(s):
    return len(s[0])
    
data_set = sorted(data_set, key=sort_key)
test_set = sorted(test_set, key=sort_key)

# Stores all the unique words in the dataset and their frequencies
vocabulary = {}

# Calculates the frequency of each unique word in the vocabulary
for word in vocab:
    if word in vocabulary:
        vocabulary[word] += 1
    else:
        vocabulary[word] = 1

print("Number of unique words in the vocabulary: ", len(vocabulary))

# Stores the integer token for each unique word in the vocabulary
ids_vocab = {}

id = 0

# Assigns words in the vocabulary to integer tokens
for word, v in vocabulary.items():
    if v > 1:
        ids_vocab[word] = id
        id += 1

# Tokenization function
def tokenize(corpus, ids_vocab):
    """
        Converts words in the dataset to integer tokens
    """

    tokenized_corpus = []
    for line, sentiment in corpus:
        new_line = []
        for i, word in enumerate(line):
            if word in ids_vocab and (i == 0 or word != line[i-1]):
                new_line.append(ids_vocab[word])

        new_line = torch.Tensor(new_line).long()
        tokenized_corpus.append((new_line, sentiment))

    return tokenized_corpus

token_corpus = tokenize(data_set, ids_vocab)
token_corpus_test = tokenize(test_set, ids_vocab)

emb_dim = 300

embeds = torch.zeros(len(ids_vocab) + 1, emb_dim)

for token, idx in ids_vocab.items():
    embeds[idx] = global_vectors[token]

# Train-Valid split of 95-05
def split_indices(n, val_pct):

    # Determine size of Validation set
    n_val = int(val_pct * n)

    # Create random permutation of 0 to n-1
    idxs = np.random.permutation(n)
    return np.sort(idxs[n_val:]), np.sort(idxs[:n_val])

train_pos_indices, val_pos_indices = split_indices(len(pos_set), 0.1)
train_neg_indices, val_neg_indices = split_indices(len(neg_set), 0.1)

train_indices = np.concatenate((train_pos_indices, train_neg_indices+len(pos_set)-1))
val_indices = np.concatenate((val_pos_indices, val_neg_indices+len(pos_set)-1))

from torch.nn.utils.rnn import pad_sequence

# ----------- Batching the data -----------
def collate_fn(instn):

    sentence = [x[0] for x in instn]
    sentence = pad_sequence(sentence, batch_first=True, padding_value=0)

    labels = torch.Tensor([x[1] for x in instn])

    return (sentence, labels)


batch_size = 128

train_sampler   = SubsetRandomSampler(train_indices)
train_loader    = DataLoader(token_corpus, batch_size, sampler=train_sampler, collate_fn=collate_fn)

val_sampler     = SubsetRandomSampler(val_indices)
val_loader      = DataLoader(token_corpus, batch_size, sampler=val_sampler, collate_fn=collate_fn)

test_loader = DataLoader(token_corpus_test, batch_size, collate_fn=collate_fn)
# ----------- Model -----------
class BILSTM(nn.Module):
    
    def __init__(self, embeds):
        super().__init__()

        self.embeddings = nn.Embedding.from_pretrained(embeds, padding_idx=0)

        self.lstm = nn.LSTM(input_size = 300, hidden_size = 200, num_layers = 3, batch_first = True, bidirectional = True, dropout = 0.3)

        self.dropout = nn.Dropout(0.3)
        self.lin1 = nn.Linear(400, 200)
        self.lin2 = nn.Linear(200, 1)

    def forward(self, xb, tsne = False):

        x = self.embeddings(xb)
        x, y = self.lstm(x)
        x = torch.cat((y[0][0, :, :], y[0][1, :, :]), dim = 1)
        x = x.squeeze(dim=0)
        x = self.lin1(x)

        if tsne == True:
            return x 

        x = F.relu(x)
        #x = self.dropout(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


model = BILSTM(embeds)
model.to(device)
opt_c = torch.optim.Adam(model.parameters(), lr = 0.001)
# loss_fn_c = F.cross_entropy - Tried Cross Entropy with log_softmax output function - gave similar results
loss_fn_c = F.binary_cross_entropy

# ----------- Main Training Loop -----------
max_epoch = 25

best_test_acc = 0
for ep in range(max_epoch):

    epoch_loss = 0

    model.train()

    for xb, yb in tqdm(train_loader):
        xb = xb.to(device)
        yb = yb.to(device)

        y_hat = model(xb)
        loss = loss_fn_c(y_hat.squeeze(), yb)

        loss.backward()

        opt_c.step()

        opt_c.zero_grad()

        epoch_loss += float(loss)

    print("Epoch: ", ep+1, " Training Loss: ", epoch_loss)


    #----------- Validation -----------

    val_labels = []
    val_pred = []

    model.eval()

    val_epoch_loss = 0

    with torch.no_grad():
        for xb, yb in tqdm(val_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            y_hat = model(xb)
            loss = loss_fn_c(y_hat.squeeze(), yb)

            val_epoch_loss += float(loss)

            val_labels.extend(torch.round(yb).cpu().detach().numpy())
            val_pred.extend(y_hat.round().cpu().detach().numpy())



    print("Validation loss: ", val_epoch_loss)
    print("Validation accuracy: ", accuracy_score(val_labels, val_pred)*100)


    if ep > 15 and prev_val_loss - val_epoch_loss > 0.05:
        print("Saving Model")
        torch.save(model.state_dict(), "best_model.pt")
    
    prev_val_loss = val_epoch_loss

    
    #----------- Testing -----------

    val_labels = []
    val_pred = []

    model.eval()

    val_epoch_loss = 0

    with torch.no_grad():
        for xb, yb in tqdm(test_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            y_hat = model(xb)
            loss = loss_fn_c(y_hat.squeeze(), yb)

            val_epoch_loss += float(loss)

            val_labels.extend(torch.round(yb).cpu().detach().numpy())
            val_pred.extend(y_hat.round().cpu().detach().numpy())



    print("Validation loss: ", val_epoch_loss)
    print("Validation accuracy: ", accuracy_score(val_labels, val_pred)*100)
    