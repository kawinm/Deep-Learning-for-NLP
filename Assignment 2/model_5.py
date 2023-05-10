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
        chars_to_remove = ['¡', '§', '…','‘', '’', '¿', '«', '»', '¨', '%', '-', '“', '”', '--', '`', '~', '*', '{', '}', '^', '=', '_', '[', ']', '|', '- ']
        
        review = rows['review'].replace('¿', " ", -1)
        for char in chars_to_remove:
            review = review.replace(char, " ", -1)

        review = review.split('<br />')
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
tokenizer = get_tokenizer("basic_english")

for line in pos_set:

    # Tokenizes the input text into words
    tokens = []
    for sent in line:
        token = tokenizer(sent)
        vocab.extend(token)
        tokens.append(token)

    data_set.append((tokens, 1))
    # Adds the extracted words to a list


print("--- Positive Finished ---")

for line in neg_set:

    # Tokenizes the input text into words
    tokens = []
    for sent in line:
        token = tokenizer(sent)
        vocab.extend(token)
        tokens.append(token)

    data_set.append((tokens, 0))
    # Adds the extracted words to a list
    

print("--- Negative Finished ---")

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
    ids_vocab[word] = id
    id += 1

# Tokenization function
def tokenize(corpus, ids_vocab):
    """
        Converts words in the dataset to integer tokens
    """

    tokenized_corpus = []
    for line, sentiment in corpus:
        new_sent = []
        for sent in line:
            new_line = []
            for i, word in enumerate(sent):
                if word in ids_vocab and (i == 0 or word != sent[i-1]):
                    new_line.append(ids_vocab[word])
            if len(new_line) > 0:
                new_sent.append(new_line)

        #new_line = torch.Tensor(new_sent).long()
        tokenized_corpus.append((new_sent, sentiment))

    return tokenized_corpus

token_corpus = tokenize(data_set, ids_vocab)

# Loading the embedding matrix
emb_dim = 300

embeds = torch.zeros(len(ids_vocab) + 1, emb_dim)

for token, idx in ids_vocab.items():
    embeds[idx] = global_vectors[token]

# Train-Valid split of 90-10
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

    max_sen_len = max([len(x[0]) for x in instn])
    sentence = [x[0] for x in instn]

    new_sent = []
    for i in range(len(sentence)):
        for j in range(max_sen_len):
            if j < len(sentence[i]):
                new_sent.append(torch.Tensor(sentence[i][j]))
            else:
                new_sent.append(torch.zeros(5))
    
    # Post padding
    padded_sent = pad_sequence(new_sent, batch_first=True, padding_value=0).long()

    labels = torch.Tensor([x[1] for x in instn])

    return (padded_sent, labels, max_sen_len)


batch_size = 32
train_sampler   = SubsetRandomSampler(train_indices)
train_loader    = DataLoader(token_corpus, batch_size, sampler=train_sampler, collate_fn=collate_fn)

val_sampler     = SubsetRandomSampler(val_indices)
val_loader      = DataLoader(token_corpus, batch_size, sampler=val_sampler, collate_fn=collate_fn)

# ----------- Model -----------
class BILSTM(nn.Module):
    
    def __init__(self, embeds):
        super().__init__()

        self.embeddings = nn.Embedding.from_pretrained(embeds, padding_idx=0)

        self.gru = nn.GRU(input_size = 300, hidden_size = 128, num_layers = 1, batch_first = True)
        self.gru2 = nn.GRU(input_size = 128, hidden_size = 100, num_layers = 1, batch_first = True)

        self.lin1 = nn.Linear(100, 64)
        self.lin2 = nn.Linear(64, 1)
        self.lin3 = nn.Linear(128, 1)
        self.lin6 = nn.Linear(100, 1)

    def forward(self, xb, max_sen_len):

        xe = self.embeddings(xb)
        out, y = self.gru(xe)
        
        x = self.lin3(out).squeeze(dim=-1)
        x = torch.softmax(x, dim=-1).unsqueeze(dim=1)
        x = torch.bmm(x, out).squeeze(dim=1)              # Weighted average

        x = x.view(-1, max_sen_len, 128)
        out, y = self.gru2(x)
        
        x = self.lin6(out).squeeze(dim=-1)
        x = torch.softmax(x, dim=-1).unsqueeze(dim=1)
        x = torch.bmm(x, out).squeeze(dim=1)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x

if torch.cuda.is_available():
    device = torch.device("cuda:2")
else:
    device = torch.device("cpu")

# -------- Text Preprocessing ----------

test_set = []
with open("./E0334 Assignment2 Test Dataset.csv", encoding='utf-8') as csvf:
    data = csv.DictReader(csvf)

    for rows in data:

        # Removing punctuations
        chars_to_remove = ['¡', '§', '…','‘', '’', '¿', '«', '»', '¨', '%', '-', '“', '”', '--', '`', '~', '*', '{', '}', '^', '=', '_', '[', ']', '|', '- ']
        
        review = rows['review'].replace('¿', " ", -1)
        for char in chars_to_remove:
            review = review.replace(char, " ", -1)

        review = review.split('<br />')
        if rows['sentiment'] == 'positive':
            test_set.append((tokens, 1))
        else:
            test_set.append((tokens, 0))

#test_set = sorted(test_set, key=sort_key)

token_corpus_test = tokenize(test_set, ids_vocab)

test_loader      = DataLoader(token_corpus_test, batch_size, collate_fn=collate_fn)


model = BILSTM(embeds)
model.to(device)
opt_c = torch.optim.AdamW(model.parameters(), lr = 0.001) # Same as Adam with weight decay = 0.001
# loss_fn_c = F.cross_entropy #Tried Cross Entropy with log_softmax output function - gave similar results
loss_fn_c = F.binary_cross_entropy

# ----------- Main Training Loop -----------
max_epoch = 10

best_test_acc = 0
for ep in range(max_epoch):

    epoch_loss = 0

    model.train()

    for xb, yb, max_sen_len in tqdm(train_loader):
        xb = xb.to(device)
        yb = yb.to(device)
        if xb.shape[0] < batch_size:
            continue

        y_hat = model(xb, max_sen_len)
        loss = loss_fn_c(y_hat.squeeze(), yb)

        loss.backward()
        opt_c.step()
        opt_c.zero_grad()

        epoch_loss += float(loss)

    print("Epoch: ", ep+1, " Training Loss: ", epoch_loss/len(train_loader))

    #----------- Validation -----------

    val_labels = []
    val_pred = []

    model.eval()

    val_epoch_loss = 0

    with torch.no_grad():
        for xb, yb, max_sen_len in tqdm(val_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            if xb.shape[0] < batch_size:
                continue

            y_hat = model(xb, max_sen_len)
            loss = loss_fn_c(y_hat.squeeze(), yb)

            val_epoch_loss += float(loss)

            val_labels.extend(torch.round(yb).cpu().detach().numpy())
            val_pred.extend(y_hat.round().cpu().detach().numpy())

    print("Validation loss: ", val_epoch_loss/len(val_loader))
    print("Validation accuracy: ", accuracy_score(val_labels, val_pred)*100)

    test_labels = []
    test_pred = []

    model.eval()

    test_epoch_loss = 0

    # ---------- Testing ----------
    with torch.no_grad():
        for xb, yb, max_sen_len in tqdm(test_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            if xb.shape[0] < batch_size:
                continue

            y_hat = model(xb, max_sen_len)
            loss = loss_fn_c(y_hat.squeeze(), yb)

            test_epoch_loss += float(loss)

            test_labels.extend(torch.round(yb).cpu().detach().numpy())
            test_pred.extend(y_hat.round().cpu().detach().numpy())

    print("Test loss: ", test_epoch_loss/len(test_loader))
    print("Test accuracy: ", accuracy_score(test_labels, test_pred)*100)

    if ep > 5 and prev_val_loss - val_epoch_loss > 0.015:
        print("Saving Model")
        torch.save(model.state_dict(), "best_model.pt")
    
    prev_val_loss = val_epoch_loss