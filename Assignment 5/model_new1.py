from unicodedata import bidirectional
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

def set_seed(seed = 42):
    '''
        For Reproducibility: Sets the seed of the entire notebook.
    '''
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    torch.use_deterministic_algorithms(True)
    
    # Sets a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(1)

positive_set = open("Train.pos", "r", encoding="latin-1").read()
negative_set = open("Train.neg", "r", encoding="latin-1").read()

# Removing punctuations
chars_to_remove = ['--', '`', '~', '<', '>', '*', '{', '}', '^', '=', '_', '[', ']', '|', '- ', '.', ',']

positive_set = positive_set.replace('<br />', " ", -1)
for char in chars_to_remove:
    positive_set = positive_set.replace(char, " ", -1)

negative_set = negative_set.replace('<br />', " ", -1)
for char in chars_to_remove:
    negative_set = negative_set.replace(char, " ", -1)


pos_set = positive_set.split("\n")[:-1]
neg_set = negative_set.split("\n")[:-1]


from torchtext.data import get_tokenizer

# Downloads GloVe and FastText
global_vectors = GloVe(name='840B', dim=300)

# ----------- Text Preprocessing -----------
nlp = spacy.load("en_core_web_md")

data_set = []
vocab = []

chars_to_remove = ['--', '`', '~', '<', '>', '*', '{', '}', '^', '=', '_', '[', ']', '|', '- ', '.', ',']

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
        new_line = []
        for i, word in enumerate(line):
            if word in ids_vocab and (i == 0 or word != line[i-1]):
                new_line.append(ids_vocab[word])

        new_line = torch.Tensor(new_line).long()
        tokenized_corpus.append((new_line, sentiment))

    return tokenized_corpus

token_corpus = tokenize(data_set, ids_vocab)

emb_dim = 600

fasttext = FastText()
embeds = torch.zeros(len(ids_vocab) + 1, emb_dim)
n = 0
for token, idx in ids_vocab.items():
    embeds[idx][:300] = global_vectors[token]
    embeds[idx][300:] = fasttext[token]

    if torch.sum(embeds[idx]) == 0:
        embeds[idx] = torch.randn(600)
        n += 1

print("Number of unknown words: ", n)

# Train-Valid split of 95-05
def split_indices(n, val_pct):

    # Determine size of Validation set
    n_val = int(val_pct * n)

    # Create random permutation of 0 to n-1
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]

train_pos_indices, val_pos_indices = split_indices(len(pos_set), 0.05)
train_neg_indices, val_neg_indices = split_indices(len(neg_set), 0.05)

val_indices = np.concatenate((val_pos_indices, val_neg_indices+len(pos_set)-1)) 
train_indices = np.concatenate((train_pos_indices, train_neg_indices+len(pos_set)-1))

print("Number of positive samples:", len(pos_set))
print("Number of negative samples:", len(neg_set))
print("Number of samples in train set: ", len(train_indices))
print("Number of samples in validation set: ", len(val_indices))

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# ----------- Batching the data -----------
def collate_fn(instn):

    sentence = [x[0] for x in instn]

    # Pre padding
    sen_len = [len(x[0]) for x in instn]
    max_len = max(sen_len)

    padded_sent = torch.zeros(1, max_len)
    sentence_pad = [torch.cat((torch.zeros(max_len-len(x[0])), x[0]), dim=0) for x in instn]
    
    for i in sentence_pad:
        padded_sent = torch.cat((padded_sent, i.unsqueeze(dim=0)), dim=0)
    padded_sent = padded_sent[1:].long()
    
    # Post padding
    #padded_sent = pad_sequence(sentence, batch_first=True, padding_value=0)

    # Pack padded sequence
    #padded_sent = pack_padded_sequence(sentence, batch_first=True)

    labels = torch.Tensor([x[1] for x in instn])

    return (padded_sent, labels)


batch_size = 100

train_sampler   = SubsetRandomSampler(train_indices)
train_loader    = DataLoader(token_corpus, batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=4)

val_sampler     = SubsetRandomSampler(val_indices)
val_loader      = DataLoader(token_corpus, batch_size, sampler=val_sampler, collate_fn=collate_fn)

if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")
# ----------- Model -----------
class BI_GRU(nn.Module):
    
    def __init__(self, embeds):
        super().__init__()

        self.embeddings = nn.Embedding.from_pretrained(embeds, padding_idx=0, freeze=True)

        self.lstm = nn.GRU(input_size = 600, hidden_size = 300, num_layers = 3, batch_first = True, dropout=0.5)

        self.lin1 = nn.Linear(300, 128)
        self.lin2 = nn.Linear(128, 1)

        self.lin3 = nn.Linear(300, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, xb):
    
        xe = self.embeddings(xb)
        out, y = self.lstm(xe)

        x = self.lin1(y[2][ :, :])
        x = F.relu(x)
        x = self.lin2(x)
        x1 = torch.sigmoid(x)

        return x1


model = BI_GRU(embeds)
model.to(device)
opt_c = torch.optim.AdamW(model.parameters(), lr = 0.0005)
loss_fn_c = F.binary_cross_entropy


# ----------- Main Training Loop -----------
max_epoch = 50

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

    print("Epoch: ", ep+1, " Training Loss: ", epoch_loss/len(train_loader))

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

    print("Validation loss: ", val_epoch_loss/len(val_loader))
    print("Validation accuracy: ", accuracy_score(val_labels, val_pred)*100)

    if ep > 25 and prev_val_loss - val_epoch_loss > 0.05:
        print("Saving Model")
        torch.save(model.state_dict(), "best_model.pt")
        break
    
    prev_val_loss = val_epoch_loss


#----------- Testing -----------

test_set = open("TestData", "r", encoding="latin-1").read()

test_set = test_set.replace('<br />', " ", -1)
for char in chars_to_remove:
    test_set = test_set.replace(char, " ", -1)

t_set = test_set.split("\n")[:-1]

test_set = []
for idx, line in enumerate(t_set):

    # Tokenizes the input text into words
    tokens = tokenizer(line)

    if idx < 331:
        test_set.append((tokens, 1))
    else:
        test_set.append((tokens, 0))

print("Number of samples in test set:", len(test_set))

token_corpus_test = tokenize(test_set, ids_vocab)
test_loader = DataLoader(token_corpus_test, batch_size, collate_fn = collate_fn)

test_labels = []
test_pred = []

model = BI_GRU(embeds)
model.to(device)
opt_c = torch.optim.AdamW(model.parameters(), lr = 0.0005)
loss_fn_c = F.binary_cross_entropy
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

test_epoch_loss = 0

with torch.no_grad():
    for xb, yb in tqdm(test_loader):
        xb = xb.to(device)
        yb = yb.to(device)

        y_hat = model(xb)
        loss = loss_fn_c(y_hat.squeeze(), yb)

        test_epoch_loss += float(loss)

        test_labels.extend(torch.round(yb).cpu().detach().numpy())
        test_pred.extend(y_hat.round().cpu().detach().numpy())

print("Test loss: ", test_epoch_loss/len(test_loader))
print("Test accuracy: ", accuracy_score(test_labels, test_pred)*100)


