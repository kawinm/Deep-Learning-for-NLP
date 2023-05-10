# %%
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
        chars_to_remove = [ '+', '#', '¡', '§', '…','‘', '’', '¿', '«', '»', '¨', '%', '-', '“', '”', '--', '`', '~', '<', '>', '*', '{', '}', '^', '=', '_', '[', ']', '|', '- ', '/']
        
        review = rows['review'].replace('<br />', " ", -1)
        review = review.replace('´', "'", -1)
        for char in chars_to_remove:
            review = review.replace(char, " ", -1)

        if rows['sentiment'] == 'positive':
            pos_set.append(review)
        else:
            neg_set.append(review)

# %%
import math

from transformers import BertTokenizer, BertModel, DistilBertTokenizer, TFDistilBertModel, DistilBertModel


# %%
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

# %%
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased' , lower = True)
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

print(bert_model(**inputs))

# %%
from torchtext.data import get_tokenizer

# Downloads GloVe and FastText
global_vectors = GloVe(name='840B', dim=300)

# ----------- Text Preprocessing -----------
nlp = spacy.load("en_core_web_md")

positive_set = []
vocab = []
tokenizerz = get_tokenizer("basic_english")

for line in pos_set:

    # Tokenizes the input text into words
    tokens = tokenizerz(line)

    positive_set.append((tokens, 1))
    # Adds the extracted words to a list
    vocab.extend(tokens)


print("--- Positive Finished ---")
negative_set = []
for line in neg_set:

    # Tokenizes the input text into words
    tokens = tokenizerz(line)

    negative_set.append((tokens, 0))
    # Adds the extracted words to a list
    vocab.extend(tokens)

vocabulary = {}

for i in vocab:
    if i not in vocabulary:
        vocabulary[i] = 1
    else:
        vocabulary[i] += 1

maxi = max(vocabulary.values())
print("Len of vocab: ", len(vocabulary.values()), maxi)

vocabu = {}

for word, value in vocabulary.items():
    if 3 <= value <= 500000:
        vocabu[word] = 1 

maxi = max(vocabu.values())
print("Len of vocab: ", len(vocabu.values()), maxi)

p_set = []
for line, sent in positive_set:
    token = []
    for word in line:
        if word in vocabu:
            token.append(word)
    p_set.append((token,1))

positive_set = p_set

n_set = []
for line, sent in negative_set:
    token = []
    for word in line:
        if word in vocabu:
            token.append(word)
    n_set.append((token, 0))

negative_set = n_set


print("--- Negative Finished ---")

# %%
# Train-Valid split of 90-10
def split_indices(n, val_pct):

    # Determine size of Validation set
    n_val = int(val_pct * n)

    # Create random permutation of 0 to n-1
    idxs = np.random.permutation(n)
    #return np.sort(idxs[n_val:]), np.sort(idxs[:n_val])
    return idxs[n_val:], idxs[:n_val]

train_pos_indices, val_pos_indices = split_indices(len(positive_set), 0.1)
train_neg_indices, val_neg_indices = split_indices(len(negative_set), 0.1)


train_indices = np.concatenate((train_pos_indices, train_neg_indices+len(positive_set)-1))
val_indices = np.concatenate((val_pos_indices, val_neg_indices+len(positive_set)-1))

# %%
from torch.nn.utils.rnn import pad_sequence

# ----------- Batching the data -----------
def collate_fn(instn):

    sentence = [" ".join(x[0]) for x in instn]

    padded_sent = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

    labels = torch.Tensor([x[1] for x in instn])

    return (padded_sent, labels)


batch_size = 32

train_sampler   = SubsetRandomSampler(train_indices)
train_loader    = DataLoader(positive_set+negative_set, batch_size, sampler=train_sampler, collate_fn=collate_fn)

val_sampler     = SubsetRandomSampler(val_indices)
val_loader      = DataLoader(positive_set+negative_set, batch_size, sampler=val_sampler, collate_fn=collate_fn)


# %%
# -------- Text Preprocessing ----------

test_set = []
with open("./E0334 Assignment2 Test Dataset.csv", encoding='utf-8') as csvf:
    data = csv.DictReader(csvf)

    for idx, rows in enumerate(data):

        # Removing punctuations
        chars_to_remove = [ '+', '#', '¡', '§', '…','‘', '’', '¿', '«', '»', '¨', '%', '-', '“', '”', '--', '`', '~', '<', '>', '*', '{', '}', '^', '=', '_', '[', ']', '|', '- ', '/']
        
        review = rows['review'].replace('<br />', " ", -1)
        review = review.replace('´', "'", -1)
        for char in chars_to_remove:
            review = review.replace(char, " ", -1)

        line = review.split()
        leni = len(line)

        if rows['sentiment'] == 'positive':
            if leni <= 512:
                positive_set.append((line, 1))
            else:
                add = (leni - 512)//2
                test_set.append((line[add:], 1))
        else:
            if leni <= 512:
                positive_set.append((line, 0))
            else:
                add = (leni - 512)//2
                test_set.append((line[add:], 0))

#test_set = sorted(test_set, key=sort_key)

test_loader   = DataLoader(test_set, batch_size, collate_fn=collate_fn)


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = bert_model

        self.lin1 = nn.Linear(768, 256)
        self.lin2 = nn.Linear(256, 32)
        self.lin3 = nn.Linear(32, 1)

        self.attn = nn.Linear(768, 1)

    def forward(self, x):
        out = self.bert(**x).last_hidden_state
        x = self.attn(out).squeeze(dim=-1)
        x = torch.softmax(x, dim=-1).unsqueeze(dim=1)
        x = torch.bmm(x, out).squeeze(dim=1)   

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = torch.sigmoid(x).squeeze(dim=-1)

        return x

# %%
if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

model = Encoder()
model.to(device)
opt_c = torch.optim.Adam(model.parameters(), lr = 0.01) # Same as Adam with weight decay = 0.001
# loss_fn_c = F.cross_entropy #Tried Cross Entropy with log_softmax output function - gave similar results
loss_fn_c = F.binary_cross_entropy

# %%
# ----------- Main Training Loop -----------
max_epoch = 10

best_test_acc = 0
for ep in range(max_epoch):

    epoch_loss = 0

    model.train()

    for xb, yb in tqdm(train_loader):
        xb = xb.to(device)
        yb = yb.to(device)

        y_hat = model(xb)
        loss = loss_fn_c(y_hat, yb)

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
            loss = loss_fn_c(y_hat, yb)

            val_epoch_loss += float(loss)

            val_labels.extend(torch.round(yb).cpu().detach().numpy())
            val_pred.extend(y_hat.round().cpu().detach().numpy())

    print("Validation loss: ", val_epoch_loss/len(val_loader))
    print("Validation accuracy: ", accuracy_score(val_labels, val_pred)*100)

    if ep > 5 and prev_val_loss - val_epoch_loss > 0.015:
        print("Saving Model")
        torch.save(model.state_dict(), "best_model.pt")
    
    prev_val_loss = val_epoch_loss

    test_labels = []
    test_pred = []

    model.eval()

    test_epoch_loss = 0

    n = 0
    # ---------- Testing ----------
    with torch.no_grad():
        for xb, yb in tqdm(test_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            y_hat = model(xb)
            loss = loss_fn_c(y_hat, yb)

            test_epoch_loss += float(loss)

            test_labels.extend(torch.round(yb).cpu().detach().numpy())
            test_pred.extend(y_hat.round().cpu().detach().numpy())


    print("Test loss: ", test_epoch_loss/len(test_loader))
    print("Test accuracy: ", accuracy_score(test_labels, test_pred)*100)

# %%
# Tokenization function

# %%
model = BILSTM(embeds)
model.load_state_dict(torch.load("best_model.pt"))
model.to(device)

test_labels = []
test_pred = []

model.eval()

test_epoch_loss = 0

n = 0
# ---------- Testing ----------
with torch.no_grad():
    for xb, yb, idx in tqdm(test_loader):
        xb = xb.to(device)
        yb = yb.to(device)

        y_hat = model(xb)
        loss = loss_fn_c(y_hat.squeeze(), yb)

        test_epoch_loss += float(loss)

        test_labels.extend(torch.round(yb).cpu().detach().numpy())
        test_pred.extend(y_hat.round().cpu().detach().numpy())

        for i, v in enumerate(torch.round(yb).cpu().detach().numpy()):
            if v != y_hat.round().cpu().detach().numpy()[i]:
                print(test_set[int(idx[i])])
                n += 1

print("Test loss: ", test_epoch_loss/len(test_loader))
print("Test accuracy: ", accuracy_score(test_labels, test_pred)*100)

# %%
# Seed doesn't work in Jupyter notebook, to replicate my results, kindly, run it as .py file


