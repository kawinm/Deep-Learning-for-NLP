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
        chars_to_remove = ['¡', '§', '…','‘', '’', '¿', '«', '»', '¨', '%', '-', '“', '”', '--', '`', '~', '<', '>', '*', '{', '}', '^', '=', '_', '[', ']', '|', '- ', '/', '<br />']
        
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
    
#data_set = sorted(data_set, key=sort_key)
#test_set = sorted(test_set, key=sort_key)

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

    labels = torch.Tensor([x[1] for x in instn])

    return (padded_sent, labels)


batch_size = 128

train_sampler   = SubsetRandomSampler(train_indices)
train_loader    = DataLoader(token_corpus, batch_size, sampler=train_sampler, collate_fn=collate_fn)

val_sampler     = SubsetRandomSampler(val_indices)
val_loader      = DataLoader(token_corpus, batch_size, sampler=val_sampler, collate_fn=collate_fn)

test_loader      = DataLoader(token_corpus_test, batch_size, collate_fn=collate_fn)

# ----------- Model -----------
class BILSTM(nn.Module):
    
    def __init__(self, embeds):
        super().__init__()

        self.embeddings = nn.Embedding.from_pretrained(embeds, padding_idx=0)

        self.lstm = nn.GRU(input_size = 300, hidden_size = 128, num_layers = 2, batch_first = True, bidirectional = True, dropout=0.5)

        self.lin1 = nn.Linear(256, 64)
        self.lin2 = nn.Linear(64, 1)

        self.lin3 = nn.Linear(256, 1)

        self.embeddings2 = nn.Embedding.from_pretrained(embeds, padding_idx=0)

        self.lstm2 = nn.LSTM(input_size = 300, hidden_size = 128, num_layers = 2, batch_first = True, bidirectional = True, dropout=0.5)

        self.lin4 = nn.Linear(256, 64)
        self.lin5 = nn.Linear(64, 1)

        self.lin6 = nn.Linear(256, 1)
        #self.lin7 = nn.Linear(512, 2)

    def forward(self, xb, tsne = False):

        xe = self.embeddings(xb)
        out, y = self.lstm(xe)
        
        x = self.lin3(out).squeeze(dim=-1)
        x = torch.softmax(x, dim=-1).unsqueeze(dim=1)
        x = torch.bmm(x, out).squeeze(dim=1)

        #x = torch.cat((x, y[2][ :, :], y[3][ :, :]), dim = 1)
        x = self.lin1(x)

        x = F.relu(x)
        x = self.lin2(x)
        x1 = torch.sigmoid(x)

        xe = self.embeddings2(xb)
        out, yy = self.lstm2(xe)
        
        x = self.lin6(out).squeeze(dim=-1)
        x = torch.softmax(x, dim=-1).unsqueeze(dim=1)
        x = torch.bmm(x, out).squeeze(dim=1)

        #x = torch.cat((x, y[2][ :, :], y[3][ :, :]), dim = 1)
        x = self.lin4(x)

        x = F.relu(x)
        x = self.lin5(x)
        x2 = torch.sigmoid(x)
        return (x1 + x2) /2

if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

model = BILSTM(embeds)
model.to(device)
opt_c = torch.optim.AdamW(model.parameters(), lr = 0.001)
# loss_fn_c = F.cross_entropy - Tried Cross Entropy with log_softmax output function - gave similar results
loss_fn_c = F.binary_cross_entropy

# ----------- Main Training Loop -----------
max_epoch = 50

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

        nn.utils.clip_grad_norm_(model.parameters(), 5)

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

    if ep > 5 and prev_val_loss - val_epoch_loss > 0.05:
        print("Saving Model")
        torch.save(model.state_dict(), "best_model.pt")
    
    prev_val_loss = val_epoch_loss

    #----------- Testing -----------

    test_labels = []
    test_pred = []

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

    
"""
--- Positive Finished ---
Number of unique words in the vocabulary:  94278
100%|████████████████████████████████████████████████████████████| 282/282 [01:16<00:00,  3.68it/s]
Epoch:  1  Training Loss:  0.4277299847057525
100%|██████████████████████████████████████████████████████████████| 32/32 [00:03<00:00,  8.42it/s]
Validation loss:  0.2938038748688996
Validation accuracy:  87.77194298574643
100%|██████████████████████████████████████████████████████████████| 79/79 [00:09<00:00,  8.60it/s]
Test loss:  0.3019353549314451
Test accuracy:  87.61
100%|████████████████████████████████████████████████████████████| 282/282 [01:16<00:00,  3.67it/s]
Epoch:  2  Training Loss:  0.25866527221304303
100%|██████████████████████████████████████████████████████████████| 32/32 [00:03<00:00,  8.22it/s]
Validation loss:  0.23861960414797068
Validation accuracy:  90.04751187796948
100%|██████████████████████████████████████████████████████████████| 79/79 [00:09<00:00,  8.51it/s]
Test loss:  0.2490602903350999
Test accuracy:  89.66
100%|████████████████████████████████████████████████████████████| 282/282 [01:18<00:00,  3.62it/s]
Epoch:  3  Training Loss:  0.22498759611489924
100%|██████████████████████████████████████████████████████████████| 32/32 [00:03<00:00,  8.42it/s]
Validation loss:  0.2291258042678237
Validation accuracy:  90.87271817954489
100%|██████████████████████████████████████████████████████████████| 79/79 [00:09<00:00,  8.52it/s]
Test loss:  0.22684687187400046
Test accuracy:  90.81
100%|████████████████████████████████████████████████████████████| 282/282 [01:20<00:00,  3.51it/s]
Epoch:  4  Training Loss:  0.19797210493091996
100%|██████████████████████████████████████████████████████████████| 32/32 [00:03<00:00,  8.02it/s]
Validation loss:  0.20981087116524577
Validation accuracy:  91.52288072018004
100%|██████████████████████████████████████████████████████████████| 79/79 [00:08<00:00,  9.69it/s]
Test loss:  0.21369755352976955
Test accuracy:  91.60000000000001
100%|████████████████████████████████████████████████████████████| 282/282 [01:21<00:00,  3.46it/s]
Epoch:  5  Training Loss:  0.174177339710031
100%|██████████████████████████████████████████████████████████████| 32/32 [00:04<00:00,  7.69it/s]
Validation loss:  0.22101755952462554
Validation accuracy:  91.29782445611403
100%|██████████████████████████████████████████████████████████████| 79/79 [00:08<00:00,  8.96it/s]
Test loss:  0.23192063860500914
Test accuracy:  91.16
100%|████████████████████████████████████████████████████████████| 282/282 [01:20<00:00,  3.52it/s]
Epoch:  6  Training Loss:  0.1543888029318752
100%|██████████████████████████████████████████████████████████████| 32/32 [00:04<00:00,  7.91it/s]
Validation loss:  0.1994918470736593
Validation accuracy:  92.1980495123781
100%|██████████████████████████████████████████████████████████████| 79/79 [00:09<00:00,  8.55it/s]
Test loss:  0.20781357539228246
Test accuracy:  92.45
100%|████████████████████████████████████████████████████████████| 282/282 [01:18<00:00,  3.59it/s]
Epoch:  7  Training Loss:  0.1363678251967785
100%|██████████████████████████████████████████████████████████████| 32/32 [00:04<00:00,  7.94it/s]
Validation loss:  0.23139404552057385
Validation accuracy:  92.14803700925232
100%|██████████████████████████████████████████████████████████████| 79/79 [00:09<00:00,  8.23it/s]
Test loss:  0.23742014056519617
Test accuracy:  91.83
100%|████████████████████████████████████████████████████████████| 282/282 [01:18<00:00,  3.57it/s]
Epoch:  8  Training Loss:  0.11643736563781475
100%|██████████████████████████████████████████████████████████████| 32/32 [00:04<00:00,  7.76it/s]
Validation loss:  0.24052960006520152
Validation accuracy:  91.99799949987496
100%|██████████████████████████████████████████████████████████████| 79/79 [00:09<00:00,  8.64it/s]
Test loss:  0.24533753691217566
Test accuracy:  91.56
100%|████████████████████████████████████████████████████████████| 282/282 [01:19<00:00,  3.56it/s]
Epoch:  9  Training Loss:  0.09829701835302808
100%|██████████████████████████████████████████████████████████████| 32/32 [00:03<00:00,  8.04it/s]
Validation loss:  0.2260686547961086
Validation accuracy:  92.09802450612654
100%|██████████████████████████████████████████████████████████████| 79/79 [00:09<00:00,  8.23it/s]
Test loss:  0.2447070226639132
Test accuracy:  91.79
100%|████████████████████████████████████████████████████████████| 282/282 [01:19<00:00,  3.56it/s]
Epoch:  10  Training Loss:  0.08727032250017984
100%|██████████████████████████████████████████████████████████████| 32/32 [00:03<00:00,  8.00it/s]
Validation loss:  0.2465045868884772
Validation accuracy:  92.02300575143786
100%|██████████████████████████████████████████████████████████████| 79/79 [00:08<00:00,  8.81it/s]
Test loss:  0.27424220005168193
Test accuracy:  92.05
100%|████████████████████████████████████████████████████████████| 282/282 [01:18<00:00,  3.58it/s]
Epoch:  11  Training Loss:  0.07419056872051236
100%|██████████████████████████████████████████████████████████████| 32/32 [00:04<00:00,  7.82it/s]
Validation loss:  0.2651670118793845
Validation accuracy:  91.84796199049762
100%|██████████████████████████████████████████████████████████████| 79/79 [00:09<00:00,  8.17it/s]
Test loss:  0.27560743387741377
Test accuracy:  91.63
100%|████████████████████████████████████████████████████████████| 282/282 [01:18<00:00,  3.59it/s]
Epoch:  12  Training Loss:  0.06658340678120969
100%|██████████████████████████████████████████████████████████████| 32/32 [00:04<00:00,  7.90it/s]
Validation loss:  0.28987071255687624
Validation accuracy:  92.448112028007
100%|██████████████████████████████████████████████████████████████| 79/79 [00:09<00:00,  8.38it/s]
Test loss:  0.29089372029787375
Test accuracy:  91.67
100%|████████████████████████████████████████████████████████████| 282/282 [01:19<00:00,  3.53it/s]
Epoch:  13  Training Loss:  0.06000985760664475
100%|██████████████████████████████████████████████████████████████| 32/32 [00:04<00:00,  7.87it/s]
Validation loss:  0.31016979389823973
Validation accuracy:  92.22305576394099
100%|██████████████████████████████████████████████████████████████| 79/79 [00:09<00:00,  8.22it/s]
Test loss:  0.34525149890893625
Test accuracy:  91.47999999999999
100%|████████████████████████████████████████████████████████████| 282/282 [01:19<00:00,  3.56it/s]
Epoch:  14  Training Loss:  0.050845625054038376
100%|██████████████████████████████████████████████████████████████| 32/32 [00:03<00:00,  8.02it/s]
Validation loss:  0.30450968141667545
Validation accuracy:  92.52313078269567
100%|██████████████████████████████████████████████████████████████| 79/79 [00:08<00:00,  9.51it/s]
Test loss:  0.32650335623493676
Test accuracy:  91.9
100%|████████████████████████████████████████████████████████████| 282/282 [01:19<00:00,  3.54it/s]
Epoch:  15  Training Loss:  0.05061801715179327
100%|██████████████████████████████████████████████████████████████| 32/32 [00:03<00:00,  9.05it/s]
Validation loss:  0.314442326547578
Validation accuracy:  92.29807451862966
100%|██████████████████████████████████████████████████████████████| 79/79 [00:08<00:00,  9.00it/s]
Test loss:  0.334222047955175
Test accuracy:  92.03"""