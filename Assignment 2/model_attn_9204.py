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

    def forward(self, xb, tsne = False):

        xe = self.embeddings(xb)
        out, y = self.lstm(xe)
        
        x = self.lin3(out).squeeze(dim=-1)
        x = torch.softmax(x, dim=-1).unsqueeze(dim=1)
        x = torch.bmm(x, out).squeeze(dim=1)

        #x = torch.cat((x, y[2][ :, :], y[3][ :, :]), dim = 1)
        x = self.lin1(x)

        if tsne == True:
            return x 

        x = F.relu(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x

if torch.cuda.is_available():
    device = torch.device("cuda:2")
else:
    device = torch.device("cpu")


model = BILSTM(embeds)
model.to(device)
opt_c = torch.optim.AdamW(model.parameters(), lr = 0.001)
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

    if ep > 15 and prev_val_loss - val_epoch_loss > 0.05:
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

    

# Observations

# Dropout: Adding dropout on LSTM works, better generalization on validation set, dropout 0.25 num layer 2 - acc 89.775 - val loss 17
# Adam is perfroming better on Val loss than Adagrad this time - 92.61 - Val loss 12.22 lr 0.001

#Validation loss:  0.24658807273954153
#Validation accuracy:  91.89797449362341
#100%|███████████████████████████████████████████████████████| 79/79 [00:01<00:00, 53.52it/s]
#Test loss:  0.2457141642329059
#Test accuracy:  91.61
"""
Epoch:  1  Training Loss:  0.3842095099653758
100%|███████████████████████████████████████████████████████| 32/32 [00:02<00:00, 14.75it/s]
Validation loss:  0.2715818313881755
Validation accuracy:  88.6721680420105
100%|███████████████████████████████████████████████████████| 79/79 [00:01<00:00, 48.86it/s]
Test loss:  0.30232435993001433
Test accuracy:  87.39
100%|█████████████████████████████████████████████████████| 282/282 [00:52<00:00,  5.36it/s]
Epoch:  2  Training Loss:  0.24363532583130168
100%|███████████████████████████████████████████████████████| 32/32 [00:02<00:00, 14.51it/s]
Validation loss:  0.22398806922137737
Validation accuracy:  90.72268067016755
100%|███████████████████████████████████████████████████████| 79/79 [00:01<00:00, 48.93it/s]
Test loss:  0.23558939578412455
Test accuracy:  90.34
100%|█████████████████████████████████████████████████████| 282/282 [01:02<00:00,  4.50it/s]
Epoch:  3  Training Loss:  0.21262501235655012
100%|███████████████████████████████████████████████████████| 32/32 [00:03<00:00,  8.90it/s]
Validation loss:  0.2100580483675003
Validation accuracy:  91.32283070767691
100%|███████████████████████████████████████████████████████| 79/79 [00:01<00:00, 43.24it/s]
Test loss:  0.22238406935070135
Test accuracy:  91.41
100%|█████████████████████████████████████████████████████| 282/282 [01:05<00:00,  4.33it/s]
Epoch:  4  Training Loss:  0.18604649772458043
100%|███████████████████████████████████████████████████████| 32/32 [00:02<00:00, 10.93it/s]
Validation loss:  0.2077047594357282
Validation accuracy:  91.89797449362341
100%|███████████████████████████████████████████████████████| 79/79 [00:02<00:00, 36.86it/s]
Test loss:  0.2207454783441145
Test accuracy:  91.96
"""

Best 

"""

--- Positive Finished ---
Number of unique words in the vocabulary:  94278
100%|█████████████████████████████████████████████████████| 282/282 [01:18<00:00,  3.61it/s]
Epoch:  1  Training Loss:  0.38441073619727545
100%|███████████████████████████████████████████████████████| 32/32 [00:03<00:00,  9.73it/s]
Validation loss:  0.2634040513075888
Validation accuracy:  88.9472368092023
100%|███████████████████████████████████████████████████████| 79/79 [00:01<00:00, 40.86it/s]
Test loss:  0.2891537631991543
Test accuracy:  87.86
100%|█████████████████████████████████████████████████████| 282/282 [01:14<00:00,  3.80it/s]
Epoch:  2  Training Loss:  0.24475180378831024
100%|███████████████████████████████████████████████████████| 32/32 [00:03<00:00,  9.65it/s]
Validation loss:  0.23249138053506613
Validation accuracy:  90.37259314828707
100%|███████████████████████████████████████████████████████| 79/79 [00:02<00:00, 33.11it/s]
Test loss:  0.2419393051274215
Test accuracy:  90.48
100%|█████████████████████████████████████████████████████| 282/282 [01:10<00:00,  4.00it/s]
Epoch:  3  Training Loss:  0.21053232619842738
100%|███████████████████████████████████████████████████████| 32/32 [00:03<00:00,  9.25it/s]
Validation loss:  0.21302170562557876
Validation accuracy:  91.32283070767691
100%|███████████████████████████████████████████████████████| 79/79 [00:02<00:00, 32.46it/s]
Test loss:  0.22652507450761675
Test accuracy:  91.12
100%|█████████████████████████████████████████████████████| 282/282 [01:13<00:00,  3.82it/s]
Epoch:  4  Training Loss:  0.18593455227554267
100%|███████████████████████████████████████████████████████| 32/32 [00:04<00:00,  7.60it/s]
Validation loss:  0.20653660385869443
Validation accuracy:  92.14803700925232
100%|███████████████████████████████████████████████████████| 79/79 [00:02<00:00, 33.59it/s]
Test loss:  0.21841572498596168
Test accuracy:  92.02
100%|█████████████████████████████████████████████████████| 282/282 [01:14<00:00,  3.76it/s]
Epoch:  5  Training Loss:  0.16283098776025554
100%|███████████████████████████████████████████████████████| 32/32 [00:03<00:00, 10.09it/s]
Validation loss:  0.22509956825524569
Validation accuracy:  91.69792448112028
100%|███████████████████████████████████████████████████████| 79/79 [00:02<00:00, 31.24it/s]
Test loss:  0.2359839982435673
Test accuracy:  91.46
100%|█████████████████████████████████████████████████████| 282/282 [01:14<00:00,  3.79it/s]
Epoch:  6  Training Loss:  0.1398749968023799
100%|███████████████████████████████████████████████████████| 32/32 [00:03<00:00,  8.90it/s]
Validation loss:  0.21369225112721324
Validation accuracy:  91.74793698424605
100%|███████████████████████████████████████████████████████| 79/79 [00:02<00:00, 34.97it/s]
Test loss:  0.2249043362238739
Test accuracy:  92.04
100%|█████████████████████████████████████████████████████| 282/282 [01:11<00:00,  3.94it/s]
Epoch:  7  Training Loss:  0.11008717490241249
100%|███████████████████████████████████████████████████████| 32/32 [00:03<00:00, 10.51it/s]
Validation loss:  0.2426753060426563
Validation accuracy:  92.27306826706678
100%|███████████████████████████████████████████████████████| 79/79 [00:02<00:00, 31.95it/s]
Test loss:  0.2589881452201288
Test accuracy:  91.93
100%|█████████████████████████████████████████████████████| 282/282 [01:15<00:00,  3.72it/s]
Epoch:  8  Training Loss:  0.08624491952888086
100%|███████████████████████████████████████████████████████| 32/32 [00:03<00:00, 10.37it/s]
Validation loss:  0.28258371399715543
Validation accuracy:  90.64766191547888
100%|███████████████████████████████████████████████████████| 79/79 [00:02<00:00, 31.21it/s]
Test loss:  0.2882381547478181
Test accuracy:  90.83
100%|█████████████████████████████████████████████████████| 282/282 [00:59<00:00,  4.76it/s]
Epoch:  9  Training Loss:  0.06890656081761451
100%|███████████████████████████████████████████████████████| 32/32 [00:02<00:00, 13.54it/s]
Validation loss:  0.27801817655563354
Validation accuracy:  91.42285571392848
100%|███████████████████████████████████████████████████████| 79/79 [00:01<00:00, 47.51it/s]
Test loss:  0.3019331025926373
Test accuracy:  91.36
100%|█████████████████████████████████████████████████████| 282/282 [01:01<00:00,  4.58it/s]
Epoch:  10  Training Loss:  0.047818618818311724
100%|███████████████████████████████████████████████████████| 32/32 [00:03<00:00, 10.63it/s]
Validation loss:  0.32135023281443864
Validation accuracy:  90.87271817954489
100%|███████████████████████████████████████████████████████| 79/79 [00:02<00:00, 33.29it/s]
Test loss:  0.342044281242769
Test accuracy:  91.28
100%|█████████████████████████████████████████████████████| 282/282 [01:07<00:00,  4.18it/s]
Epoch:  11  Training Loss:  0.039267444928632454
100%|███████████████████████████████████████████████████████| 32/32 [00:02<00:00, 14.33it/s]
Validation loss:  0.3540804162621498
Validation accuracy:  91.47286821705426
100%|███████████████████████████████████████████████████████| 79/79 [00:01<00:00, 46.71it/s]
Test loss:  0.3697567610423776
Test accuracy:  91.27
100%|█████████████████████████████████████████████████████| 282/282 [00:55<00:00,  5.08it/s]
Epoch:  12  Training Loss:  0.03211013582809201
100%|███████████████████████████████████████████████████████| 32/32 [00:02<00:00, 12.97it/s]
Validation loss:  0.3231248748488724
Validation accuracy:  92.09802450612654
100%|███████████████████████████████████████████████████████| 79/79 [00:01<00:00, 45.07it/s]
Test loss:  0.35990333236471006
Test accuracy:  91.18
100%|█████████████████████████████████████████████████████| 282/282 [01:05<00:00,  4.31it/s]
Epoch:  13  Training Loss:  0.03492127772394224
100%|███████████████████████████████████████████████████████| 32/32 [00:03<00:00,  9.81it/s]
Validation loss:  0.3892567246221006
Validation accuracy:  91.49787446861716
100%|███████████████████████████████████████████████████████| 79/79 [00:02<00:00, 38.37it/s]
Test loss:  0.42755376160899294
Test accuracy:  91.28
100%|█████████████████████████████████████████████████████| 282/282 [01:07<00:00,  4.17it/s]
Epoch:  14  Training Loss:  0.020056407019770076
100%|███████████████████████████████████████████████████████| 32/32 [00:02<00:00, 11.79it/s]
Validation loss:  0.4355550520122051
Validation accuracy:  91.02275568892223
100%|███████████████████████████████████████████████████████| 79/79 [00:02<00:00, 36.16it/s]
Test loss:  0.4541135524647145
Test accuracy:  91.16
100%|█████████████████████████████████████████████████████| 282/282 [01:08<00:00,  4.10it/s]
Epoch:  15  Training Loss:  0.015734368238799566
100%|███████████████████████████████████████████████████████| 32/32 [00:03<00:00, 10.07it/s]
Validation loss:  0.4593230914324522
Validation accuracy:  91.19779944986247
100%|███████████████████████████████████████████████████████| 79/79 [00:01<00:00, 42.26it/s]
Test loss:  0.48673683235162424
Test accuracy:  91.07
100%|█████████████████████████████████████████████████████| 282/282 [01:07<00:00,  4.17it/s]
Epoch:  16  Training Loss:  0.01607038393424302
100%|███████████████████████████████████████████████████████| 32/32 [00:03<00:00,  9.70it/s]
Validation loss:  0.6430112831294537
Validation accuracy:  88.99724931232808
100%|███████████████████████████████████████████████████████| 79/79 [00:02<00:00, 30.74it/s]
Test loss:  0.6574981757734395
Test accuracy:  89.0
100%|█████████████████████████████████████████████████████| 282/282 [01:07<00:00,  4.16it/s]
Epoch:  17  Training Loss:  0.01350464289196757
100%|███████████████████████████████████████████████████████| 32/32 [00:03<00:00,  8.62it/s]
Validation loss:  0.5161228869110346
Validation accuracy:  91.52288072018004
Saving Model
100%|███████████████████████████████████████████████████████| 79/79 [00:02<00:00, 32.91it/s]
Test loss:  0.5505692804161506
Test accuracy:  91.07
100%|█████████████████████████████████████████████████████| 282/282 [00:55<00:00,  5.07it/s]
Epoch:  18  Training Loss:  0.01509798524083061
100%|███████████████████████████████████████████████████████| 32/32 [00:02<00:00, 14.61it/s]
Validation loss:  0.42512010131031275
Validation accuracy:  91.57289322330583
Saving Model
100%|███████████████████████████████████████████████████████| 79/79 [00:01<00:00, 46.32it/s]
Test loss:  0.4766239146643047
Test accuracy:  91.25
100%|█████████████████████████████████████████████████████| 282/282 [01:00<00:00,  4.65it/s]
Epoch:  19  Training Loss:  0.011367641878200791
100%|███████████████████████████████████████████████████████| 32/32 [00:02<00:00, 12.41it/s]
Validation loss:  0.5658172620460391
Validation accuracy:  90.97274318579645
100%|███████████████████████████████████████████████████████| 79/79 [00:01<00:00, 41.61it/s]
Test loss:  0.6146464538347872
Test accuracy:  90.96"""