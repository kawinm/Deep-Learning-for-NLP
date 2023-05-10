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
    
#5
set_seed(1)

positive_set = open("Train.pos", "r", encoding="latin-1").read()
negative_set = open("Train.neg", "r", encoding="latin-1").read()
test_set = open("TestData", "r", encoding="latin-1").read()

# Removing punctuations
chars_to_remove = ['--', '`', '~', '<', '>', '*', '{', '}', '^', '=', '_', '[', ']', '|', '- ', '.', ',']

positive_set = positive_set.replace('<br />', " ", -1)
for char in chars_to_remove:
    positive_set = positive_set.replace(char, " ", -1)

negative_set = negative_set.replace('<br />', " ", -1)
for char in chars_to_remove:
    negative_set = negative_set.replace(char, " ", -1)

test_set = test_set.replace('<br />', " ", -1)
for char in chars_to_remove:
    test_set = test_set.replace(char, " ", -1)

pos_set = positive_set.split("\n")[:-1]
neg_set = negative_set.split("\n")[:-1]
t_set = test_set.split("\n")[:-1]


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

test_set = []
for idx, line in enumerate(t_set):

    # Tokenizes the input text into words
    tokens = tokenizer(line)

    if idx < 331:
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

emb_dim = 600


fasttext = FastText()
embeds = torch.zeros(len(ids_vocab) + 1, emb_dim)
n = 0

for token, idx in ids_vocab.items():
    embeds[idx][:300] = global_vectors[token]
    embeds[idx][300:] = fasttext[token]

    if torch.sum(embeds[idx]) == 0:
        embeds[idx] = torch.randn(600)
        n +=1
print(n)
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
print("Number of samples in test set:", len(test_set))
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

test_loader     = DataLoader(token_corpus_test, batch_size, collate_fn=collate_fn)

# ----------- Model -----------
class BILSTM(nn.Module):
    
    def __init__(self, embeds):
        super().__init__()

        self.embeddings = nn.Embedding.from_pretrained(embeds, padding_idx=0, freeze=True)

        self.lstm = nn.GRU(input_size = 600, hidden_size = 300, num_layers = 2, batch_first = True, dropout=0.5)

        self.lin1 = nn.Linear(300, 128)
        self.lin2 = nn.Linear(128, 1)

        self.lin3 = nn.Linear(300, 1)

        self.embeddings2 = nn.Embedding.from_pretrained(embeds, padding_idx=0, freeze=True)

        self.lstm2 = nn.LSTM(input_size = 600, hidden_size = 300, num_layers = 2, batch_first = True, bidirectional = True, dropout=0.5)

        self.lin4 = nn.Linear(600, 128)
        self.lin5 = nn.Linear(128, 1)

        self.lin6 = nn.Linear(600, 1)

        #self.alpha = nn.Parameter(torch.Tensor([1]))
        #self.beta = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, xb, tsne = False):
        
        padding_masking = xb > 0
        padding_masking = padding_masking.type(torch.int)

        xe = self.embeddings(xb)
        out, y = self.lstm(xe)
        
        x = self.lin3(out).squeeze(dim=-1)
        x = x * padding_masking
        x = torch.softmax(x, dim=-1).unsqueeze(dim=1)
        x = torch.bmm(x, out).squeeze(dim=1)

        #x = torch.cat(( y[2][ :, :], y[3][ :, :]), dim = 1)
        x = self.lin1(y[1][ :, :])

        x = F.relu(x)
        x = self.lin2(x)
        x1 = torch.sigmoid(x)

        xe = self.embeddings2(xb)
        out, yy = self.lstm2(xe)
        
        x = self.lin6(out).squeeze(dim=-1)
        x = x * padding_masking
        x = torch.softmax(x, dim=-1).unsqueeze(dim=1)
        x = torch.bmm(x, out).squeeze(dim=1)

        #x = torch.cat((x, y[2][ :, :], y[3][ :, :]), dim = 1)
        x = self.lin4(x)

        x = F.relu(x)
        x = self.lin5(x)
        x2 = torch.sigmoid(x)
        #print(self.alpha.item(), self.beta.item())
        #return (x1 * self.alpha + x2 * self.beta) / (self.alpha + self.beta)
        return x1

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

model = BILSTM(embeds)
model.to(device)
opt_c = torch.optim.AdamW(model.parameters(), lr = 0.0005)
# loss_fn_c = F.cross_entropy - Tried Cross Entropy with log_softmax output function - gave similar results
loss_fn_c = F.binary_cross_entropy
#sch = torch.optim.lr_scheduler.CyclicLR(opt_c, 0.0001, 0.0005, step_size_up=200, cycle_momentum=False,)
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

        #sch.step()

        nn.utils.clip_grad_norm_(model.parameters(), 2)

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
        torch.save(model.state_dict(), "best_model_"+str(ep)+".pt")
    
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

100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 47.80it/s]
Epoch:  1  Training Loss:  0.5651552168946518
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 145.81it/s]
Validation loss:  0.5189962208271026
Validation accuracy:  74.8
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 189.66it/s]
Test loss:  0.5045464549745832
Test accuracy:  76.58610271903324
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 57.81it/s]
Epoch:  2  Training Loss:  0.4533107823447177
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 162.10it/s]
Validation loss:  0.48231709003448486
Validation accuracy:  75.8
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 186.95it/s]
Test loss:  0.4664456376007625
Test accuracy:  76.28398791540786
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 59.76it/s]
Epoch:  3  Training Loss:  0.4136428215001759
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 161.61it/s]
Validation loss:  0.43588160872459414
Validation accuracy:  78.2
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 188.78it/s]
Test loss:  0.47062196476118906
Test accuracy:  77.19033232628398
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 54.40it/s]
Epoch:  4  Training Loss:  0.37709598996137317
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 164.34it/s]
Validation loss:  0.4222192704677582
Validation accuracy:  77.60000000000001
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 189.75it/s]
Test loss:  0.43359094858169556
Test accuracy:  80.06042296072508
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 57.05it/s]
Epoch:  5  Training Loss:  0.34165340771800595
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 141.65it/s]
Validation loss:  0.43707741498947145
Validation accuracy:  80.4
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 182.77it/s]
Test loss:  0.4604837937014444
Test accuracy:  79.60725075528701
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 61.67it/s]
Epoch:  6  Training Loss:  0.31172240125505546
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 162.24it/s]
Validation loss:  0.44407609701156614
Validation accuracy:  78.4
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 186.43it/s]
Test loss:  0.4313718633992331
Test accuracy:  81.1178247734139
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 60.55it/s]
Epoch:  7  Training Loss:  0.2647310733795166
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 164.59it/s]
Validation loss:  0.4724866211414337
Validation accuracy:  78.4
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 188.72it/s]
Test loss:  0.4547896257468632
Test accuracy:  80.66465256797582
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 61.42it/s]
Epoch:  8  Training Loss:  0.22249194030698977
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 162.94it/s]
Validation loss:  0.5348136782646179
Validation accuracy:  78.2
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 189.31it/s]
Test loss:  0.5665162929466793
Test accuracy:  81.2688821752266
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 57.30it/s]
Epoch:  9  Training Loss:  0.185594914932
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 163.67it/s]
Validation loss:  0.6146192729473114
Validation accuracy:  80.60000000000001
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 186.04it/s]
Test loss:  0.629418637071337
Test accuracy:  80.21148036253777
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 59.21it/s]
Epoch:  10  Training Loss:  0.15499295482510014
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 160.22it/s]
Validation loss:  0.6511508285999298
Validation accuracy:  77.8
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 184.60it/s]
Test loss:  0.6000965493065971
Test accuracy:  78.85196374622356
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 58.84it/s]
Epoch:  11  Training Loss:  0.12008658333828574
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 161.93it/s]
Validation loss:  0.7279137909412384
Validation accuracy:  79.60000000000001
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 185.26it/s]
Test loss:  0.8127897254058293
Test accuracy:  79.45619335347432
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 54.31it/s]
Epoch:  12  Training Loss:  0.09550228142424634
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 166.64it/s]
Validation loss:  0.7040274024009705
Validation accuracy:  77.4
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 184.85it/s]
Test loss:  0.635592017854963
Test accuracy:  80.81570996978851
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 52.27it/s]
Epoch:  13  Training Loss:  0.0819650442192429
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 161.66it/s]
Validation loss:  0.8160832643508911
Validation accuracy:  78.8
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 190.21it/s]
Test loss:  0.7841978328568595
Test accuracy:  79.7583081570997
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 59.16it/s]
Epoch:  14  Training Loss:  0.06583161936386635
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 164.45it/s]
Validation loss:  0.8913721084594727
Validation accuracy:  77.0
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 190.64it/s]
Test loss:  0.860488338129861
Test accuracy:  78.85196374622356
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 53.46it/s]
Epoch:  15  Training Loss:  0.037771747655872455
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 161.93it/s]
Validation loss:  0.984852921962738
Validation accuracy:  79.60000000000001
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 190.62it/s]
Test loss:  0.9608741062028068
Test accuracy:  80.66465256797582
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 53.51it/s]
Epoch:  16  Training Loss:  0.03137376620311682
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 161.43it/s]
Validation loss:  1.2103933572769165
Validation accuracy:  78.0
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 190.09it/s]
Test loss:  1.163402693612235
Test accuracy:  80.21148036253777
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 60.07it/s]
Epoch:  17  Training Loss:  0.044646614618403344
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 145.90it/s]
Validation loss:  1.064273738861084
Validation accuracy:  79.2
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 183.59it/s]
Test loss:  0.9577541521617344
Test accuracy:  80.66465256797582
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 53.54it/s]
Epoch:  18  Training Loss:  0.026642753053946713
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 164.00it/s]
Validation loss:  1.2323540449142456
Validation accuracy:  78.8
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 186.08it/s]
Test loss:  1.2481581739016943
Test accuracy:  80.06042296072508
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 57.19it/s]
Epoch:  19  Training Loss:  0.02090908911226219
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 160.75it/s]
Validation loss:  1.243124008178711
Validation accuracy:  78.8
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 188.80it/s]
Test loss:  1.216683260032109
Test accuracy:  79.7583081570997
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 58.97it/s]
Epoch:  20  Training Loss:  0.012905576966065717
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 159.31it/s]
Validation loss:  1.3276618242263794
Validation accuracy:  80.60000000000001
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 176.33it/s]
Test loss:  1.4788657086236137
Test accuracy:  80.06042296072508
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 54.03it/s]
Epoch:  21  Training Loss:  0.02482647629252537
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 163.84it/s]
Validation loss:  1.1455819487571717
Validation accuracy:  79.4
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 174.30it/s]
Test loss:  1.0792984366416931
Test accuracy:  79.45619335347432
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 59.92it/s]
Epoch:  22  Training Loss:  0.012154687876433232
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 160.18it/s]
Validation loss:  1.4472072124481201
Validation accuracy:  78.2
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 187.34it/s]
Test loss:  1.3367221525737218
Test accuracy:  79.60725075528701
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 59.34it/s]
Epoch:  23  Training Loss:  0.019921341288442675
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 159.14it/s]
Validation loss:  1.3727387547492982
Validation accuracy:  78.4
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 191.07it/s]
Test loss:  1.675061583518982
Test accuracy:  80.36253776435045
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 56.56it/s]
Epoch:  24  Training Loss:  0.018406329254350182
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 150.00it/s]
Validation loss:  1.2451785326004028
Validation accuracy:  79.80000000000001
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 180.18it/s]
Test loss:  1.2621142438479833
Test accuracy:  79.00302114803625
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 60.36it/s]
Epoch:  25  Training Loss:  0.010648871624787141
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 161.57it/s]
Validation loss:  1.429085659980774
Validation accuracy:  77.0
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 192.25it/s]
Test loss:  1.3840453369276864
Test accuracy:  79.60725075528701
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 56.11it/s]
Epoch:  26  Training Loss:  0.015299280996336357
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 156.47it/s]
Validation loss:  1.5169595003128051
Validation accuracy:  77.8
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 186.12it/s]
Test loss:  1.7164651325770788
Test accuracy:  81.2688821752266
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 54.15it/s]
Epoch:  27  Training Loss:  0.020173222612663123
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 164.51it/s]
Validation loss:  1.3925410032272338
Validation accuracy:  79.0
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 186.04it/s]
Test loss:  1.3808154038020544
Test accuracy:  79.60725075528701
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 61.00it/s]
Epoch:  28  Training Loss:  0.015093788400439448
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 161.65it/s]
Validation loss:  1.3498054504394532
Validation accuracy:  79.2
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 186.50it/s]
Test loss:  1.190738993031638
Test accuracy:  80.96676737160121
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 61.57it/s]
Epoch:  29  Training Loss:  0.011447521510815836
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 160.86it/s]
Validation loss:  1.4322837114334106
Validation accuracy:  78.8
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 187.24it/s]
Test loss:  1.3466019034385681
Test accuracy:  79.7583081570997
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 50.04it/s]
Epoch:  30  Training Loss:  0.014225638974580521
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 148.40it/s]
Validation loss:  1.3518836736679076
Validation accuracy:  81.39999999999999
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 171.20it/s]
Test loss:  1.5202834350722176
Test accuracy:  83.08157099697885
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 57.87it/s]
Epoch:  31  Training Loss:  0.003519452555567688
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 161.25it/s]
Validation loss:  1.5360361099243165
Validation accuracy:  80.4
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 186.36it/s]
Test loss:  1.7900353840419225
Test accuracy:  81.1178247734139
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 60.02it/s]
Epoch:  32  Training Loss:  0.007959295414529104
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 161.03it/s]
Validation loss:  1.4264266967773438
Validation accuracy:  78.8
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 187.72it/s]
Test loss:  1.5854366847446986
Test accuracy:  82.02416918429003
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 59.22it/s]
Epoch:  33  Training Loss:  0.004432507489553628
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 149.00it/s]
Validation loss:  1.867227268218994
Validation accuracy:  80.4
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 188.77it/s]
Test loss:  1.79685149874006
Test accuracy:  82.17522658610272
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 53.50it/s]
Epoch:  34  Training Loss:  0.0030977569031069596
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 164.09it/s]
Validation loss:  2.195110535621643
Validation accuracy:  79.4
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 163.51it/s]
Test loss:  2.039061665534973
Test accuracy:  80.81570996978851
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 61.59it/s]
Epoch:  35  Training Loss:  0.011725471002634885
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 158.83it/s]
Validation loss:  1.502583909034729
Validation accuracy:  78.0
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 170.05it/s]
Test loss:  1.6752537999834334
Test accuracy:  80.66465256797582
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 54.94it/s]
Epoch:  36  Training Loss:  0.01218164354642412
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 163.63it/s]
Validation loss:  1.6909342765808106
Validation accuracy:  78.8
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 189.67it/s]
Test loss:  1.7252279009137834
Test accuracy:  80.36253776435045
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 54.77it/s]
Epoch:  37  Training Loss:  0.009750128013700688
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 150.75it/s]
Validation loss:  1.4840702056884765
Validation accuracy:  78.2
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 182.73it/s]
Test loss:  1.7059626068387712
Test accuracy:  80.21148036253777
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 59.76it/s]
Epoch:  38  Training Loss:  0.010750667256529836
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 164.66it/s]
Validation loss:  1.387263250350952
Validation accuracy:  78.2
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 160.81it/s]
Test loss:  1.781611638409751
Test accuracy:  81.41993957703929
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 56.12it/s]
Epoch:  39  Training Loss:  0.005120940912629043
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 162.14it/s]
Validation loss:  1.8135441780090331
Validation accuracy:  78.4
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 175.40it/s]
Test loss:  2.05621269771031
Test accuracy:  80.66465256797582
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 58.70it/s]
Epoch:  40  Training Loss:  0.0038965661145323937
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 159.63it/s]
Validation loss:  1.8122987270355224
Validation accuracy:  78.2
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 184.37it/s]
Test loss:  1.9679317304066248
Test accuracy:  79.60725075528701
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 55.54it/s]
Epoch:  41  Training Loss:  0.0047069611500774045
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 158.62it/s]
Validation loss:  1.6973422765731812
Validation accuracy:  79.0
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 171.59it/s]
Test loss:  2.073854531560625
Test accuracy:  79.90936555891238
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 54.17it/s]
Epoch:  42  Training Loss:  0.008077050844401943
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 159.17it/s]
Validation loss:  1.5918452501296998
Validation accuracy:  80.0
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 182.79it/s]
Test loss:  2.4274889401027133
Test accuracy:  80.36253776435045
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 56.83it/s]
Epoch:  43  Training Loss:  0.007996709422073572
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 161.35it/s]
Validation loss:  1.8559942960739135
Validation accuracy:  78.8
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 185.99it/s]
Test loss:  2.4477510281971524
Test accuracy:  80.66465256797582
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 55.92it/s]
Epoch:  44  Training Loss:  0.017573326099304003
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 163.14it/s]
Validation loss:  1.1252426981925965
Validation accuracy:  79.2
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 183.16it/s]
Test loss:  1.3047533290726798
Test accuracy:  79.60725075528701
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 58.86it/s]
Epoch:  45  Training Loss:  0.009450784364813253
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 162.68it/s]
Validation loss:  1.2583684921264648
Validation accuracy:  78.4
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 189.07it/s]
Test loss:  1.5085762398583549
Test accuracy:  80.51359516616314
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 51.62it/s]
Epoch:  46  Training Loss:  0.003265651795204337
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 159.56it/s]
Validation loss:  1.5839773893356324
Validation accuracy:  78.2
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 184.85it/s]
Test loss:  1.844653844833374
Test accuracy:  79.45619335347432
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 52.45it/s]
Epoch:  47  Training Loss:  0.004957018890510019
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 148.07it/s]
Validation loss:  1.457999086380005
Validation accuracy:  79.4
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 162.69it/s]
Test loss:  1.8149877446038383
Test accuracy:  78.70090634441088
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 58.23it/s]
Epoch:  48  Training Loss:  0.0007812651416890348
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 142.96it/s]
Validation loss:  2.4042797088623047
Validation accuracy:  80.60000000000001
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 184.13it/s]
Test loss:  2.265595027378627
Test accuracy:  80.66465256797582
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 57.98it/s]
Epoch:  49  Training Loss:  0.0010456514798007496
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 164.15it/s]
Validation loss:  2.138386535644531
Validation accuracy:  78.8
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 180.48it/s]
Test loss:  2.1800647974014282
Test accuracy:  79.30513595166163
100%|█████████████████████████████████████████████████████████████████████| 95/95 [00:01<00:00, 53.18it/s]
Epoch:  50  Training Loss:  0.01241635227436889
100%|██████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 163.74it/s]
Validation loss:  1.3012142181396484
Validation accuracy:  79.0
Saving Model
100%|██████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 186.52it/s]
Test loss:  1.2389126930918013
Test accuracy:  79.45619335347432"""