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

print("--- Negative Finished ---")

# %%
# Sorting the samples based on their sequence length
def sort_key(s):
    return len(s[0])
    
#data_set = sorted(data_set, key=sort_key)   # Sorting did not gave better result

# %%
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

# %%
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

# %%
# Train-Valid split of 90-10
def split_indices(n, val_pct):

    # Determine size of Validation set
    n_val = int(val_pct * n)

    # Create random permutation of 0 to n-1
    idxs = np.random.permutation(n)
    #return np.sort(idxs[n_val:]), np.sort(idxs[:n_val])
    return idxs[n_val:], idxs[:n_val]

train_pos_indices, val_pos_indices = split_indices(len(pos_set), 0.1)
train_neg_indices, val_neg_indices = split_indices(len(neg_set), 0.1)

train_indices = np.concatenate((train_pos_indices, train_neg_indices+len(pos_set)-1))
val_indices = np.concatenate((val_pos_indices, val_neg_indices+len(pos_set)-1))

# %%
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


batch_size = 64

train_sampler   = SubsetRandomSampler(train_indices)
train_loader    = DataLoader(token_corpus, batch_size, sampler=train_sampler, collate_fn=collate_fn)

val_sampler     = SubsetRandomSampler(val_indices)
val_loader      = DataLoader(token_corpus, batch_size, sampler=val_sampler, collate_fn=collate_fn)

def tokenize_test(corpus, ids_vocab):
    """
        Converts words in the dataset to integer tokens
    """

    tokenized_corpus = []
    for line, sentiment, idx in corpus:
        new_line = []
        for i, word in enumerate(line):
            if word in ids_vocab and (i == 0 or word != line[i-1]):
                new_line.append(ids_vocab[word])

        new_line = torch.Tensor(new_line).long()
        tokenized_corpus.append((new_line, sentiment, idx))

    return tokenized_corpus


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

        tokens = tokenizer(review)

        if rows['sentiment'] == 'positive':
            test_set.append((tokens, 1, idx))
        else:
            test_set.append((tokens, 0, idx))

#test_set = sorted(test_set, key=sort_key)

# ----------- Batching the data -----------
def collate_fn_test(instn):

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

    idx = torch.Tensor([x[2] for x in instn])

    return (padded_sent, labels, idx)

token_corpus_test = tokenize_test(test_set, ids_vocab)

test_loader      = DataLoader(token_corpus_test, batch_size, collate_fn=collate_fn_test)

# %%
class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]   

# %%
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)
        
        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

# %%
class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

# %%
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out

# %%
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# %%
class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information
        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

# %%
class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information
        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)

# %%
class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x

# %%
class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.lin1 = nn.Linear(200, 128)
        self.lin2 = nn.Linear(128, 32)
        self.lin3 = nn.Linear(32, 1)

        self.attn = nn.Linear(200, 1)

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)
            
        out = x
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
def make_pad_mask(q, k):
        len_q, len_k = q.size(1), k.size(1)

        src_pad_idx = 0
        # batch_size x 1 x 1 x len_k
        k = k.ne(src_pad_idx).unsqueeze(1).unsqueeze(2)
        # batch_size x 1 x len_q x len_k
        k = k.repeat(1, 1, len_q, 1)

        # batch_size x 1 x len_q x 1
        q = q.ne(src_pad_idx).unsqueeze(1).unsqueeze(3)
        # batch_size x 1 x len_q x len_k
        q = q.repeat(1, 1, 1, len_k)

        mask = k & q
        return 

# %%
if torch.cuda.is_available():
    device = torch.device("cuda:2")
else:
    device = torch.device("cpu")


model = Encoder(len(ids_vocab), 3000, 200, 64, 2, 1, 0.1, device)
model.to(device)
opt_c = torch.optim.AdamW(model.parameters(), lr = 0.001) # Same as Adam with weight decay = 0.001
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

        src_mask = make_pad_mask(xb, xb)

        y_hat = model(xb, src_mask)
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

            src_mask = make_pad_mask(xb, xb)

            y_hat = model(xb, src_mask)
            loss = loss_fn_c(y_hat.squeeze(), yb)

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
        for xb, yb, idx in tqdm(test_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            src_mask = make_pad_mask(xb, xb)

            y_hat = model(xb, src_mask)
            loss = loss_fn_c(y_hat.squeeze(), yb)

            test_epoch_loss += float(loss)

            test_labels.extend(torch.round(yb).cpu().detach().numpy())
            test_pred.extend(y_hat.round().cpu().detach().numpy())

    print("Test loss: ", test_epoch_loss/len(test_loader))
    print("Test accuracy: ", accuracy_score(test_labels, test_pred)*100)

# %%
# Tokenization function

# %%
model = Encoder(len(ids_vocab), 3000, 200, 64, 2, 1, 0.1, device)
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
print(n, 79*128)
print("Test loss: ", test_epoch_loss/len(test_loader))
print("Test accuracy: ", accuracy_score(test_labels, test_pred)*100)

# %%
# Seed doesn't work in Jupyter notebook, to replicate my results, kindly, run it as .py file

"""
#Epoch:  7  Training Loss:  0.14017409423687108
100%|██████████████████████████████████████████████████| 63/63 [00:00<00:00, 90.96it/s]
Validation loss:  0.3170031209786733
Validation accuracy:  88.9722430607652
Saving Model
100%|███████████████████████████████████████████████| 157/157 [00:01<00:00, 103.51it/s]
Test loss:  0.34445758443918956
Test accuracy:  88.47
"""

