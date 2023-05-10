# %%
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Splits data into batches of defined size
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from tqdm import tqdm

import random, os

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

train_dataset = 'ArithOpsTrain.xlsx'
df = pd.read_excel(train_dataset)

from torchtext.data import get_tokenizer

# ----------- Text Preprocessing -----------
#nlp = spacy.load("en_core_web_md")
tokenizer = get_tokenizer("basic_english")

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 0, 11, 10
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# %%
dataset = []
vocab, vocab_target = [], []
for idx, row in enumerate(df.iterrows()):
    if idx == 0:
        print(row[1][1], row[1][2], row[1][3], row[1][4], row[1][5])
        continue 
    
    context = tokenizer(row[1][1])
    vocab.extend(context)
    
    ques = tokenizer(row[1][2])
    vocab.extend(ques)

    ans = tokenizer(row[1][3])
    vocab_target.extend(ans)

    ip = [float(x) for x in row[1][4].split()]
    out = float(row[1][5])
    dataset.append((context, ques, ans, ip, out))

# %%
vocab_to_id = {}
ids = 1
for word in vocab:
    if word not in vocab_to_id:
        vocab_to_id[word] = ids
        ids += 1

vocab_target_to_id = {}
idt = 1
for word in vocab_target:
    if word not in vocab_target_to_id:
        vocab_target_to_id[word] = idt
        idt += 1

print(ids, idt)

# %%
dataset_tokenized = []

for context, ques, ans, ip, out in dataset:
    context_token = []
    for word in context:
        context_token.append(vocab_to_id[word])
    
    ques_token = []
    for word in ques:
        ques_token.append(vocab_to_id[word])

    ans_token = []
    for word in ans:
        ans_token.append(vocab_target_to_id[word])
    
    dataset_tokenized.append((context_token, ques_token, ans_token, ip, out))

# %%
dataset_tokenized[0]

# %%
def split_indices(n, val_pct):

    # Determine size of Validation set
    n_val = int(val_pct * n)

    # Create random permutation of 0 to n-1
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]

train_indices, val_indices = split_indices(len(dataset), 0.2)

from torch.nn.utils.rnn import pad_sequence

# ----------- Batching the data -----------
def collate_fn(instn):
    context = [torch.Tensor([ids] + x[1] +[ids] + x[0] + [ids]) for x in instn]
    ques = [torch.Tensor([ids] + x[1] + [ids]) for x in instn]
    ans = [torch.Tensor([idt+1] + x[2] + [idt]) for x in instn]
    ip = [x[3] for x in instn]
    out = [x[4] for x in instn]

    context_pad = pad_sequence(context, batch_first=True, padding_value=0).long()
    ques_pad = pad_sequence(ques, batch_first=True, padding_value=0).long()
    ans_pad = pad_sequence(ans, batch_first=True, padding_value=0).long()

    return (context_pad, ques_pad, ans_pad, ip, out)


batch_size = 128

train_sampler   = SubsetRandomSampler(train_indices)
train_loader    = DataLoader(dataset_tokenized, batch_size, sampler=train_sampler, collate_fn=collate_fn)

val_sampler     = SubsetRandomSampler(val_indices)
val_loader      = DataLoader(dataset_tokenized, batch_size, sampler=val_sampler, collate_fn=collate_fn)

import math

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 300):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert torch.Tensor of input indices into corresponding torch.Tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size, embed = None):
        super(TokenEmbedding, self).__init__()

        if embed != None:
            self.embedding = nn.Embedding.from_pretrained(embed)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.15,
                 embed = None):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size, embed)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor,
                src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))    # N * seq_len * 300 
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(tgt=self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory=memory,
                          tgt_mask=tgt_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


SRC_VOCAB_SIZE = ids+1
TGT_VOCAB_SIZE = idt+2
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512

NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DEVICE = torch.device("cuda:1")
model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)


model = model.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

opt = torch.optim.AdamW(model.parameters(), lr=0.0001)

def calculate_metric(ans, ip):
    out = []
    for b in range(len(ans)):
        stack = []
        expression = ans[b]
        for c in expression[::-1]:
            if "number" in c:
                idx = int(c[-1])
                if len(ip[b]) <= idx:
                    stack.append(float(ip[b][0]))
                    out.append(float(ip[b][0]) + float(ip[b][1]))
                    break
                stack.append(float(ip[b][idx]))
            elif c in "+-/*":
                try:
                    o1 = stack.pop()
                    o2 = stack.pop()
                except:
                    o1 = ip[b][0]
                    o2 = ip[b][1]
                if c == '+':
                    stack.append(o1 + o2)
                elif c == '-':
                    stack.append(o1 - o2)
                elif c == '*':
                    stack.append(o1 * o2)
                elif c == '/':
                    try:
                        stack.append(o1 / o2)
                    except:
                        print(ans[b], ip[b])
        try:
            out.append(stack.pop())
        except:
            out.append(float(ip[b][0]) + float(ip[b][1]))
    return out

id_to_vocab_target = {}
for i, v in vocab_target_to_id.items():
    id_to_vocab_target[v] = i
id_to_vocab_target[0] = "<pad>"
id_to_vocab_target[idt] = "EOS"
id_to_vocab_target[idt+1] = "<BOS>"
#id_to_vocab_target

print(id_to_vocab_target)

# ----------- Main Training Loop -----------
max_epoch = 25

best_test_acc = 0
for ep in range(max_epoch):

    epoch_loss = 0

    model.train()
    train_labels = []
    train_pred = []
    correct_train, B_train = 0, 0
    for cont, ques, ans, ip, out in train_loader:
        loss = 0
        cont = cont.to(DEVICE).transpose(0,1)       # Seq len x B 
        ques = ques.to(DEVICE).transpose(0,1)       # seq eq len x
        ans = ans.to(DEVICE).transpose(0,1)

        ans_inp = ans[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(cont, ans_inp)

        logits = model(cont, ans_inp, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        opt.zero_grad()

        ans_out = ans[1:, :]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), ans_out.reshape(-1))
        loss.backward()
        opt.step()

        epoch_loss += float(loss)

        pred = []
        ans = ans.transpose(0,1)
        logits = logits.transpose(0,1)
        for b in range(ans.shape[0]):
            p, ans_conv = [], []
            for i in range(ans.shape[1]-1):
                p.append(id_to_vocab_target[torch.argmax(logits[b,i,:]).item()])
                #ans_conv.append(id_to_vocab_target[ans[b, i].item()])
            pred.append(p)

        if ep > 2:
            outt = calculate_metric(pred, ip)
            #print(outt)
            for i in range(len(out)):
                #print(outt[i], out[i])
                if abs(float(outt[i]) - float(out[i])) < 0.0001:
                    correct_train += 1
            B_train += len(out)       

    print("Epoch: ", ep+1, " Training Loss: ", epoch_loss/ len(train_loader))
    #print("Train accuracy: ", accuracy_score(train_labels, train_pred)*50)
    if ep > 2:
        print("Train EM: ", (correct_train/ B_train)*100)

    val_epoch_loss = 0

    model.eval()
    train_labels = []
    train_pred = []
    correct_train, B_train = 0, 0
    for cont, ques, ans, ip, out in val_loader:
        loss = 0
        cont = cont.to(DEVICE).transpose(0,1)
        ques = ques.to(DEVICE).transpose(0,1)
        ans = ans.to(DEVICE).transpose(0,1)

        ans_inp = ans[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(cont, ans_inp)

        logits = model(cont, ans_inp, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        ans_out = ans[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), ans_out.reshape(-1))


        val_epoch_loss += float(loss)

        pred = []
        ans = ans.transpose(0,1)
        logits = logits.transpose(0,1)
        for b in range(ans.shape[0]):
            p = []
            for i in range(ans.shape[1]-1):
                p.append(id_to_vocab_target[torch.argmax(logits[b,i,:]).item()])
            pred.append(p)
        
        if ep > 2:
            outt = calculate_metric(pred, ip)
            for i in range(len(out)):
                if abs(float(outt[i]) - float(out[i])) < 0.0001:
                    correct_train += 1
            B_train += len(out)       

    print("Epoch: ", ep+1, " Val Loss: ", val_epoch_loss/ len(val_loader))
    #print("Train accuracy: ", accuracy_score(train_labels, train_pred)*50)
    if ep > 2:
        print("Val EM: ", (correct_train/ B_train)*100)
        if val_epoch_loss < min_val_epoch_loss:
            torch.save(model.state_dict(), "best_model.pt")
            print("-------Saving Model-------")
            min_val_epoch_loss = val_epoch_loss
    else:
        min_val_epoch_loss = float("infinity")

print(B_train, correct_train)


model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

model.load_state_dict(torch.load("best_model.pt")) 
model = model.to(DEVICE)

test_dataset = 'ArithOpsTestData1.xlsx'
df = pd.read_excel(test_dataset)

testdataset = []
vocab, vocab_target = [], []
for idx, row in enumerate(df.iterrows()):
    
    context = tokenizer(row[1][0])
    
    ques = tokenizer(row[1][1])

    ip = [float(x) for x in row[1][2].split()]
    testdataset.append((context, ques, ip))

testdataset_tokenized = []

for context, ques, ip in testdataset:
    context_token = []
    for word in context:
        if word in vocab_to_id:
            context_token.append(vocab_to_id[word])
    
    ques_token = []
    for word in ques:
        if word in vocab_to_id:
            ques_token.append(vocab_to_id[word])
    
    testdataset_tokenized.append((context_token, ques_token, ip))

# ----------- Batching the data -----------
def collate_fn_test(instn):
    context = [torch.Tensor([ids] + x[1] +[ids] + x[0] + [ids]) for x in instn]
    ques = [torch.Tensor([ids] + x[1] + [ids]) for x in instn]
    ip = [x[2] for x in instn]

    context_pad = pad_sequence(context, batch_first=True, padding_value=0).long()
    ques_pad = pad_sequence(ques, batch_first=True, padding_value=0).long()

    return (context_pad, ques_pad, ip)


batch_size = 1

test_loader    = DataLoader(testdataset_tokenized, batch_size, shuffle=False, collate_fn=collate_fn_test)


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)

        #print(ys.shape, memory.shape, tgt_mask.shape)
        out = model.transformer.decoder(tgt=model.positional_encoding(
                          model.tgt_tok_emb(ys)), memory=memory, tgt_mask=tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == idt:
            break
    return ys

model.eval()
output = []
for cont, ques, ip in tqdm(test_loader):
    loss = 0
    cont = cont.to(DEVICE)[0,:]
    ques = ques.to(DEVICE)

    cont = cont.unsqueeze(dim=1)

    num_tokens = cont.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    
    #print(cont.shape, src_mask.shape)
    ans_pred = []
    for out in greedy_decode(model, cont, src_mask, cont.shape[0]+5, idt+1):
        ans_pred.append(id_to_vocab_target[int(out.item())])
    print(ans_pred)
    output.append(calculate_metric([ans_pred], ip)[0])
print("Output: ", output)

"""
# ----------- Main Training Loop -----------
max_epoch = 1

best_test_acc = 0
for ep in range(max_epoch):

    epoch_loss = 0

    model.train()
    train_labels = []
    train_pred = []
    correct_train, B_train = 0, 0
    for cont, ques, ans, ip, out in tqdm(val_loader):
        loss = 0
        cont = cont.to(DEVICE)
        ques = ques.to(DEVICE)
        ans = ans.to(DEVICE)

        ans_inp = ans[:, :-1]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(cont, ans_inp)

        logits = model(cont, ans_inp, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        opt.zero_grad()

        ans_out = ans[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), ans_out.reshape(-1))


        epoch_loss += float(loss)

        for b in range(ans.shape[0]):
            p = [[]]
            for i in range(ans.shape[1]-1):
                print(id_to_vocab_target[torch.argmax(logits[b,i,:]).item()], end=" ")
                p[0].append(id_to_vocab_target[torch.argmax(logits[b,i,:]).item()])
                print(id_to_vocab_target[ans[b, i].item()])
            print()
            outt = calculate_metric(p, [ip[b]])
            if abs(float(outt[0]) - float(out[b])) < 0.0001:
                print("incorrect, ", float(outt[0]),float(out[b]))
            print(outt, out[b])
        #y_hat = torch.softmax(y_hat, dim = 1).argmax(dim=1)

        #correct, B = exact_match(y_hat, yb)
        #correct_train += correct
        #B_train += B
        #train_labels.extend(yb.cpu().detach().numpy())
        #train_pred.extend(y_hat.cpu().detach().numpy())

    print("Epoch: ", ep+1, " Training Loss: ", epoch_loss)
    #print("Train accuracy: ", accuracy_score(train_labels, train_pred)*100)
    #print("Train EM: ", (correct_train/ B_train)*100)

        

# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

"""


