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

file_errors_location = 'ArithOpsTrain.xlsx'
df = pd.read_excel(file_errors_location)

dataset = []
for idx, row in enumerate(df.iterrows()):
    if idx == 0:
        print(row[1][1], row[1][2], row[1][3], row[1][4], row[1][5])
        continue 

    ans = row[1][3].replace("number0", "0", -1)
    ans = ans.replace("number1", "1", -1)
    ans = ans.replace("number2", "2", -1)
    ans = ans.replace("number3", "3", -1)
    ans = ans.replace("number4", "4", -1)
    dataset.append((row[1][1], row[1][2], ans, row[1][4], row[1][5]))

def split_indices(n, val_pct):

    # Determine size of Validation set
    n_val = int(val_pct * n)

    # Create random permutation of 0 to n-1
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]

train_indices, val_indices = split_indices(len(dataset), 0.2)

from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
tok = BartTokenizer.from_pretrained("facebook/bart-large")

# ----------- Batching the data -----------
def collate_fn(instn):

    qc = tok([x[0] for x in instn], return_tensors="pt", truncation=False, padding=True)
    question = tok([x[1] for x in instn], return_tensors="pt", truncation=False, padding=True)
    answer = tok([x[2] + "</s>" for x in instn], return_tensors="pt", truncation=False, padding=True)
    input_val = [x[3] for x in instn]
    output = [x[4] for x in instn]

    return (qc, question, answer, input_val, output)

batch_size = 64

train_sampler   = SubsetRandomSampler(train_indices)
trainloader    = DataLoader(dataset, batch_size, sampler=train_sampler, collate_fn=collate_fn)

val_sampler     = SubsetRandomSampler(val_indices)
valloader      = DataLoader(dataset, batch_size, sampler=val_sampler, collate_fn=collate_fn)

device = torch.device("cuda:0")
model.to(device)

loss_fn = nn.CrossEntropyLoss(reduction='none')
opt = torch.optim.AdamW(model.parameters(), lr = 0.0001)

def calculate_metric(ans, ip, out):
    for i in range(ans):
        pass
    return ans

for ep in range(10):

    model = model.train()
    epoch_loss = 0
  
    for qc, qs, ans, ip, out in tqdm(trainloader):
        loss = 0
        qc_input_ids = qc["input_ids"].to(device)
        qc_attention_mask = qc["attention_mask"].to(device)
        qs_input_ids = qs["input_ids"].to(device)
        qs_attention_mask = qs["attention_mask"].to(device)
        ans_input_ids = ans["input_ids"].to(device)
        ans_attention_mask = ans["attention_mask"].to(device)

        outputs = model(input_ids=qc_input_ids, attention_mask=qc_attention_mask, decoder_input_ids=qs_input_ids, decoder_attention_mask=qs_attention_mask).logits

        #pad = torch.zeros(qc_input_ids.shape[0], qc_input_ids.shape[1]- ans_input_ids.shape[1]).to(device)
        #target = torch.cat((ans_input_ids, pad), dim = 1)
        #print(outputs.shape, ans_input_ids.shape, target.shape)
        #ans_input_ids = ans_input_ids * ans_attention_mask
        #loss = loss_fn(outputs, target)

        #correct_predictions += torch.sum(preds.argmax(dim=1).squeeze() == target)
        #losses.append(loss.item())

        #for i in range(ans_input_ids.shape[1]):
        #   loss += torch.mean(loss_fn(outputs[:, i], ans_input_ids[:, i].long()) * ans_attention_mask[:, i])
        #print(torch.transpose(outputs,1,2).shape, ans_input_ids.shape)
        loss = torch.mean(loss_fn(torch.transpose(outputs,1,2)[:,:,:ans_input_ids.shape[1]-1], ans_input_ids[:, 1:].long()) * ans_attention_mask[:, 1:])

        #print(tok.batch_decode(torch.argmax(outputs, dim =2), skip_special_tokens=True))

        loss.backward()
        opt.step()
        opt.zero_grad()
        epoch_loss += loss
    print(loss)
    print(tok.batch_decode(torch.argmax(outputs, dim =2), skip_special_tokens=True))
    print("correct",tok.batch_decode(ans_input_ids, skip_special_tokens=True) )

    val_epoch_loss = 0
  
    for qc, qs, ans, ip, out in tqdm(valloader):
        loss = 0
        qc_input_ids = qc["input_ids"].to(device)
        qc_attention_mask = qc["attention_mask"].to(device)
        qs_input_ids = qs["input_ids"].to(device)
        qs_attention_mask = qs["attention_mask"].to(device)
        ans_input_ids = ans["input_ids"].to(device)
        ans_attention_mask = ans["attention_mask"].to(device)

        outputs = model(input_ids=qc_input_ids, attention_mask=qc_attention_mask, decoder_input_ids=qs_input_ids, decoder_attention_mask=qs_attention_mask).logits

        #pad = torch.zeros(qc_input_ids.shape[0], qc_input_ids.shape[1]- ans_input_ids.shape[1]).to(device)
        #target = torch.cat((ans_input_ids, pad), dim = 1)
        #print(outputs.shape, ans_input_ids.shape, target.shape)

        loss = torch.mean(loss_fn(torch.transpose(outputs,1,2)[:,:,:ans_input_ids.shape[1]-1], ans_input_ids[:, 1:].long()) * ans_attention_mask[:, 1:])
        val_epoch_loss += loss
    print("VALIDATION: ", val_epoch_loss)


#model.eval()
for qc, qs, ans, ip, out in tqdm(trainloader):
        loss = 0
        qc_input_ids = qc["input_ids"].to(device)
        qc_attention_mask = qc["attention_mask"].to(device)
        qs_input_ids = qs["input_ids"].to(device)
        qs_attention_mask = qs["attention_mask"].to(device)
        ans_input_ids = ans["input_ids"].to(device)
        ans_attention_mask = ans["attention_mask"].to(device)

        outputs = model(input_ids=qc_input_ids, attention_mask=qc_attention_mask, decoder_input_ids=qs_input_ids, decoder_attention_mask=qs_attention_mask).logits

        pad = torch.zeros(qc_input_ids.shape[0], qc_input_ids.shape[1]- ans_input_ids.shape[1]).to(device)
        target = torch.cat((ans_input_ids, pad), dim = 1)

        #print(torch.argmax(outputs, dim =2))
        print(tok.batch_decode(torch.argmax(outputs, dim =2), skip_special_tokens=True))
        print("correct",tok.batch_decode(ans_input_ids, skip_special_tokens=True) )