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

from transformers import BertTokenizer, BertForSequenceClassification


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
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt", lower_case=True)
print(inputs)
with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()

# %%
from torchtext.data import get_tokenizer


positive_set = []

for line in pos_set:
    positive_set.append((line, 1))


print("--- Positive Finished ---")

negative_set = []
for line in neg_set:
    negative_set.append((line, 0))

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


# %%
if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

#model = Encoder(len(ids_vocab), 2500, 200, 128, 4, 2, 0.1, device)
model.to(device)
opt_c = torch.optim.AdamW(model.parameters(), lr = 2e-5, eps = 1e-8) # Same as Adam with weight decay = 0.001
# loss_fn_c = F.cross_entropy #Tried Cross Entropy with log_softmax output function - gave similar results
loss_fn_c = F.cross_entropy

# %%
# ----------- Main Training Loop -----------
max_epoch = 10

best_test_acc = 0
for ep in range(max_epoch):

    epoch_loss = 0

    model.train()
    itera = tqdm(train_loader)
    for xb, yb in itera:
        xb = xb.to(device)
        yb = yb.to(device).long()

        #y_hat = model(**xb, labels=yb).logits
        y_hat = model(**xb, labels=yb.unsqueeze(dim=1))
        loss = y_hat[0]
        #loss = loss_fn_c(y_hat, yb.long())
        #itera.set_postfix({"loss": loss})
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

            y_hat = model(**xb).logits
            loss = loss_fn_c(y_hat, yb.long())

            val_epoch_loss += float(loss)

            val_labels.extend(torch.round(yb).cpu().detach().numpy())
            val_pred.extend(y_hat.argmax(dim=1).cpu().detach().numpy())

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

            y_hat = model(**xb).logits
            loss = loss_fn_c(y_hat, yb.long())

            test_epoch_loss += float(loss)

            test_labels.extend(torch.round(yb).cpu().detach().numpy())
            test_pred.extend(y_hat.argmax(dim=1).cpu().detach().numpy())


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


