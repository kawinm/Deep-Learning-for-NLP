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


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device

# %%
import pandas as pd
df = pd.read_csv('Train dataset.csv')
df.shape

# %%
df_test = pd.read_csv('E0334 Assignment2 Test Dataset.csv')

# %% [markdown]
# We have 50k movie reviews. Let's check do we have missing values?

# %% [markdown]
# Great! we have no missing values in sentiment and review text. Let's check do we have class imbalance?

# %%
def to_sentiment(rating):
  rating = str(rating)
  if rating == 'positive':
    return 0
  else: 
    return 1

df['sentiment_score'] = df.sentiment.apply(to_sentiment)
df_test['sentiment_score'] = df_test.sentiment.apply(to_sentiment)

# %% [markdown]
# The count plot shows that the dataset is balanced and have no missing values.

# %% [markdown]
# ##Data Preprocessing

# %% [markdown]
# The dataset needs to be pre processed before passing them to the model. So, we need to add special tokens to separate sentences and do classification. we will pass sequences of constant length (Padding) and have to create array of 0s and 1s which is nothing but attention mask.

# %% [markdown]
# The Transformers library provides a wide variety of Transformer models. It works with TensorFlow and PyTorch! we will be working with Pytorch and will be using BERT model for sentiment analysis. We are using cased version of BERT and tokenizer.

# %%
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

# %% [markdown]
# Loading a pre-trained BertTokenizer

# %%
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# %%
sample_txt = 'I want to learn how to do sentiment analysis using BERT and tokenizer.'

# %% [markdown]
# Encode_plus method of tokenizer adds special tokens like seperator[SEP], classifier [CLS], performs padding [PAD] so that BERT knows we are doing classification

# %%
encoding = tokenizer.encode_plus(
  sample_txt,
  max_length=32,
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',  # Return PyTorch tensors
  truncation = True
)

encoding.keys()

# %%


# %% [markdown]
# we are choosing maximum length of 400.

# %%
MAX_LEN = 500

# %% [markdown]
# Let's create pytorch dataset.

# %%
class MovieReviewDataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
      truncation = True
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

# %% [markdown]
# Let's split the dataset as 70% for training, and 15% each for validation and testing.

# %%
df_train, df_val = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)



# %% [markdown]
# Let's create data loaders.

# %%
def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = MovieReviewDataset(
    reviews=df.review.to_numpy(),
    targets=df.sentiment_score.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

# %%
BATCH_SIZE = 32

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# %%
data = next(iter(train_data_loader))
data.keys()

# %%
print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

# %% [markdown]
# ##Sentiment Classification with BERT and Hugging Face

# %% [markdown]
# We are using the BERT model and build the sentiment classifier on top of it. We then try to use the model on our sample text.

# %%
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

# %%
last_hidden_state, pooled_output = bert_model(
  input_ids=encoding['input_ids'], 
  attention_mask=encoding['attention_mask']
)

# %%
# %% [markdown]
# Let's create the classifier that uses BERT model. We will use a dropout layer for regularization and a fully connected layer for output.

# %%
class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(0.15)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    ip = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    ).last_hidden_state

    output = self.drop(ip)
    return self.out(output)

# %%
class_names = ['negative', 'positive']

# %%
len(class_names)

# %%
import torch
torch.cuda.empty_cache()


# %%
model = SentimentClassifier(len(class_names))
model = model.to(device)

# %%
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length

# %% [markdown]
# Let's apply softmax function to the outputs to get the predicted probabilities.

# %%
F.softmax(model(input_ids, attention_mask), dim=1)

# %% [markdown]
# %%
EPOCHS = 10
from tqdm import tqdm

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = F.cross_entropy

# %%
def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for d in tqdm(data_loader):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    target = d["targets"].to(device)
    targets = F.one_hot(d["targets"].to(device).long(), num_classes=2)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds.argmax(dim=1).squeeze() == target)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples , np.mean(losses)

# %% [markdown]
# Evaluate the model on a given data loader.

# %%
def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      target = d["targets"].to(device)
      targets = F.one_hot(d["targets"].to(device).long(), num_classes=2)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds.argmax(dim=1).squeeze() == target)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples , np.mean(losses)

# %%


history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):

  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)

  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,    
    loss_fn, 
    optimizer, 
    device, 
    scheduler, 
    len(df_train)
  )

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn, 
    device, 
    len(df_val)
  )

  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)

  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'best_model_state.bin')
    best_accuracy = val_acc

  test_acc, _ = eval_model(
    model,
    test_data_loader,
    loss_fn,
    device,
    len(df_test)
    )

  print("Test acc", test_acc.item())

# %% [markdown]
# Let's look at the training vs validation accuracy.

# %% [markdown]
# 

# 

# %% [markdown]
# The training accuracy starts to approach 100% after 9 epochs. You can try to fine-tune the parameters a bit more.

# %% [markdown]
# ##Evaluation

# %% [markdown]
# Let's calculate the accuracy on the test data.

# %%
test_acc, _ = eval_model(
  model,
  test_data_loader,
  loss_fn,
  device,
  len(df_test)
)

test_acc.item()

# %% [markdown]
# Our model seems to generalize well.

# %% [markdown]
# Let's look at the classification report

# %%
def get_predictions(model, data_loader):
  model = model.eval()
  
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []

  with torch.no_grad():
    for d in data_loader:

      texts = d["review_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = F.one_hot(d["targets"].to(device).long(), num_classes=2)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(probs)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values

# %%
y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_data_loader
)

# %%
print(classification_report(y_test, y_pred, target_names=class_names))

# %% [markdown]
# Confusion Matrix

# %%


cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)

# %% [markdown]
# This gives a good overview of the performance of our model.

# %% [markdown]
# Lets look at the example from the test data.

# %%
idx = 5

review_text = y_review_texts[idx]
true_sentiment = y_test[idx]
pred_df = pd.DataFrame({
  'class_names': class_names,
  'values': y_pred_probs[idx]
})

# %%
print("\n".join(wrap(review_text)))
print()
print(f'True sentiment: {class_names[true_sentiment]}')

# %% [markdown]
# Let's look at the confidence of each sentiment of our model

# %%
sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
plt.ylabel('sentiment')
plt.xlabel('probability')
plt.xlim([0, 1]);

# %%



