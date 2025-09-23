import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
import numpy as np
import random

SEED = 2024
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT = Field(tokenize='spacy',
             tokenizer_language='en_core_web_sm',
             include_lengths=True)

LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)

train_data, test_data = TabularDataset.splits(
    path='/content/data',
    train='train.csv',
    test='test.csv',
    format='csv',
    fields=[('text', TEXT), ('label', LABEL)],
    skip_header=True
)

MAX_VOCAB_SIZE = 25_000
TEXT.build_vocab(train_data,
                 max_size=MAX_VOCAB_SIZE,
                 vectors="glove.6B.100d",
                 unk_init=torch.Tensor.normal_)

BATCH_SIZE = 64
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
    device=device
)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, pad_idx, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.to('cpu'), enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model = SentimentLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, PAD_IDX).to(device)

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss().to(device)

def binary_accuracy(preds, y):
    rounded = torch.round(torch.sigmoid(preds))
    return (rounded == y).float().mean()

def train_fn(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss, epoch_acc = 0.0, 0.0
    for batch in iterator:
        text, text_lengths = batch.text
        labels = batch.label.to(device)

        optimizer.zero_grad()
        preds = model(text, text_lengths).squeeze(1)
        loss = criterion(preds, labels)
        acc = binary_accuracy(preds, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_fn(model, iterator, criterion):
    model.eval()
    epoch_loss, epoch_acc = 0.0, 0.0
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            labels = batch.label.to(device)
            preds = model(text, text_lengths).squeeze(1)
            loss = criterion(preds, labels)
            acc = binary_accuracy(preds, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def predict_sentiment(model, sentence):
    model.eval()
    tokenized = TEXT.preprocess(sentence)
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).unsqueeze(1).to(device)
    length_tensor = torch.LongTensor(length)
    with torch.no_grad():
        score = torch.sigmoid(model(tensor, length_tensor))
    return score.item()

# ============== 训练示例 ==============
EPOCHS = 3
for epoch in range(EPOCHS):
    train_loss, train_acc = train_fn(model, train_iterator, optimizer, criterion)
    test_loss, test_acc = evaluate_fn(model, test_iterator, criterion)
    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

# ============== 简单预测 ==============
positive_review = "This movie was fantastic! I really enjoyed it."
negative_review = "The film was terrible and boring."
print(f"Positive review score: {predict_sentiment(model, positive_review):.4f}")
print(f"Negative review score: {predict_sentiment(model, negative_review):.4f}") 