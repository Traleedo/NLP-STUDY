import torch
from torch import nn, optim

from Embedding.utils import  positive_samples, negative_sampling, tokenize_paragraphs

words,tokenized = tokenize_paragraphs()
vocab = list(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.input_emb = nn.Embedding(vocab_size, embedding_dim)
        self.output_emb = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center, context, negatives):
        center_vec = self.input_emb(center)          # (1, D)
        context_vec = self.output_emb(context)        # (1, D)
        neg_vecs = self.output_emb(negatives)         # (K, D)

        pos_score = torch.sum(center_vec * context_vec, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score))

        neg_score = torch.matmul(neg_vecs, center_vec.T).squeeze()
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score)))

        return -(pos_loss + neg_loss)
embedding_dim = 100
vocab_size = len(vocab)
model = Word2Vec(vocab_size, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
positives = positive_samples(tokenized,word2idx)
epochs = 200

for epoch in range(epochs):
    total_loss = 0
    for center, context in positives:
        center = torch.tensor([center])
        context = torch.tensor([context])
        negatives = torch.tensor(negative_sampling(context.item(),vocab_size=vocab_size))

        loss = model(center, context, negatives)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

