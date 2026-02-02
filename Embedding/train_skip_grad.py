import torch
from torch import nn, optim
from tqdm import tqdm
from Embedding.utils import positive_samples, tokenize_paragraphs, negative_sampling_batch
from torch.utils.data import Dataset, DataLoader
words,tokenized = tokenize_paragraphs()
vocab = list(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
class Word2VecDataset(Dataset):
    def __init__(self, positive_pairs):
        self.data = positive_pairs  # [(center, context), ...]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center, context = self.data[idx]
        return torch.tensor(center), torch.tensor(context)

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.input_emb = nn.Embedding(vocab_size, embedding_dim)
        self.output_emb = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center, context, negatives):
        """
        center:    (B,)
        context:   (B,)
        negatives: (B, K)
        """
        center_vec = self.input_emb(center)          # (B, D)
        context_vec = self.output_emb(context)       # (B, D)
        neg_vecs = self.output_emb(negatives)        # (B, K, D)

        # 正样本
        pos_score = torch.sum(center_vec * context_vec, dim=1)  # (B,)
        pos_loss = torch.log(torch.sigmoid(pos_score))          # (B,)

        # 负样本
        neg_score = torch.bmm(
            neg_vecs, center_vec.unsqueeze(2)
        ).squeeze(2)                                              # (B, K)

        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score)), dim=1)  # (B,)

        loss = -(pos_loss + neg_loss)   # (B,)
        return loss.mean()
embedding_dim = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = len(vocab)
batch_size = 128
epochs = 200
K = 5  # negative samples
positives = positive_samples(tokenized,word2idx)
dataset = Word2VecDataset(positives)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

model = Word2Vec(vocab_size, embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
best_loss = float('inf')
for epoch in range(epochs):
    total_loss = 0

    for centers, contexts in tqdm(dataloader, desc=f"{epoch+1}/{epochs}"):
        centers = centers.to(device)
        contexts = contexts.to(device)
        negatives = negative_sampling_batch(
            contexts, vocab_size, K
        ).to(device)

        loss = model(centers, contexts, negatives)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    if(total_loss/len(dataloader) < best_loss):
        best_loss = total_loss/len(dataloader)
        torch.save(model.state_dict(), "model/model_skip_grad.pth")
