import glob
import random
import re
from typing import Optional, List,AnyStr
from transformers import AutoTokenizer
WINDOW_SIZE = 5

def positive_samples(tokenized,word2idx)->Optional[List[str]]:
    positives = []
    for sentence in tokenized:
        indexed = [word2idx[w] for w in sentence.split()]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - WINDOW_SIZE), min(len(indexed), i + WINDOW_SIZE + 1)):
                if i != j:
                    positives.append((center, indexed[j]))
    return positives
def negative_sampling(pos_word,vocab_size ,num_neg=5):
    negatives = []
    while len(negatives) < num_neg:
        neg = random.randint(0, vocab_size - 1)
        if neg != pos_word:
            negatives.append(neg)
    return negatives
def tokenize_paragraphs(
):
    # tokenize
    txt_files = glob.glob('*.txt')
    tokenized = []
    for file_path in txt_files:
        with open(file_path, "r") as f:
            text = f.read()
        tokenized.extend([
            s.strip()
            for s in re.split(r'[.!?,]+', text)
            if s.strip()
        ])

    # flatten
    words = [w  for sentence in tokenized for w in sentence.split()]
    return words,tokenized
