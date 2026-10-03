"""
Text preprocessing for the İzmir tourist-review study.

Design principles
- Original review text is never modified; processed text is a derived column.
- Negation words are preserved (they carry sentiment).
- Typographic apostrophes (’ ‘ ʼ) are normalised BEFORE contraction expansion,
  so "don’t" becomes "do not" (the v5 pipeline missed these).
- Vocabulary is fitted on the training partition only; unseen tokens -> <UNK>.
"""
import re
import unicodedata
from collections import Counter

import numpy as np

LABELS = ["Negative", "Neutral", "Positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

NEGATIONS = {
    "not", "no", "never", "nothing", "nobody", "neither", "nor", "none",
    "nowhere", "cannot", "without",
}

STOPWORDS = {
    "a", "an", "the", "is", "it", "was", "this", "that", "these", "those",
    "for", "and", "or", "of", "in", "to", "i", "we", "our", "my", "me", "us",
    "had", "have", "has", "be", "been", "are", "were", "would", "could",
    "with", "on", "at", "by", "from", "up", "go", "get", "got", "do", "did",
    "does", "will", "shall", "may", "might", "just", "also", "than", "then",
    "when", "where", "which", "who", "what", "how", "there", "their", "they",
    "them", "its", "his", "her", "your", "each", "as", "if", "about", "he",
    "she", "you", "am", "into", "out", "s", "t",
}  # 'but', 'very', 'so', 'only', 'even', 'too' deliberately kept (contrast/intensity cues)

CONTRACTIONS = {
    "won't": "will not", "can't": "can not", "shan't": "shall not",
    "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
    "'d": " would", "'m": " am", "'s": "",
}

_APOS = re.compile(r"[‘’ʼ`´]")
_URL = re.compile(r"http\S+|www\.\S+")
_NONALPHA = re.compile(r"[^a-zçğıöşü\s]")
_WS = re.compile(r"\s+")


def clean_text(text, keep_negations=True, normalise_apostrophes=True,
               remove_stopwords=True):
    """Return a whitespace-joined token string."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace(" ", " ").replace(" ", " ")
    if normalise_apostrophes:
        t = _APOS.sub("'", t)
    t = t.lower()
    t = _URL.sub(" ", t)
    for k, v in CONTRACTIONS.items():
        t = t.replace(k, v)
    t = _NONALPHA.sub(" ", t)
    toks = [w for w in _WS.split(t) if len(w) >= 2 or w in NEGATIONS]
    out = []
    for w in toks:
        if w in NEGATIONS:
            if keep_negations:
                out.append(w)
            continue
        if remove_stopwords and w in STOPWORDS:
            continue
        out.append(w)
    return " ".join(out)


class Vocab:
    PAD, UNK = 0, 1

    def __init__(self, max_size=8000, min_freq=1):
        self.max_size, self.min_freq = max_size, min_freq
        self.itos = ["<PAD>", "<UNK>"]
        self.stoi = {}

    def fit(self, texts):
        c = Counter(w for t in texts for w in t.split())
        for w, f in c.most_common(self.max_size - 2):
            if f >= self.min_freq:
                self.itos.append(w)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        return self

    def encode(self, texts, max_len=150):
        X = np.zeros((len(texts), max_len), dtype="int32")
        for i, t in enumerate(texts):
            ids = [self.stoi.get(w, self.UNK) for w in t.split()][:max_len]
            X[i, :len(ids)] = ids
        return X

    def __len__(self):
        return len(self.itos)
