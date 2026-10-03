"""Model definitions and a common training routine for all experiments."""
import gzip
import os
import random

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402
from tensorflow.keras import layers, Model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOVE = os.environ.get("GLOVE_PATH", os.path.join(os.path.dirname(ROOT), "emb/glove100.gz"))

# ------------------------------------------------------------------ hyper-parameters
HP = dict(max_vocab=8000, max_len=100, embed_dim=100, batch_size=32, lr=1e-3,
          max_epochs=40, patience=8, optimizer="Adam")
CNN_HP = dict(kernels=(3, 4, 5), filters=128, filters2=64, spatial_dropout=0.3,
              dense=(128, 64), dropout=(0.5, 0.4))
LSTM_HP = dict(units=(64, 32), spatial_dropout=0.2, dense=32, dropout=(0.4, 0.3))


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


# ------------------------------------------------------------------ lexicon baseline
def vader_scores(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    s = SentimentIntensityAnalyzer()
    return np.array([s.polarity_scores(t)["compound"] for t in texts])


def vader_labels(compound, lo=-0.05, hi=0.05):
    return np.where(compound >= hi, 2, np.where(compound <= lo, 0, 1))


def tune_vader(val_compound, y_val):
    """Select the neutral band on validation data only (grid search)."""
    best = (-1, -0.05, 0.05)
    for lo in np.arange(-0.9, 0.01, 0.05):
        for hi in np.arange(0.0, 0.96, 0.05):
            if hi <= lo:
                continue
            f = f1_score(y_val, vader_labels(val_compound, lo, hi), average="macro")
            if f > best[0]:
                best = (f, round(float(lo), 2), round(float(hi), 2))
    return best


# ------------------------------------------------------------------ classical ML
def tfidf_model(kind, C, ngram, class_weight):
    vec = TfidfVectorizer(ngram_range=ngram, min_df=1, sublinear_tf=True)
    clf = (LogisticRegression(C=C, max_iter=5000, class_weight=class_weight)
           if kind == "lr" else LinearSVC(C=C, class_weight=class_weight))
    return vec, clf


# ------------------------------------------------------------------ embeddings
_GLOVE_CACHE = {}


def glove_matrix(vocab, dim=100):
    if not _GLOVE_CACHE:
        with gzip.open(GLOVE, "rt", encoding="utf8") as f:
            first = f.readline().split()
            if len(first) != 2:  # file without header line
                _GLOVE_CACHE[first[0]] = np.asarray(first[1:], dtype="float32")
            for line in f:
                p = line.rstrip().split(" ")
                _GLOVE_CACHE[p[0]] = np.asarray(p[1:], dtype="float32")
    rng = np.random.default_rng(0)
    M = rng.normal(0, 0.1, (len(vocab), dim)).astype("float32")
    M[0] = 0
    hit = 0
    for i, w in enumerate(vocab.itos[2:], start=2):
        if w in _GLOVE_CACHE:
            M[i] = _GLOVE_CACHE[w]; hit += 1
    return M, hit / max(1, len(vocab) - 2)


def _embedding(vocab_size, emb_matrix):
    if emb_matrix is None:
        return layers.Embedding(vocab_size, HP["embed_dim"])
    return layers.Embedding(vocab_size, emb_matrix.shape[1],
                            embeddings_initializer=tf.keras.initializers.Constant(emb_matrix),
                            trainable=True)


# ------------------------------------------------------------------ neural models
def build_cnn(vocab_size, emb_matrix=None, kernels=None, n_classes=3):
    kernels = kernels or CNN_HP["kernels"]
    inp = layers.Input((HP["max_len"],), dtype="int32")
    x = _embedding(vocab_size, emb_matrix)(inp)
    x = layers.SpatialDropout1D(CNN_HP["spatial_dropout"])(x)
    pools = []
    for k in kernels:
        c = layers.Conv1D(CNN_HP["filters"], k, activation="relu", padding="same")(x)
        c = layers.Conv1D(CNN_HP["filters2"], k, activation="relu", padding="same")(c)
        pools.append(layers.GlobalMaxPooling1D()(c))
    h = layers.Concatenate()(pools) if len(pools) > 1 else pools[0]
    for units, dr in zip(CNN_HP["dense"], CNN_HP["dropout"]):
        h = layers.Dense(units, activation="relu")(h)
        h = layers.Dropout(dr)(h)
    out = layers.Dense(n_classes, activation="softmax")(h)
    return Model(inp, out, name="MultiKernelCNN")


def build_bilstm(vocab_size, emb_matrix=None, n_classes=3):
    inp = layers.Input((HP["max_len"],), dtype="int32")
    x = _embedding(vocab_size, emb_matrix)(inp)
    x = layers.SpatialDropout1D(LSTM_HP["spatial_dropout"])(x)
    x = layers.Bidirectional(layers.LSTM(LSTM_HP["units"][0], return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(LSTM_HP["units"][1]))(x)
    x = layers.Dropout(LSTM_HP["dropout"][0])(x)
    x = layers.Dense(LSTM_HP["dense"], activation="relu")(x)
    x = layers.Dropout(LSTM_HP["dropout"][1])(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inp, out, name="BiLSTM")


class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    """Model selection on validation macro-F1 (not accuracy, not the test set)."""

    def __init__(self, Xv, yv, patience):
        super().__init__()
        self.Xv, self.yv, self.patience = Xv, yv, patience
        self.best, self.best_w, self.wait, self.best_epoch, self.hist = -1, None, 0, 0, []

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.Xv, verbose=0).argmax(1)
        f = f1_score(self.yv, p, average="macro")
        self.hist.append(dict(epoch=epoch + 1, loss=float(logs["loss"]), acc=float(logs["accuracy"]),
                              val_loss=float(logs["val_loss"]), val_acc=float(logs["val_accuracy"]),
                              val_macro_f1=float(f)))
        if f > self.best:
            self.best, self.best_w, self.wait, self.best_epoch = f, self.model.get_weights(), 0, epoch + 1
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_w is not None:
            self.model.set_weights(self.best_w)


def train_neural(arch, Xtr, ytr, Xv, yv, vocab_size, seed, emb_matrix=None,
                 class_weighted=True, kernels=None):
    set_seed(seed)
    tf.keras.backend.clear_session()
    m = (build_cnn(vocab_size, emb_matrix, kernels) if arch == "cnn"
         else build_bilstm(vocab_size, emb_matrix))
    m.compile(optimizer=tf.keras.optimizers.Adam(HP["lr"]),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cw = None
    if class_weighted:
        w = compute_class_weight("balanced", classes=np.arange(3), y=ytr)
        cw = {i: float(v) for i, v in enumerate(w)}
    cb = MacroF1Checkpoint(Xv, yv, HP["patience"])
    m.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=HP["max_epochs"], batch_size=HP["batch_size"],
          class_weight=cw, callbacks=[cb], verbose=0, shuffle=True)
    return m, cb
