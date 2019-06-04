import os
from collections import Counter
from typing import Tuple
import json

import wordsegment as ws

json_cache = os.path.join(os.path.dirname(__file__), "ws_cache.json")


# Generate N-Gram pairs
def pairs(iterable):
    iterator = iter(iterable)
    values = [next(iterator)]
    for value in iterator:
        values.append(value)
        yield ' '.join(values)
        del values[0]


# The segmenter requires a clean up function. Normaly, it strips ponctuation
# We overwrite the clean-up by not cleaning up
def identity(value):
    return value


# Integer classes for Word Content (x) and Word Boundary (S)
classes = {
    "x": 0,
    "S": 1
}


def match(segmenter: ws.Segmenter, x: str, truth: str) -> Tuple[Tuple[int, int]]:
    """ Given a segmenter, compute the prediction and returns the integer class mask for truth and prediction

    """
    pred = segmenter.segment(x)

    truth_mask = [
        "x" if ngram[1] != " " else "S"
        for ngram in zip(*[truth[i:] for i in range(2)])
        if ngram[0] != " "
    ] + ["S"]
    pred_mask = [
        char
        for tok in pred
        for char in ["x"] * (len(tok) - 1) + ["S"]
    ]

    return tuple(
        (classes[t], classes[p])
        for t, p in zip(truth_mask, pred_mask)
    )

# Load datasets
text = []
with open(os.path.join(os.path.dirname(__file__), "..", "datasets", "fro", "train.tsv")) as input_text:
    for line in input_text.readlines():
        inp, truth = tuple(line.strip().lower().split("\t"))
        text.extend(truth.split())

# Change the segmenter information
segmenter = ws.Segmenter()

# Generate the dictionary and set-up the segmenter
segmenter.unigrams.clear()
segmenter.unigrams.update(Counter(text))
segmenter.bigrams.clear()
segmenter.bigrams.update(Counter(pairs(text)))
segmenter.words = list(set(text))


segmenter.total = float(sum(segmenter.unigrams.values()))
segmenter.clean = identity
segmenter.limit = max(list(map(len, text)))

# Load the test corpus
test = []
padding = 0
with open(os.path.join(os.path.dirname(__file__), "..", "datasets", "fro", "test.tsv")) as input_text:
    for line in input_text.readlines():
        inp, truth = tuple(line.strip().lower().split("\t"))
        test.extend([(inp, truth)])
        padding = max(len(truth), padding)

# If compute is True, recompute the data without using the cache
compute = False
if compute:
    results = list([
        res
        for x, y in test
        for res in match(segmenter, x, y)
    ])
    truth, pred = zip(*results)
    results = {
        "truth": truth,
        "pred": pred
    }
    with open(json_cache, "w") as f:
        json.dump(results, f)
else:
    with open(json_cache) as f:
        results = json.load(f)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
print("Accuracy:", accuracy_score(results["truth"], results["pred"]))
print("Precision, Recall, FScore, Support", precision_recall_fscore_support(results["truth"], results["pred"], average="macro"))
print("Matrix", confusion_matrix(results["truth"], results["pred"]))

with open(os.path.join(os.path.dirname(__file__), "..", "article", "input.txt")) as input_text:
    print("Completely unknown text", " ".join(segmenter.segment(input_text.read())))
