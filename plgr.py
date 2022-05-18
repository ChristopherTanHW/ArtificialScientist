import string
import itertools
import random

def compute_labels(limit=5):
    all_labels = string.ascii_lowercase[:limit]
    train = []
    dev = []
    perms = list(itertools.permutations(all_labels, 3))
    perms = sorted(list(perms))
    perms.sort()
    random.Random(0).shuffle(perms)
    n = len(perms) // 2
    train = perms[:n]
    dev = perms[n:]
    return train, dev

def compute_labels_med(limit=20):
    all_labels = string.ascii_lowercase[:limit]
    train = []
    dev = []
    train_vocab = set()
    for i in range(0, len(all_labels)-5, 2):
        a, b, c, d, e = all_labels[i:i+5]
        for trip in itertools.permutations([a, b, d]):
            train.append(trip)
            train_vocab |= set(trip)
        for trip in itertools.permutations([b, c, e]):
            dev.append(trip)
    dev = [(a, b, c) for a, b, c in dev if a in train_vocab and b in train_vocab and c in train_vocab]
    return train, dev

t,d =compute_labels()
t_med, d_med = compute_labels_med()
print(t)
print('\n')
print(d)
print('\n')
print(t_med)
print('\n')
print(d_med)