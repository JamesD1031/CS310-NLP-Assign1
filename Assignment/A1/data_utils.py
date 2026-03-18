from collections import Counter


class Vocab:
    def __init__(self, word_to_idx):
        self.word_to_idx = word_to_idx
        self.idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    def __call__(self, words):
        return [self.word_to_idx.get(word, 0) for word in words]

    def __len__(self):
        return len(self.word_to_idx)

    def __getitem__(self, word):
        return self.word_to_idx.get(word, 0)

    def __contains__(self, word):
        return word in self.word_to_idx


def build_vocab_from_iter(iterator, specials=("<unk>",), min_freq=1):
    word_freq = Counter()
    for tokens in iterator:
        word_freq.update(tokens)

    word_to_idx = {}
    for token in specials:
        if token not in word_to_idx:
            word_to_idx[token] = len(word_to_idx)

    for word, freq in sorted(word_freq.items(), key=lambda item: item[1], reverse=True):
        if freq >= min_freq and word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

    return Vocab(word_to_idx)
