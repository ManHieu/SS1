import numpy as np
from nltk import line_tokenize
from collections import defaultdict
from read_data import ReadFile
class VocabGenerator:
    def __init__(self, file):
        self.file = file

    def generate_vocabulary(self):
        self.min_df = 10
        def compute_idf(df, corpus_size):
            assert df > 0
            return np.log(corpus_size / df) 
        with open(self.file) as file:
            data = file.read()
            lines = line_tokenize(data)
            corpus_size = len(lines)
            doc_count = defaultdict(int)
            for line in lines:
                components = line.split("_____")
                text = components[-1]
                features = list(set(text.split()))
                for w in features:
                    doc_count[w] += 1
            # words = list(doc_count.keys())
            # idfs = []
            # for word in words:
            #     if doc_count[word] > self.min_df:
            #         idf = compute_idf(doc_count[word], corpus_size)
            #         idfs.append(idf)
            #     else: 
            #         words.remove(word)
            # vocab = zip(words, idfs)
            vocab = [(word, compute_idf(doc_count[word], corpus_size)) for word in list(doc_count.keys())
                     if doc_count[word] > self.min_df]
            feature_idfs = []
            for (feature, idf) in vocab:
                feature_idfs.append(feature + ":" + str(idf))
            with open('data\\vocab.txt', 'w') as file:
                file.write("\n".join(feature_idfs))

vc = VocabGenerator('data\\20k_vb.txt')
vc.generate_vocabulary()