from collections import defaultdict
from os.path import split
from threading import enumerate


class Member:
    def __init__(self, r_d, label=None, doc_id=None):
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id


class Cluster:
    def __init__(self):
        self._centroid = None
        self._members = []
    
    def reset_members(self):
        self._members = []

    def add_member(self, member):
        self._members.append(member)

class Kmeans:
    def __init__(self, num_cluster):
        self._num_clusters = num_cluster
        self._E = []
        self._S = 0
        self._clusters = [Cluster() for _ in range(self._num_clusters)]

def load_data(self, data_path):
    def sparse_to_dense(sparse_r_d, vocab_size):
        pass
    
    with open(data_path) as f:
        d_lines = f.read().splitlines()
    
    with open('..\data\tfidf.txt') as f:
        vocab_size = len(f.read().splitlines())

    self._data = []
    self._label_count = defaultdict(int)
    for data_id, d in enumerate(d_lines):
        feature = d.split('_____')
        label, doc_id = int(feature[0]), int(feature[1])
        label_count += 1
        r_d = sparse_to_dense(sparse_r_d=feature[2], vocab_size=vocab_size)

        self._data.append(Member(r_d, label, doc_id))
