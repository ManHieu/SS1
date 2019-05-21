from nltk import line_tokenize, word_tokenize
import numpy as np
from read_data import ReadFile
class TfidfVectorizer:
    def __init__(self):
        print("start!")

    def get_tfidf(self, filename, file_tfidf):
        with open(r'D:\DS_lab\Project\PreProcessing\data\vocab.txt') as file:
            word_idfs = [(line.split(":")[0], float(line.split(":")[1])) for line in line_tokenize(file.read())]
        idfs = dict(word_idfs)
        IDwords = dict((word, index) for index, (word, idf) in enumerate(word_idfs))
        data = []
        with open(filename) as file:
            documents = [(line.split("_____")[0], line.split("_____")[1]) for line in line_tokenize(file.read())]
            for document in documents:
                words = [w for w in document[1].split() if w in list(idfs.keys())]
                set_of_words = list(set(words))
                sum_words = len(words)
                word_tfidfs = []
                sum_squares = 0
                for word in set_of_words:
                    tfidf = idfs[word] * words.count(word) / sum_words
                    sum_squares += tfidf ** 2
                    word_tfidfs.append((IDwords[word], tfidf))
                word_tfidfs_normalize = [(str(index) + ":" + str(tfidf / np.sqrt(sum_squares))) for index, tfidf in word_tfidfs]
                sparse_data = " ".join(word_tfidfs_normalize)
                data.append("_____".join([document[0], sparse_data]))
        with open(file_tfidf, 'w') as file:
            file.write("\n".join(data))

# rf = ReadFile()
# texts = rf.readdata('20test')
# rf.savedata(texts, 'test.txt')
print('start TFIDF ...')
tf = TfidfVectorizer()
tf.get_tfidf('data\\20k_vb.txt', 'data\\tfidf.txt')
print('done ... ')