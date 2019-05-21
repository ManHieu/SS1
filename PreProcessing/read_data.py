import os
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer

class ReadFile:
    def __init__(self, path = r'D:\Documents\Experience\ML\Deploy\Implement\PreProcessing\data'):
       self.stopwords = set(stopwords.words('english'))
       self.port = PorterStemmer()
       self.path = path
# Đọc file và trả về đoạn văn bản đã được làm sạch
    def readdata(self, filename, count = 0):
        filename = self.path + "\\" + filename
        texts = []
        for folder in os.listdir(filename):
            fname = [filename, folder]
            fname = "\\".join(fname)
            for f in os.listdir(fname):
                path = [fname, f]
                path = "\\".join(path)
                with open(path) as file:
                    text = file.read()
                    text = text.lower()
                    words = word_tokenize(text)
                    words = [w for w in words if w not in self.stopwords and len(w) < 12 and len(w) >2 and w.isalpha()]
                    words = [self.port.stem(w) for w in words]
                    text = " ".join(words)
                    data = [str(count), text]
                    text = "_____".join(data)
                    texts.append(text)
            count += 1
        return texts

# Lưu đoạn text vào tập txt
    def savedata(self, texts, filesave):
        link = r'D:\Documents\Experience\ML\Deploy\Implement\PreProcessing\data' + "\\" + filesave
        with open(link, 'w') as file:
            data = "\n".join(texts)
            file.write(data)

rf = ReadFile()
texts = rf.readdata('20_newsgroups')
print('start..')
rf.savedata(texts, '20k_vb.txt')
print('done')