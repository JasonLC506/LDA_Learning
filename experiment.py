import re
import numpy as np
from scipy.sparse import csr_matrix
import cPickle

from emotion_topic_model import ETM
from post_content_extract import post_content_extract
from data_processing import Corpus, dataProcessing

filename = "data/CNN"
pattern = re.compile(r'^5550296508_')
posts = post_content_extract(filename, pattern)

emotions = cPickle.load(open("data/CNN_post_emotion", "r"))
# cp = Corpus(posts.values())
# cp.preprocessing()
#
# dataW = cp.matrix
# dataToken = cp.corpus
# # Ndocs = 1000
# # V = 10000
# E = 6
# # word_dist = np.arange(V)*1.0/V/(V-1)*2.0
# # dataW = csr_matrix(np.random.multinomial(10, word_dist, Ndocs))          # synthesize corpus data
# dataE = np.random.multinomial(10, np.arange(E).astype(np.float32)/float(E*(E-1)*0.5), cp.Ndocs) / 10.0       # synthesize emotion data

cp, dataE, id_map = dataProcessing(posts, emotions)

dataW = cp.matrix
model = ETM(K=10)
model.fit(dataE,dataW)