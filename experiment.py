import re
import numpy as np
from scipy.sparse import csr_matrix
import cPickle
from matplotlib import pyplot as plt

from emotion_topic_model import ETM
from post_content_extract import post_content_extract
from data_processing import Corpus, dataProcessing

EMOTICON_LIST = ["LIKE", "LOVE", "SAD", "WOW", "HAHA", "ANGRY"]

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

print "######### data statistics ##########"
print "Ndocs", cp.Ndocs
print "Ntokens", cp.Ntokens
print "V", cp.matrix.shape[1]
print "E", dataE.shape[1]

dataW = cp.matrix
model = ETM(K=20)
# model.fit(dataE,dataW)
model._restoreCheckPoint(filename="ckpt/ETM_K20")
theta, phi = model.theta, model.phi
# find top words for each topic #
n_top_words = 8
for i, topic_dist in enumerate(phi.tolist()):
    topic_words = np.array(cp.words)[np.argsort(topic_dist)][:-n_top_words:-1]
    print "Topic {}: {}".format(i, ','.join(topic_words))
for i in range(6):
    plt.plot(theta[i],label="e: %s" % EMOTICON_LIST[i])
plt.legend()
plt.show()