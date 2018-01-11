import numpy as np
from functions import multinomial, probNormalize

class dataDUE(object):
    def __init__(self):
        self.E = 0                                  # dimension of emotion
        self.U = 0                                  # number of users
        self.Md = None                              # count of document-level total emoticons List[D]
        self.D = 0                                  # number of documents

    def generate(self, batch_size=1, random_shuffle=False):
        # depends on inputs, yield [document_id, [[reader_id],[emoticon]]] #
        ### example ###
        for d in range(self.D):
            yield d, [np.arange(self.Md[d]), multinomial(probNormalize(np.random.random(self.E)), self.Md[d])]
        ###############
    pass