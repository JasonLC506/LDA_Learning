"""
Implement
{author = {Ding, Zhuoye and Qiu, Xipeng and Zhang, Qi and Huang, Xuanjing},
journal = {IJCAI International Joint Conference on Artificial Intelligence},
title = {{Learning topical translation model for microblog hashtag suggestion}},
year = {2013}
} --- [1]
"""
import numpy as np
import cPickle
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm

from functions import probNormalize, multinomial

class TTM(object):
    def __init__(self, K):
        """
        :param K: # topics
        """
        # model hyperparameters #
        self.alpha = 0.1                                        # topic distribution prior
        self.beta = 0.01                                        # topic-word distribution prior
        self.gamma = 0.1                                        # topic-emotion distribution prior
        self.delta = 0.01                                       # background-vs-topic distribution prior

        # data dimensions #
        self.E = 0                                              # number of emotions
        self.K = K                                              # number of topics
        self.D = 0                                              # number of documents
        self.Nd = []                                            # number of words of documents (varying over docs)
        self.Md = []                                            # number of emotions of documents (varying over docs)
        self.V = 0                                              # size of vocabulary

        # model latent variables #
        self.theta = None                                       # document-level topic distribution [self.D, self.K]
        self.pi = None                                          # background-vs-topic distribution
        self.eta = None                                         # topic-emotion distribution [self.K, self.E]
        self.phiB = None                                        # background word distribution [self.V]
        self.phiT = None                                        # topic-word distribution [self.K, self.V]
        self.z = None                                           # document-level topic [self.D]
        self.y = None                                           # word-level background-vs-topic indicator "[self.D, self.Nd]"

        # intermediate variables for fitting #
        self.YI = None                                          # count of background-vs-topic indicator over corpus [2]
        self.TE = None                                          # count of topic-emotion cooccurrences [self.K, self.E]
        self.Y0V = None                                         # count of background word [self.V]
        self.Y1TV = None                                        # count of topic-word cooccurrences [self.K, self.V]

        # save & restore #
        self.checkpoint_file = "ckpt/TTM"

    def fit(self, dataE, dataW, corpus=None, alpha=0.1, beta=0.01, gamma=0.1, delta=0.01, max_iter=500, resume=None):
        """
        Collapsed Gibbs sampler
        :param dataE: Emotion counts of each document     np.ndarray([self.D, self.E])
        :param dataW: Indexed corpus                      np.ndarray([self.D, self.V]) scipy.sparse.csr_matrix
        """
        self._setHyperparameters(alpha=alpha, beta=beta, gamma=gamma, delta=delta)
        if corpus is None:
            dataToken = self._matrix2corpus(dataW=dataW)
        else:
            dataToken = corpus

        self._setDataDimension(dataE=dataE, dataW=dataW, dataToken=dataToken)
        if resume is None:
            self._initialize(dataE=dataE, dataW=dataW, dataToken=dataToken)
        else:
            self._restoreCheckPoint(filename=resume)

        ppl_initial = self._ppl(dataE=dataE, dataW=dataW, dataToken=dataToken)
        print "before training, ppl: %s" % str(ppl_initial)

        ## Gibbs Sampling ##
        for epoch in range(max_iter):
            self._GibbsSamplingLocal(dataE=dataE, dataW=dataW, dataToken=dataToken, epoch=epoch)
            self._estimateGlobal(dataE)
            ppl = self._ppl(dataE=dataE, dataW=dataW, dataToken=dataToken)
            print "epoch: %d, ppl: %s" % (epoch, str(ppl))
            self._saveCheckPoint(epoch, ppl)

    def _setHyperparameters(self, alpha, beta, gamma, delta):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    """ copied from ETM"""
    def _matrix2corpus(self, dataW):
        start = datetime.now()

        dataToken = []
        for d in range(dataW.shape[0]):
            docW = dataW.getrow(d)
            docToken = []
            for w_id in docW.indices:
                w_freq = docW[0, w_id]
                for i in range(w_freq):
                    docToken.append(w_id)
            dataToken.append(docToken)

        duration = datetime.now() - start
        print "_matrix2corpus() takes %fs" % duration.total_seconds()
        return dataToken

    def _setDataDimension(self, dataE, dataW, dataToken):
        self.E = dataE.shape[1]
        self.D = dataE.shape[0]
        self.Nd = map(lambda x: len(x), dataToken)
        self.V = dataW.shape[1]
        self.Md = np.sum(dataE, axis=1).tolist()

    def _initialize(self, dataE, dataW, dataToken):
        start = datetime.now()

        self.theta = probNormalize(np.random.random([self.D, self.K]))
        self.pi = probNormalize(np.random.random([2]))
        self.eta = probNormalize(np.random.random([self.K, self.E]))
        self.phiB = probNormalize(np.random.random([self.V]))
        self.phiT = probNormalize(np.random.random([self.K, self.V]))
        self.z = np.zeros([self.D], dtype=np.int8)
        self.y = []
        for d in range(self.D):
            self.z[d] = multinomial(self.theta[d])
            Nd = self.Nd[d]
            self.y.append(multinomial(self.pi, Nd))

        self.YI = np.zeros([2], dtype=np.int32)
        self.Y0V = np.zeros([self.V], dtype=np.int32)
        self.Y1TV = np.zeros([self.K, self.V], dtype=np.int32)
        self.TE = np.zeros([self.K, self.E], dtype=np.int32)
        for d in range(self.D):
            self.TE[self.z[d], :] += dataE[d]
            docToken = dataToken[d]
            doc_z = self.z[d]
            doc_y = self.y[d]
            for n in range(self.Nd[d]):
                w = docToken[n]
                w_y = doc_y[n]
                self.YI[w_y] += 1
                if w_y == 0:
                    self.Y0V[w] += 1
                elif w_y == 1:
                    self.Y1TV[doc_z, w] += 1
                else:
                    print w_y, type(w_y)
                    raise ValueError("w_y type error")

        duration = datetime.now() - start
        print "_initialize() takes %fs" % duration.total_seconds()

    def _GibbsSamplingLocal(self, dataE, dataW, dataToken, epoch):
        """
        Gibbs sampling word-level background-vs-topic and document-level topic
        """
        pbar = tqdm(range(self.D),
                    total = self.D,
                    desc='({0:^3})'.format(epoch))
        for d in pbar:                                 # sequentially sampling
            doc_Nd = self.Nd[d]
            docE = dataE[d]
            docToken = dataToken[d]
            doc_z = self.z[d]

            # intermediate parameters calculation #
            doc_Y1T = np.sum(self.Y1TV, axis=1)

            for n in range(doc_Nd):
                w = docToken[n]
                w_y = self.y[d][n]

                ## sampling for y ##
                # calculate leave-one out statistics #
                YI_no_dn_y, Y0V_no_dn_y, Y1TV_no_dn_y = self.YI, self.Y0V, self.Y1TV
                doc_Y1T_no_dn_y = doc_Y1T

                YI_no_dn_y[w_y] += -1
                if w_y == 0:
                    Y0V_no_dn_y[w] += -1
                else:
                    Y1TV_no_dn_y[doc_z, w] += -1
                    doc_Y1T_no_dn_y[doc_z] += -1
                # conditional probability #
                prob_w_y = np.zeros([2],dtype=np.float32)
                prob_w_y[0] = (self.delta + YI_no_dn_y[0]) * (self.beta + Y0V_no_dn_y[w]) / \
                              (self.V * self.beta + YI_no_dn_y[0])
                prob_w_y[1] = (self.delta + YI_no_dn_y[1]) * (self.beta + Y1TV_no_dn_y[doc_z, w]) / \
                              (self.V * self.beta + doc_Y1T_no_dn_y[doc_z])
                prob_w_y = probNormalize(prob_w_y)
                # new sampled result #
                w_y_new = multinomial(prob_w_y)
                # update #
                self.y[d][n] = w_y_new
                YI_no_dn_y[w_y_new] += 1
                if w_y_new == 0:
                    Y0V_no_dn_y[w] += 1
                else:
                    Y1TV_no_dn_y[doc_z, w] += 1
                    doc_Y1T_no_dn_y[doc_z] += 1
                self.YI, self.Y0V, self.Y1TV = YI_no_dn_y, Y0V_no_dn_y, Y1TV_no_dn_y
                doc_Y1T = doc_Y1T_no_dn_y

            ## sampling for z ##
            # calculate leave-one out statistics #
            TE_no_d_z, Y1TV_no_d_z = self.TE, self.Y1TV

            TE_no_d_z[doc_z,:] += -docE

    def _saveCheckPoint(self, epoch, ppl = None, filename = None):
        if filename is None:
            filename = self.checkpoint_file
        state = {
            "theta": self.theta,
            "pi": self.pi,
            "eta": self.eta,
            "phiT": self.phiT,
            "phiB": self.phiB,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "z": self.z,
            "y": self.y,
            "YI": self.YI,
            "Y0V": self.Y0V,
            "Y1TV": self.Y1TV,
            "TE": self.TE,
            "epoch": epoch,
            "ppl": ppl
        }
        with open(filename, "w") as f_ckpt:
            cPickle.dump(state, f_ckpt)

    def _restoreCheckPoint(self, filename=None):
        if filename is None:
            filename = self.checkpoint_file
        state = cPickle.load(open(filename, "r"))
        # restore #
        self.theta = state["theta"]
        self.pi = state["pi"]
        self.eta = state["eta"]
        self.phiT = state["phiT"]
        self.phiB = state["phiB"]
        self.alpha = state["alpha"]
        self.beta = state["beta"]
        self.gamma = state["gamma"]
        self.delta = state["delta"]
        self.z = state["z"]
        self.y = state["y"]
        self.YI = state["YI"]
        self.Y0V = state["Y0V"]
        self.Y1TV = state["Y1TV"]
        self.TE = state["TE"]
        epoch = state["epoch"]
        ppl = state["ppl"]
        print "restore state from file '%s' on epoch %d with ppl: %s" % (filename, epoch, str(ppl))

