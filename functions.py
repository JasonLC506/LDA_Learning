import numpy as np
from scipy.special import gamma, gammaln

def probNormalize(distributions):
    if distributions.ndim > 1:
        return distributions / np.sum(distributions, axis=1, keepdims=True)
    else:
        return distributions / np.sum(distributions)

def multinomial(prob, size=1):
    return np.argmax(np.random.multinomial(1, prob, size), axis=1)

def multivariateBeta_inv(x):
    """
    calculate inverse multivariate beta function, as normalization factor for dirichlet distribution
    :param x: np.ndarray(n, d)
    """
    a = np.sum(gammaln(x), axis=1)
    b = gammaln(np.sum(x, axis=1))
    logresult = b - a
    print "############# multivariateBeta_inv #################"
    print "eta", x
    print "a", a
    print "b", b
    print "logresult", logresult
    print "####################################################"
    result = np.exp(logresult)
    return result

if __name__ == "__main__":
    x = np.ones([2,3])
    print multivariateBeta_inv(x)