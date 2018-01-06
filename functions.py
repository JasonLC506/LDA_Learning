import numpy as np
from scipy.special import gamma

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
    a = np.prod(gamma(x), axis=1)
    b = gamma(np.sum(x, axis=1))
    # print "############# multivariateBeta_inv #################"
    # print "eta", x
    # print "a", a
    # print "b", b
    # print "####################################################"
    return np.divide(b, a)

if __name__ == "__main__":
    x = np.ones([2,3])
    print multivariateBeta_inv(x)