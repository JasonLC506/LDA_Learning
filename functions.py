import numpy as np

def probNormalize(distributions):
    if distributions.ndim > 1:
        return distributions / np.sum(distributions, axis=1, keepdims=True)
    else:
        return distributions / np.sum(distributions)

def multinomial(prob, size=1):
    return np.argmax(np.random.multinomial(1, prob, size), axis=1)