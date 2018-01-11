import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import cPickle
from datetime import datetime

a = np.ones([2,3,4])
b = np.ones([3,4])
print a + b