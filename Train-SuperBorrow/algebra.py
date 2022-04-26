import numpy as np
import math
from scipy.stats import entropy

#------------------------------------------------------------------
def cosine(x, y):
	"""
    Compute the cosine similarity between two vectors x and y. 
	"""
	x=normalize(x)
	y=normalize(y)
	return np.dot(x,y.T)
#------------------------------------------------------------------
def normalize(x):
    """
    L2 normalize vector x. 
    """
    norm_x = np.linalg.norm(x)
    return x if norm_x == 0 else (x / norm_x)
#------------------------------------------------------------------
def softmax_fun(x):
	e_x=np.exp(x - np.max(x))
	return e_x / np.sum(e_x)
#------------------------------------------------------------------
def sigmoid_fun(x):
	return 1/(1+np.exp(-x))
#------------------------------------------------------------------	
def KL_divergence(a,b):
	return np.sum(a*np.log(a/b))	
#------------------------------------------------------------------	
def JSD(P, Q):
	M = 0.5 * (P + Q)
	return 0.5 * (entropy(P, M) + entropy(Q, M))

#------------------------------------------------------------------		
def ReLU(x):
    return abs(x) * (x > 0)	
#------------------------------------------------------------------ 