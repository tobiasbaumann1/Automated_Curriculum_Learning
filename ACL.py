import numpy as np

class Bandit_ACL(object):
	def __init__(self, n_actions):
		self.n_actions = n_actions
		self.weights = np.