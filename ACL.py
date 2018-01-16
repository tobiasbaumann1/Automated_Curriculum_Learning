import numpy as np

class Bandit(object):
	def __init__(self, n_arms, eps = 0.1, step_size = 0.01):
		self.n_arms = n_arms
		self.eps = eps
		self.step_size = step_size
		self.weights = np.zeros(n_arms)
		self.t = 0

	def sample_arm(self):
		self.t += 1
		return np.random.choice(self.n_arms, p = self.get_arm_probabilities())

	def get_arm_probabilities(self):
		sum_exp_weights = np.sum(np.exp(self.weights))
		return (1 - self.eps) * np.exp(self.weights) / sum_exp_weights + self.eps / self.n_arms

	def update_weights(self, a, r):
		self.weights[a] += self.step_size * r / self.get_arm_probabilities()[a] #only Exp3 so far, not Exp3.S!

