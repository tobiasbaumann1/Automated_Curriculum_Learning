import numpy as np
import tensorflow as tf

class Bandit(object):
	def __init__(self, n_arms, eps = 0.1, step_size = 0.01):
		self.n_arms = n_arms
		self.eps = eps
		self.step_size = step_size
		self.weights = np.zeros(n_arms)
		self.t = 0

	def choose_action(self, *args):
		self.t += 1
		return np.random.choice(self.n_arms, p = self.get_arm_probabilities())

	def get_arm_probabilities(self):
		sum_exp_weights = np.sum(np.exp(self.weights))
		return (1 - self.eps) * np.exp(self.weights) / sum_exp_weights + self.eps / self.n_arms

	def update_weights(self, a, r):
		self.weights[a] += self.step_size * r / self.get_arm_probabilities()[a] #only Exp3 so far, not Exp3.S!

class Contextual_Bandit(object):
	def __init__(self, n_arms, n_features, name = '', e_greedy = 0.9, e_greedy_increment = None, learning_rate = 0.01):
		self.n_arms = n_arms
		self.n_features = n_features
		self.epsilon_max = e_greedy
		self.epsilon_increment = e_greedy_increment
		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
		self.learning_rate = learning_rate
		self._build_net(name)
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def _build_net(self, str):
	    # ------------------ build evaluate_net ------------------
	    self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
	    self.r_target = tf.placeholder(tf.float32, [None, self.n_arms], name='r_target')  # for calculating loss
	    #self.importance_weight = tf.placeholder(tf.float32, [None, 1], name='importance_weight')
	    with tf.variable_scope('eval_net'):
	        # c_names(collections_names) are the collections to store variables
	        c_names, n_l1, w_initializer, b_initializer = \
	            ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
	            tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

	        # first layer. collections is used later when assign to target net
	        with tf.variable_scope('l1'):
	            w1 = tf.get_variable(str+'w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
	            b1 = tf.get_variable(str+'b1', [1, n_l1], initializer=b_initializer, collections=c_names)
	            l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

	        # second layer. collections is used later when assign to target net
	        with tf.variable_scope('l2'):
	            w2 = tf.get_variable(str+'w2', [n_l1, self.n_arms], initializer=w_initializer, collections=c_names)
	            b2 = tf.get_variable(str+'b2', [1, self.n_arms], initializer=b_initializer, collections=c_names)
	            self.r = tf.matmul(l1, w2) + b2

	    with tf.variable_scope('loss'):
	        self.loss = tf.reduce_mean(tf.squared_difference(self.r_target, self.r))
	    with tf.variable_scope('train'):
	        self._train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
		
	def choose_action(self, s):		
		argmax = np.argmax(self.sess.run([self.r], feed_dict={self.s: s[np.newaxis, :]}))
		p = self.get_arm_probabilities(argmax)
		a = np.random.choice(self.n_arms, p = p)
		#importance_weight = 1 / p[a]
		return a

	def get_arm_probabilities(self, argmax):
		p = np.full(self.n_arms, fill_value = (1-self.epsilon)/self.n_arms)
		p[argmax] += self.epsilon
		return p

	def learn(self, s, a, r):
		r_eval = self.sess.run([self.r], feed_dict={self.s: s[np.newaxis, :]})
		r_target = np.array(r_eval).flatten()
		r_target[a] = r
		_, self.cost = self.sess.run([self._train_op, self.loss],
                             		feed_dict={self.s: s[np.newaxis, :],
                                    self.r_target: r_target[np.newaxis, :]})
		# importance weight? add in tensorflow computations and as argument to learn etc.
		# increase epsilon
		if self.epsilon < self.epsilon_max:
			self.epsilon += self.epsilon_increment

