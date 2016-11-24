
import tensorflow as tf

from network import NeuralNetwork

# in case of a few number of neurons in the hidden unit specify tf.nn.relu as activation function
class RegressionNetwork(NeuralNetwork):
    def __init__(self, session, state_dim, hidden_units=[100, 200], learning_rate=0.1, act_function=tf.nn.tanh):
        NeuralNetwork.__init__(self, session, state_dim, hidden_units, learning_rate, act_function)
        self._add_linear_out()
        self._define_loss()

    def _add_linear_out(self):
        W = tf.Variable(tf.zeros([self._hidden_units[-1], 1]))
        b = tf.Variable(tf.zeros([1]))
        self.net_y = tf.matmul(self.net_y, W) + b

    def _define_loss(self):
        self._loss = tf.reduce_mean(tf.pow(tf.sub(self.net_y, self.target_out), 2.0))
        self._optimizer = tf.train.GradientDescentOptimizer(self._learning_rate).minimize(self._loss)
