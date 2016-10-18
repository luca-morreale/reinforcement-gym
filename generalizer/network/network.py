import tensorflow as tf
import numpy as np

class NetworkGeneralizer():
    ''' Class's internal variables
    _session
    _learning_rate
    _hidden_units
    net_x
    net_y
    target_out
    _loss
    _optimizer
    '''

    def __init__(self, session, state_dim, hidden_units=[100, 200], learning_rate=0.1):
        self._session = session
        self._learning_rate = learning_rate

        self._setup_layers(hidden_units, state_dim)
        self.net_x, self.net_y = self._create_net()
        self.target_out = tf.placeholder(tf.float32, [None, 1])

        self._define_loss()

    def _setup_layers(self, hidden_units, state_dim):
        hidden_units.insert(0, state_dim)
        hidden_units.append(1)
        self._hidden_units = hidden_units

    def _create_net(self):
        x = tf.placeholder(tf.float32, [None, self._hidden_units[0]])
        y = x

        for i in range(1, len(self._hidden_units)):
            W = tf.Variable(tf.zeros([self._hidden_units[i - 1], self._hidden_units[i]]))
            b = tf.Variable(tf.zeros([self._hidden_units[i]]))
            y = tf.nn.tanh(tf.matmul(y, W) + b)

        return x, y

    def _define_loss(self):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.net_y, self.target_out))
        self._loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        self._optimizer = tf.train.GradientDescentOptimizer(self._learning_rate).minimize(self._loss)

    def train(self, inputs, target_out):
        return self._session.run([self.net_y, self._optimizer], feed_dict={
                self.net_x: inputs,
                self.target_out: target_out
            })

    def predict(self, inputs):
        return self._session.run(self.net_y, feed_dict={
                self.net_x: inputs
            })
