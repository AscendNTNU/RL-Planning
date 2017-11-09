import tensorflow as tf
import tensorflow.contrib.layers as layers
# Neural network setup
class Qnetwork():
    def __init__(self, lr, s_size, a_size, h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.inputs = tf.placeholder(shape=[None,s_size], dtype=tf.float32)
        self.dropout_ratio = tf.placeholder(shape=(), dtype=tf.float32)
        hidden = layers.fully_connected(self.inputs, h_size)
        hidden = layers.dropout(hidden, self.dropout_ratio)
        hidden2 = layers.fully_connected(hidden, int(h_size/2))
        # hidden2 = layers.dropout(hidden2, self.dropout_ratio)

        output1, output2 = tf.split(hidden2, 2, 1)
        self.advantage_weights = tf.Variable(tf.random_normal([int(h_size/4), a_size]))
        self.value_weights = tf.Variable(tf.random_normal([int(h_size/4), 1]))

        self.advantage = tf.matmul(output1, self.advantage_weights)
        self.value = tf.matmul(output2, self.value_weights)
        
        #Then combine them together to get our final Q-values.
        self.q_values = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
        self.q_dist = layers.softmax(self.q_values)
        self.predict = tf.argmax(self.q_values, 1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.q_values, self.actions_onehot), axis=1)
    
        self.error = tf.reduce_mean(tf.square(self.targetQ - self.Q))
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.error)