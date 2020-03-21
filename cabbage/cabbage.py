import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pandas.io.parsers import read_csv
import numpy as np
# avgTemp,minTemp,maxTemp,rainFall,avgPrice
class Cabbage:

    def model(self):
        tf.global_variables_initializer()
        data = read_csv('cabbage_price.csv', sep=',')
        xy = np.array(data, dtype=np.float32)
        x_data = xy[:,1:-1]
        y_data = xy[:,[-1]]
        X = tf.placeholder(tf.float32, shape=[None, 4])
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        W = tf.Variable(tf.random_normal([4,1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        hypothesis = tf.matmul(X, W) + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        optmizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optmizer.minimize(cost)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(100000):
                cost_, hypo_, _ = sess.run([cost, hypothesis, train], {X: x_data, Y: y_data})
                if step % 500 == 0:
                    print(f' step: {step}, cost: {cost_} ')
                    print(f' price: {hypo_}')
            saver = tf.train.Saver()
            saver.save(sess, 'cabbage.ckpt')

    def initialize(self, avgTemp,minTemp,maxTemp,rainFall):
        self.avgTemp = avgTemp
        self.minTemp = minTemp
        self.maxTemp = maxTemp
        self.rainFall = rainFall

    def service(self):
        X = tf.placeholder(tf.float32, shape=[None, 4])
        W = tf.Variable(tf.random_normal([4,1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, 'cabbage/cabbage.ckpt')
            data = [[self.avgTemp, self.minTemp, self.maxTemp, self.rainFall],]
            arr = np.array(data, dtype=np.float32)
            dict = sess.run( tf.matmul(X, W) + b, {X: arr[0:4]})
        return int(dict[0])

if __name__ == '__main__':
    cabbage = Cabbage()
    cabbage.model()
