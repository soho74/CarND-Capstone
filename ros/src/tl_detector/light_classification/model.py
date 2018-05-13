import os
import sys
import time
from sklearn import metrics
from sklearn.utils import shuffle
from data_utils import gen_batch_function
import numpy as np
import tensorflow as tf


class Config:

    def __init__(self):
        self.batch_size = 32
        self.epochs = 200
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.keep_prob = 0.4
        self.intervel = 1


def conv(x, W, b, strides=(1, 1, 1, 1), padding='SAME'):
    y = tf.nn.conv2d(x, W, strides=strides, padding=padding)
    y = tf.nn.bias_add(y, b)
    return y


def max_pool(x, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), 
             padding='SAME'):
    return tf.nn.max_pool(
        x, ksize=ksize, strides=strides, padding=padding)


class TLClassifier:

    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = gen_batch_function(batch_size=self.cfg.batch_size)
        self.inputs, self.labels, self.keep_prob = \
            self.add_placeholders()
        self.logits = self.add_model()
        self.loss, self.celoss, self.train_op = \
            self.add_loss_and_train_op()
        self.prediction = self.predict()
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def add_placeholders(self):
        inputs = tf.placeholder(tf.float32, name='images')
        labels = tf.placeholder(tf.int32, name='labels')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        return inputs, labels, keep_prob

    def create_feed_dict(self, images, labels=None, keep_prob=1.):
        feed = {self.inputs: images,
                self.keep_prob: keep_prob}
        if labels is not None:
            feed[self.labels] = labels
        return feed

    def add_model(self):
        kp = self.keep_prob

        x1 = tf.nn.dropout(self.inputs, kp)
        W1 = tf.get_variable('W1', (3, 3, 3, 32))
        b1 = tf.get_variable('b1', (32,))
        o1 = conv(x1, W1, b1)
        o1 = tf.nn.relu(o1) 
        # o1 = max_pool(o1)

        x2 = tf.nn.dropout(o1, kp)
        W2 = tf.get_variable('W2', (3, 3, 32, 32))
        b2 = tf.get_variable('b2', (32,))
        o2 = conv(x2, W2, b2)
        o2 = tf.nn.relu(o2) + x2
        o2 = max_pool(o2)

        x3 = tf.nn.dropout(o2, kp)
        W3 = tf.get_variable('W3', (3, 3, 32, 32))
        b3 = tf.get_variable('b3', (32,))
        o3 = conv(x3, W3, b3)
        o3 = tf.nn.relu(o3) + x3
        o3 = max_pool(o3)
    
        x4 = tf.nn.dropout(o3, kp)
        W4 = tf.get_variable('W4', (1, 1, 32, 128))
        b4 = tf.get_variable('b4', (128,))
        o4 = conv(x4, W4, b4) 
        o4 = tf.nn.relu(o4)

        x5 = tf.nn.dropout(o4, kp)
        x5 = tf.reshape(x5, (-1, 12800))
        W5 = tf.get_variable('W5', (12800, 100))
        b5 = tf.get_variable('b5', (100, ))
        o5 = tf.matmul(x5, W5) + b5
        o5 = tf.nn.relu(o5)

        x6 = tf.nn.dropout(o5, kp)
        W6 = tf.get_variable('W6', (100, 4))
        b6 = tf.get_variable('b6', (4, ))
        o6 = tf.matmul(x6, W6) + b6

        return o6

    def add_loss_and_train_op(self):
        celoss = tf.losses.sparse_softmax_cross_entropy(
            self.labels, self.logits)
        weight_decay = self.cfg.weight_decay
        learning_rate = self.cfg.learning_rate
        tvars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tvars])
        loss = tf.reduce_mean(celoss) + weight_decay * l2_loss
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        return loss, celoss, train_op

    def predict(self):
        return tf.argmax(self.logits, 1, name='predict')

    def run_epoch(self, session, load=''):
        latest_ckpt = tf.train.latest_checkpoint(load)
        num_epochs = self.cfg.epochs
        keep_prob = self.cfg.keep_prob
        intervel = self.cfg.intervel
        if latest_ckpt:
            self.saver.restore(session, latest_ckpt)
        else:
            session.run(self.init_op)
        start_time = time.time()
        best_acc = 0
        if not os.path.exists('./save'):
            os.makedirs('./save')

        best_celoss = float('inf')
        for epoch in range(num_epochs):
            total_loss = 0
            total_celoss = 0
            X_train, y_train = shuffle(self.dataset[0], self.dataset[1])
            for i in range(X_train.shape[0]):
                X, y = shuffle(X_train[i], y_train[i])
                feed_dict = self.create_feed_dict(X, y, keep_prob)
                loss, celoss, _ = session.run(
                    [self.loss, self.celoss, self.train_op], 
                    feed_dict=feed_dict)
                total_loss += loss
                total_celoss += celoss
                sys.stdout.write('\r')
                time_cost = time.time() - start_time
                step = i + 1
                o_str = '\rEpoch {:>3}, step {:>4}, time {:>5.0f}, celoss {:.4f}, loss {:.4f}'
                cur_loss = total_loss / step
                cur_celoss = total_celoss / step
                sys.stdout.write(
                    o_str.format(epoch+1, step, time_cost, cur_celoss, cur_loss))
                sys.stdout.flush()

            dev_celoss, dev_acc = self.evaluation(
                session, self.dataset[2], self.dataset[3])
            if best_acc < dev_acc or best_celoss > dev_celoss:
                best_acc = dev_acc
                best_celoss = dev_celoss
                self.saver.save(session, os.path.join('./save', 'tlc.ckpt'))
            if (epoch + 1) % intervel == 0:
                test_celoss, test_acc = self.evaluation(
                    session, self.dataset[4], self.dataset[5], 'Test')

    def evaluation(self, session, Xs, ys, evaltye='Valid', load=None):
        if load is not None:
            latest_ckpt = tf.train.latest_checkpoint(load)
            self.saver.restore(session, latest_ckpt)
        total_celoss = 0
        eval_preds = []
        step = 1
        labels = []
        for i in range(Xs.shape[0]):
            X, y = Xs[i], ys[i]
            feed_dict = self.create_feed_dict(X, y)
            celoss, preds = session.run(
                [self.celoss, self.prediction], feed_dict=feed_dict)
            total_celoss += celoss
            eval_preds += list(preds)
            labels += y
            step += 1

        avg_celoss = total_celoss / step
        labels = np.array(labels)
        eval_preds = np.array(eval_preds)
        if evaltye == 'Valid':
            print
        eval_acc = metrics.accuracy_score(labels, eval_preds)
        o_str = '{:<5}, celoss {:.4f}, Accuracy {:.4f}'
        print(o_str.format(evaltye, avg_celoss, eval_acc))
        if evaltye == 'Test':
            print
        return avg_celoss, eval_acc


if __name__ == "__main__":
    cfg = Config()
    tlc = TLClassifier(cfg)
    with tf.Session() as sess:
        tlc.run_epoch(sess)
        # tlc.evaluation(sess, tlc.dataset[4], tlc.dataset[5], 'Test', load='save/')


