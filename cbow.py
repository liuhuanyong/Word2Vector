#!/usr/bin/env python3
# coding: utf-8
# File: cbow.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-21

import math
import random
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import collections
class CBOW:

    def __init__(self):
        self.data_index = 0

    #定义读取数据的函数，并把数据转成列表
    def read_data(self, filename):
        with open(filename) as f:
            words = f.read().split(' ')
        return words

    #创建数据集
    def build_dataset(self, words, min_count):
        # 创建词汇表，过滤低频次词语，这里使用的人是mincount>=5，其余单词认定为Unknown,编号为0,
        # 这一步在gensim提供的wordvector中，采用的是minicount的方法
        #对原words列表中的单词使用字典中的ID进行编号，即将单词转换成整数，储存在data列表中，同时对UNK进行计数
        count = [['UNK', -1]]
        reserved_words = [item for item in collections.Counter(words).most_common() if item[1] >= min_count]
        count.extend(reserved_words)
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
            data.append(index)
        count[0][1] = unk_count
        print(len(count))
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reverse_dictionary


    #生成训练样本，assert断言：申明其布尔值必须为真的判定，如果发生异常，就表示为假
    def generate_batch(self, batch_size, skip_window, data):
        # 该函数根据训练样本中词的顺序抽取形成训练集
        # batch_size:每个批次训练多少样本
        # skip_window:单词最远可以联系的距离（本次实验设为5，即目标单词只能和相邻的两个单词生成样本），2*skip_window>=num_skips

        span = 2 * skip_window + 1  # [ skip_window target skip_window ]

        batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        buffer = collections.deque(maxlen=span)

        for _ in range(span):
            buffer.append(data[self.data_index])
            self.data_indexdata_index = (self.data_index + 1) % len(data)

        for i in range(batch_size):
            target = skip_window
            target_to_avoid = [skip_window]

            col_idx = 0
            for j in range(span):
                if j == span // 2:
                    continue
                batch[i, col_idx] = buffer[j]
                col_idx += 1
            labels[i, 0] = buffer[target]

            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)

        assert batch.shape[0] == batch_size and batch.shape[1] == span - 1

        return batch, labels

    def train_wordvec(self, vocabulary_size, batch_size, embedding_size, window_size, num_sampled, num_steps, data):
        #定义CBOW Word2Vec模型的网络结构

        graph = tf.Graph()

        with graph.as_default(), tf.device('/cpu:0'):
            train_dataset = tf.placeholder(tf.int32, shape=[batch_size, 2 * window_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                              stddev=1.0 / math.sqrt(embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # 与skipgram不同， cbow的输入是上下文向量的均值，因此需要做相应变换
            context_embeddings = []
            for i in range(2 * window_size):
                context_embeddings.append(tf.nn.embedding_lookup(embeddings, train_dataset[:, i]))

            avg_embed = tf.reduce_mean(tf.stack(axis=0, values=context_embeddings), 0, keep_dims=False)

            loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=avg_embed,
                                           labels=train_labels, num_sampled=num_sampled,
                                           num_classes=vocabulary_size))
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            average_loss = 0
            for step in range(num_steps):
                batch_data, batch_labels = self.generate_batch(batch_size, window_size, data)
                feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
                _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += l
                if step % 2000 == 0:
                    if step > 0:
                        average_loss = average_loss / 2000
                        # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0
            final_embeddings = normalized_embeddings.eval()

        return final_embeddings

    #训练主函数
    def train(self, filepath, modelpath, cluster_imgpath):
       # filepath = 'text8' #训练数据文件
       # model_path = 'wordvec.bin'
       # cluster_imgpath = 'cluster_cbow.png'
        min_count = 5 # 默认最低频次的单词
        batch_size = 200  # 每次迭代训练选取的样本数目
        embedding_size = 200  # 生成词向量的维度
        window_size = 1  # 考虑前后几个词，窗口大小
        num_steps = 100001#定义最大迭代次数，创建并设置默认的session，开始实际训练
        num_sampled = 100  # Number of negative examples to sample.
        words = self.read_data(filepath)
        print('Data size', len(words))
        data, count, dictionary, reverse_dictionary = self.build_dataset(words, min_count)
        print('Most common words (+UNK)', count[:5])
        print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
        print('Reserved words', len(count))
        vocabulary_size = len(count)
        final_embeddings = self.train_wordvec(vocabulary_size, batch_size, embedding_size, window_size, num_sampled, num_steps, data)
        print('saving embeddings into localpath.......')
        self.save_embedding(final_embeddings, modelpath, reverse_dictionary)
        print('clustering words using trained embeddings')
        self.cluster_show(reverse_dictionary, final_embeddings, cluster_imgpath)

    #定义可视化Word2Vec效果的函数
    def plot_with_labels(self, low_dim_embs, labels, imagename):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figsize= (18, 18))
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label, xy = (x, y), xytext= (5, 2), textcoords = 'offset points', ha = 'right', va = 'bottom')
        plt.savefig(imagename)

    #聚类展示
    def cluster_show(self, reverse_dictionary, final_embeddings, imagename):
        #使用tsne进行降维
        tsne = TSNE(perplexity = 30, n_components = 2, init = 'pca', n_iter = 5000)
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in range(plot_only)]
        self.plot_with_labels(low_dim_embs, labels, imagename)

    #保存embedding文件
    def save_embedding(self, final_embeddings, model_path, reverse_dictionary):
        f = open(model_path,'w+')
        for index, item in enumerate(final_embeddings):
            f.write(reverse_dictionary[index] + '\t' + ' '.join([str(vec) for vec in item]) + '\n')
        f.close()


filepath = './data/text8'
modelpath = './model/cbow_wordvec.bin'
cluster_imgpath = './img/cbow_cluster.png'

vector = CBOW()
vector.train(filepath, modelpath, cluster_imgpath)
