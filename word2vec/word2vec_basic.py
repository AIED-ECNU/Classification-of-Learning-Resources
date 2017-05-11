# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: Download the data.从下面的网址中把文本下下来，如果不行，可以先下载放到这一位置上
# url = 'http://mattmahoney.net/dc/'
#
#
# def maybe_download(filename, expected_bytes):
#    """Download a file if not present, and make sure it's the right size."""
#    #os.path.abspath(path)   返回path规范化的绝对路径
#    if not os.path.exists(filename):
#      filename, _ = urllib.request.urlretrieve(url + filename, filename)
#    statinfo = os.stat(filename)#stat 系统调用时用来返回相关文件的系统状态信息
#    if statinfo.st_size == expected_bytes:
#      print('Found and verified', filename)
#    else:
#      print(statinfo.st_size)
#      raise Exception(
#          'Failed to verify ' + filename + '. Can you get to it with a browser?')
#    return filename

#filename = maybe_download('text8.zip', 31344016)
filename = "text8"
#文档路径改一下
# Read the data into a list of strings.读文本，输出单词数Data size 17005207
# def read_data(filename):
#   """Extract the first file enclosed in a zip file as a list of words."""
#   with zipfile.ZipFile(filename) as f:
#     data = tf.compat.as_str(f.read(f.namelist()[0])).split()
#   return data
def read_data(filename):
   """Extract the first file enclosed in a zip file as a list of words."""
   with open(filename) as f:
     for line in f:
        data = line.split()
   return data

vocabulary = read_data(filename)
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
# 建立词典，代替稀有的词语用UNK记号，UNK(也就是unknown单词，即词频率少于一定数量的稀有词的代号)
#限制了这个demo可以学习一共50000个不同的单词
vocabulary_size = 50000

#建立数据集，加工词 ，输出
#Most common words (+UNK) [['UNK', 418391], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764)]
#Sample data [5237, 3081, 12, 6, 195, 2, 3134, 46, 59, 156] ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']
def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  #计数器（Counter）是一个容器，用来跟踪值出现了多少次。
  #most_common会产生[('a', 5), ('r', 2), ('b', 2), ('c', 1), ('d', 1)]这样的效果
  dictionary = dict()
  #新建字典，例如dict = {"a" : "apple", "b" : "banana", "g" : "grape", "o" : "orange"}
  for word, _ in count:###？？？
    dictionary[word] = len(dictionary)##给文本里的词按出现次数标上序号，the：1
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)###是按照每个词的出现次数的值，先出现可能是anarchism排第几位
  count[0][1] = unk_count
  #字典的初始化，元素为value:key
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary
#调用上面的build_dataset函数
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])#包括UNK的前五个
##Sample data [5237, 3081, 12, 6, 195, 2, 3134, 46, 59, 156] ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']
print('Sample data', data[:16], [reverse_dictionary[i] for i in data[:16]])
##data从文本第一个词顺序进行


data_index = 0
# Step 3: Function to generate a training batch for the skip-gram model.训练函数
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  ##断言assert表示如果后面表达式为假，抛出错误；如果为真，继续执行
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  ##创建一个ndarray对象，shape是数组形状，dtype是数组中元素类型；batch是一行，labels是1列
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  ##规定双向序列deque最大长度为span
  buffer = collections.deque(maxlen=span)
  for _ in range(span):#循环span次
    buffer.append(data[data_index])##buffer应该是[5237, 3081, 12, 6, 195, 2, 3134, 46, 59, 156] 的数据，代表文本第一个按出现次数进行排位的第几位
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):#循环batch_size // num_skips次
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        ##随机取（0,1,2）中一个数，对应label单词，即一个单词附近的单词，但不会没取到自己，因为如果还是1 就会再取随机数，直到取到不是1 的数
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target) ##targets_to_avoid可以重复，比如[1,0,1,1]
      batch[i * num_skips + j] = buffer[skip_window]
      ##一直都是buffer[1]，但是buffer是个双向序列，可以不断变化，i=0 时，buffer=【50, 3272, 11】。i=1 时，buffer=【3272, 11，6】往后推了一个
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels
##调用函数，参数为8，2，1
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
##batch_size表示输出多少行，num_skips表示一个词出现几次，skip_window表示从skip_window+1个单词开始
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.建立模型

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.128维向量表示单词
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.随机的16个单词
valid_window = 100  # Only pick dev samples in the head of the distribution.
##从100个取16个不重复的。
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()
##tf.Graph类表示可计算的图
with graph.as_default():

  # Input data.
  #### tf.placeholder：占位符，由后面的feed_dict 参数指定
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
 #tf.constant(value,dtype=None,shape=None,name='Const')创建一个常量tensor，先给出value，可以设定其shape
  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    #vocabulary_size=50000,embedding_size=128;
    ##tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None)返回一个形状为shape的tensor，其中的元素服从minval和maxval之间的均匀分布。
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)##在embeddings中检索train_inputs所要求的内容

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,#128维，50000个
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,###负样本？？？
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  ###'x' is [[1, 1, 1]
  #         [1, 1, 1]]  tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
#两个矩阵相乘
  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
    #generate_batch用法:::batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)   ##batch_size表示输出多少行，num_skips表示一个词出现几次，skip_window表示从skip_window+1个单词开始
    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val
##每2000一次
    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.上2000批的平均失败估测，随次数增多，值变小
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    ##10000的整数倍时，输出nearest
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):#16
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):##循环8 次
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in range(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
