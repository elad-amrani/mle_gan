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

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
sys.stdout = sys.stderr
import numpy as np
import tensorflow as tf
import os

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("sample_mode", False,
                  "Must have trained model ready. Only does sampling")
flags.DEFINE_bool("train_gan", False,
                  "Train model in GAN mode")
flags.DEFINE_bool("train_lm", False,
                  "Train model in LM mode")
flags.DEFINE_bool("save_embeddings", False,
                  "Save Generator's embeddings")
flags.DEFINE_bool("load_embeddings", False,
                  "Initialize Generator and Discriminator with pre-trained embeddings")
flags.DEFINE_string("d_arch", "nn",
                    "Discriminator architecture. nn / dnn / lstm.")
flags.DEFINE_string("file_prefix", "ptb",
                  "will be looking for file_prefix.train.txt, file_prefix.test.txt and file_prefix.valid.txt in data_path")
flags.DEFINE_string("seed_for_sample", "i am",
                  "supply seeding phrase here. it must only contain words from vocabluary")
flags.DEFINE_integer('gan_steps', 10,
                     'Train discriminator / generator for gan_steps mini-batches before switching (in dual mode)')
flags.DEFINE_integer('d_steps', 10,
                     'Train discriminator for d_steps mini-batches before training generator (in gan mode)')
flags.DEFINE_integer('g_steps', 10,
                     'Train generator for g_steps mini-batches after training discriminator (in gan mode)')
flags.DEFINE_float('gan_lr', 1e-3, 
                   'GAN learning rate')
flags.DEFINE_float('tau', 1e-1, 
                   'Gumbel Softmax temperature')
flags.DEFINE_integer('total_epochs', 26,
                     'The total number of epochs for training')


FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class Generator(object):
  """Generator."""

  def __init__(self, is_training, config, reuse = False, mode = "LM"):
      with tf.variable_scope("Generator", reuse=reuse):

        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        tau = config.tau
    
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        
    
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
          lstm_cell = tf.contrib.rnn.DropoutWrapper(
              lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
    
        self._initial_state = cell.zero_state(batch_size, data_type())
    
    
        with tf.device("/cpu:0"):
          if (FLAGS.load_embeddings):
              embeddings = np.load(os.path.join(FLAGS.data_path, FLAGS.file_prefix + "_embeddings.npy"))
              self._embedding = tf.get_variable("embedding", dtype=data_type(),initializer=embeddings)
          else:
              self._embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
          inputs = tf.nn.embedding_lookup(self._embedding, self._input_data)
    
        if is_training and config.keep_prob < 1:
          inputs = tf.nn.dropout(inputs, config.keep_prob)
    
        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = [tf.squeeze(input_step, [1])
        #           for input_step in tf.split(1, num_steps, inputs)]
        # outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
        state = self._initial_state
        self._logits = []
        self._soft_embed = []
        self._probs = []
        self._gumbels_softmax = []
        
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if (mode == "LM"):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(inputs[:, time_step, :], state)
                else: # (mode == "GAN")
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                        (cell_output, state) = cell(self._soft_embed[time_step-1], state)
                    else:
                        (cell_output, state) = cell(tf.random_normal([batch_size, size], dtype="float32", mean=0, stddev=config.noise_var), state)
                
                # logit
                logit = tf.matmul(cell_output, softmax_w) + softmax_b
                self._logits.append(logit)
                
                # Probability vector
                prob = tf.nn.softmax(logit)
                self._probs.append(prob)
                
                # Gumbel softmax trick
                uniform = tf.random_uniform([batch_size, vocab_size], minval=0, maxval=1)
                gumbel = -tf.log(-tf.log(uniform))
                gumbel_softmax = tf.nn.softmax((1.0 / tau) * (gumbel + logit))
                self._gumbels_softmax.append(gumbel_softmax)
                
                # "Soft" embedding
                soft_embedding = tf.matmul(prob, self._embedding)
                self._soft_embed.append(soft_embedding)
        
        # Reshape
        self._logits = tf.reshape(tf.concat(axis=1, values=self._logits), [-1, vocab_size])
        self._probs = tf.reshape(tf.concat(axis=1, values=self._probs), [-1, vocab_size])
        self._gumbels_softmax = tf.reshape(tf.concat(axis=1, values=self._gumbels_softmax), [-1, vocab_size])
        
        # During inference sample output from probability vector
        self.sample = tf.multinomial(self._logits, 1)
        
        # LM loss
        lm_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [self._logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._lm_cost = lm_cost = tf.reduce_sum(lm_loss) / batch_size
        self._final_state = state  
            
        if not is_training:
          return
    
        self._lr = tf.Variable(0.0, trainable=False)
        
        tvars = tf.trainable_variables()
        
        grads, _ = tf.clip_by_global_norm(tf.gradients(lm_cost, tvars),
                                          config.max_grad_norm)
        if config.optimizer == 'RMSPropOptimizer':
          optimizer = tf.train.RMSPropOptimizer(self._lr)
        elif config.optimizer == 'AdamOptimizer':
          optimizer = tf.train.AdamOptimizer()
        elif config.optimizer == 'MomentumOptimizer':
          optimizer = tf.train.MomentumOptimizer(self._lr, momentum=0.8, use_nesterov=True)
        else:
          optimizer = tf.train.GradientDescentOptimizer(self._lr)
    
        self._lm_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())
    
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets


  @property
  def initial_state(self):
    return self._initial_state

  @property
  def lm_cost(self):
    return self._lm_cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def lm_op(self):
    return self._lm_op

  @property
  def logits(self):
    return self._logits

  @property
  def probs(self):
    return self._probs

  @property
  def gumbels_softmax(self):
    return self._gumbels_softmax

  @property
  def embedding(self):
    return self._embedding


class Discriminator(object):
  """Discriminator."""
  def __init__(self, is_training, config, probs = None, reuse = False, d_arch = "nn"):        
      with tf.variable_scope("Discriminator", reuse=reuse):
          
          self.batch_size = batch_size = config.batch_size
          self.num_steps = num_steps = config.num_steps
          size = config.hidden_size
          vocab_size = config.vocab_size      
          
          self.ids = tf.placeholder(tf.int32, [batch_size, num_steps])             
          
          with tf.device("/cpu:0"):
              if (FLAGS.load_embeddings):
                  embeddings = np.load(os.path.join(FLAGS.data_path, FLAGS.file_prefix + "_embeddings.npy"))
                  self._embedding = tf.get_variable("embedding", dtype=data_type(),initializer=embeddings)
              else:
                  self._embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
              # Apply embeddings lookup for real data or during inference of generator
              if (probs is None):
                  inputs = tf.nn.embedding_lookup(self._embedding, self.ids)
              else:
                  inputs = tf.matmul(probs, self._embedding)
                  
          if (d_arch == "nn"):
                        
              # Flatten embeddings
              flat = tf.reshape(inputs, [-1, size * num_steps])
                          
              # Fully connected layer
              softmax_w = tf.get_variable("softmax_w", [size * num_steps, 1], dtype=data_type())
              softmax_b = tf.get_variable("softmax_b", [1], dtype=data_type())
                
              # Classify
              self._logit = tf.matmul(flat, softmax_w) + softmax_b
              
          elif (d_arch == "dnn"):
              
              # Flatten embeddings
              flat = tf.reshape(inputs, [-1, size * num_steps])
                          
              # First layer
              softmax_w1 = tf.get_variable("softmax_w1", [size * num_steps, 400], dtype=data_type())
              softmax_b1 = tf.get_variable("softmax_b1", [400], dtype=data_type())
              layer_1 = tf.matmul(flat, softmax_w1) + softmax_b1
              layer_1 = tf.nn.relu(layer_1)
              
              # Second layer
              softmax_w2 = tf.get_variable("softmax_w2", [400, 1], dtype=data_type())
              softmax_b2 = tf.get_variable("softmax_b2", [1], dtype=data_type())
                   
              # Classify
              self._logit = tf.matmul(layer_1, softmax_w2) + softmax_b2
              
          elif (d_arch == "lstm"):
              
              inputs = tf.reshape(inputs, [batch_size, num_steps, size])
              
              lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
              cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
              self._initial_state = cell.zero_state(batch_size, data_type())
             
              outputs = []
              state = self._initial_state
              with tf.variable_scope("RNN"):
                  for time_step in range(num_steps):
                      if time_step > 0: tf.get_variable_scope().reuse_variables()
                      (cell_output, state) = cell(inputs[:, time_step, :], state)
                      outputs.append(cell_output)
            
              # Extract last output
              softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
              softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
              
              #Classify
              self._logit = tf.matmul(outputs[-1], softmax_w) + softmax_b

              
          
  @property
  def logit(self):
    return self._logit      
    


class SmallConfig(object):
  """Small config."""
  is_char_model = False
  optimizer = 'AdamOptimizer'
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
#  max_max_epoch = 13
  max_max_epoch = FLAGS.total_epochs
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  noise_var = 1
  gan_learning_rate = FLAGS.gan_lr
  tau = FLAGS.tau


class MediumConfig(object):
  """Medium config."""
  is_char_model = False
  optimizer = 'GradientDescentOptimizer'
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
#  max_max_epoch = 39
  max_max_epoch = FLAGS.total_epochs
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000
  noise_var = 1
  gan_learning_rate = FLAGS.gan_lr
  tau = FLAGS.tau


class LargeConfig(object):
  """Large config."""
  is_char_model = False
  optimizer = 'GradientDescentOptimizer'
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
#  max_max_epoch = 55
  max_max_epoch = FLAGS.total_epochs
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000
  noise_var = 1
  gan_learning_rate = FLAGS.gan_lr
  tau = FLAGS.tau


class CharLargeConfig(object):
  """Large config."""
  is_char_model = True
  optimizer = 'MomentumOptimizer'
  init_scale = 0.004
  learning_rate = 0.05
  max_grad_norm = 15
  num_layers = 3
  num_steps = 100
  hidden_size = 512
  max_epoch = 14
#  max_max_epoch = 255
  max_max_epoch = FLAGS.total_epochs
  keep_prob = 0.5
  lr_decay = 1 / 1.15
  #batch_size = 64
  batch_size = 1
  vocab_size = 10000
  noise_var = 1

class CharLargeConfig1(object):
  """Large config."""
  is_char_model = True
  optimizer = 'RMSPropOptimizer'
  init_scale = 0.004
  learning_rate = 0.01
  max_grad_norm = 15
  num_layers = 3
  num_steps = 128
  hidden_size = 512
  max_epoch = 14
#  max_max_epoch = 255
  max_max_epoch = FLAGS.total_epochs  
  keep_prob = 0.5
  lr_decay = 1 / 1.15
  batch_size = 16
  vocab_size = 10000
  noise_var = 1


class CharSmallConfig(object):
  """Large config."""
  is_char_model = True
  optimizer = 'RMSPropOptimizer'
  init_scale = 0.04
  learning_rate = 0.05
  max_grad_norm = 15
  num_layers = 3
  num_steps = 128
  hidden_size = 256
  max_epoch = 14
#  max_max_epoch = 155
  max_max_epoch = FLAGS.total_epochs
  keep_prob = 0.5
  lr_decay = 1 / 1.15
  batch_size = 8
  vocab_size = 10000
  noise_var = 1



class TestConfig(object):
  """Tiny config, for testing."""
  is_char_model = False
  optimizer = 'GradientDescentOptimizer'
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  noise_var = 1

def get_grams(lines):
    lines_joined = [''.join(l) for l in lines]

    unigrams = dict()
    bigrams = dict()
    trigrams = dict()
    quadgrams = dict()
    quintagrams = dict()
    sextagrams = dict()
    token_count = 0

    for l in lines_joined:
        l = l.split(" ")
        l = filter(lambda x: x != ' ' and x != '', l)

        for i in range(len(l)):
            token_count += 1
            unigrams[l[i]] = unigrams.get(l[i], 0) + 1
            if i >= 1:
                bigrams[(l[i - 1], l[i])] = bigrams.get((l[i - 1], l[i]), 0) + 1
            if i >= 2:
                trigrams[(l[i - 2], l[i - 1], l[i])] = trigrams.get((l[i - 2], l[i - 1], l[i]), 0) + 1
            if i >= 3:
                quadgrams[(l[i - 3], l[i - 2], l[i - 1], l[i])] = quadgrams.get((l[i - 3], l[i - 2], l[i - 1], l[i]), 0) + 1
            if i >= 4:
                quintagrams[(l[i - 4], l[i - 3], l[i - 2], l[i - 1], l[i])] = quintagrams.get((l[i - 4], l[i - 3], l[i - 2], l[i - 1], l[i]), 0) + 1
            if i >= 5:
                sextagrams[(l[i - 5], l[i - 4], l[i - 3], l[i - 2], l[i - 1], l[i])] = sextagrams.get((l[i - 5], l[i - 4], l[i - 3], l[i - 2], l[i - 1], l[i]), 0) + 1

    return unigrams, bigrams, trigrams, quadgrams, quintagrams, sextagrams


def percentage_real(samples_grams, real_grams):
    grams_in_real = 0

    for g in samples_grams:
        if g in real_grams:
            grams_in_real += 1
    if len(samples_grams) > 0:
        return grams_in_real * 1.0 / len(samples_grams)
    return 0

def do_sample(session, model, data, num_samples):
  """Sampled from the model"""
  samples = []
  state = session.run(model.initial_state)
  fetches = [model.final_state, model.sample]
  sample = None
  for x in data:
    feed_dict = {}
    feed_dict[model.input_data] = [[x]]
    for layer_num, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[layer_num].c
      feed_dict[h] = state[layer_num].h

    state, sample = session.run(fetches, feed_dict)
  if sample is not None:
    samples.append(sample[0][0])
  else:
    samples.append(0)
  k = 1
  while k < num_samples:
    feed_dict = {}
    feed_dict[model.input_data] = [[samples[-1]]]
    for layer_num, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[layer_num].c
      feed_dict[h] = state[layer_num].h
    state, sample = session.run(fetches, feed_dict)
    samples.append(sample[0][0])
    k += 1
  return samples


def run_lm_epoch(session, model, data, is_train=False, verbose=False):
  """Runs the LM model on the given data."""
  epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps)):
    if is_train:
      fetches = [model.lm_cost, model.final_state, model.lm_op]
    else:
      fetches = [model.lm_cost, model.final_state]
    feed_dict = {}
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y
    for layer_num, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[layer_num].c
      feed_dict[h] = state[layer_num].h

    if is_train:
      cost, state, _ = session.run(fetches, feed_dict)
    else:
      cost, state = session.run(fetches, feed_dict)

    costs += cost
    iters += model.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * model.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def run_gan_epoch(session, modelD, data, config, dLossReal, dLossFake, dLoss, gLoss, dOpt, gOpt, is_train=False, verbose=False):
  """Runs the GAN model on the given data."""
  epoch_size = ((len(data) // modelD.batch_size) - 1) // modelD.num_steps
  start_time = time.time()
  dCostsReal = 0.0
  dCostsFake = 0.0
  dCosts = 0.0
  gCosts = 0.0  
  iters = 0

  for step, (x, y) in enumerate(reader.ptb_iterator(data, modelD.batch_size, modelD.num_steps)):
      
    feed_dict = {}
    feed_dict[modelD.ids] = x
    
#    if is_train:
    if (step % (FLAGS.d_steps + FLAGS.g_steps) < FLAGS.d_steps): # Train discriminator
        fetches = [dOpt, dLossReal, dLossFake, dLoss, gLoss]
        _, dCostReal, dCostFake, dCost, gCost = session.run(fetches, feed_dict)
    else: # Train generator
        fetches = [gOpt, dLossReal, dLossFake, dLoss, gLoss]
        _, dCostReal, dCostFake, dCost, gCost= session.run(fetches, feed_dict)
        
    dCostsReal += dCostReal
    dCostsFake += dCostFake
    dCosts += dCost
    gCosts += gCost
    iters += 1
        
#    else:
#        fetches = [model.lm_cost, model.final_state]

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f dLossReal: %.3f dLossFake: %.3f dLoss: %.3f gLoss: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, dCostsReal / iters, dCostsFake / iters, dCosts / iters, gCosts / iters,
             iters * modelD.batch_size / (time.time() - start_time)))

  return dCostsReal / iters, dCostsFake / iters, dCosts / iters, gCosts / iters


def run_dual_epoch(session, modelD, modelLM, data, config, dLossReal, dLossFake, dLoss, gLoss, dOpt, gOpt, epoch_num, is_train=False,
                  verbose=False):
    """Runs the GAN model on the given data."""
    epoch_size = ((len(data) // modelD.batch_size) - 1) // modelD.num_steps
    start_time = time.time()
    dCostsReal = 0.0
    dCostsFake = 0.0
    dCosts = 0.0
    gCosts = 0.0
    iters = 0
    gLMcosts = 0.0
    gLMiters = 0
    state = session.run(modelLM.initial_state)

    for step, (x, y) in enumerate(reader.ptb_iterator(data, modelD.batch_size, modelD.num_steps)):
        
        trainD = (step % (2 * FLAGS.gan_steps) < FLAGS.gan_steps) == (epoch_num % 2 == 0) 

        if (trainD):  # Train discriminator
            fetches = [dOpt, dLossReal, dLossFake, dLoss, gLoss, modelLM.lm_cost, modelLM.final_state]            
        else:  # Train generator
            fetches = [gOpt, dLossReal, dLossFake, dLoss, gLoss, modelLM.lm_cost, modelLM.final_state]

        feed_dict = {}
        feed_dict[modelD.ids] = x
        feed_dict[modelLM.input_data] = x
        feed_dict[modelLM.targets] = y
        
        for layer_num, (c, h) in enumerate(modelLM.initial_state):
                        feed_dict[c] = state[layer_num].c
                        feed_dict[h] = state[layer_num].h
                        
        _, dCostReal, dCostFake, dCost, gCost, gLMCost, state = session.run(fetches, feed_dict)
                        
        dCostsReal += dCostReal
        dCostsFake += dCostFake
        dCosts += dCost
        gCosts += gCost
        iters += 1
        gLMcosts += gLMCost
        gLMiters += modelLM.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f dLossReal: %.3f dLossFake: %.3f dLoss: %.3f gLoss: %.3f gPerplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, dCostsReal / iters, dCostsFake / iters, dCosts / iters, gCosts / iters,
                   np.exp(gLMcosts / gLMiters), iters * modelD.batch_size / (time.time() - start_time)))

    return dCostsReal / iters, dCostsFake / iters, dCosts / iters, gCosts / iters, np.exp(gLMcosts / gLMiters)

def pretty_print(items, is_char_model, id2word):
  if not is_char_model:
    return ' '.join([id2word[x] for x in items])
  else:
    return ''.join([id2word[x] for x in items]).replace('_', ' ')


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  elif FLAGS.model == "charlarge":
    return CharLargeConfig()
  elif FLAGS.model == "charlarge1":
    return CharLargeConfig1()
  elif FLAGS.model == "charsmall":
    return CharSmallConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path, FLAGS.file_prefix)
  train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
  vocab_size = len(word_to_id)
  #print(word_to_id)
  print('Distinct terms: %d' % vocab_size)
  config = get_config()
  config.vocab_size = config.vocab_size if config.vocab_size < vocab_size else vocab_size
  eval_config = get_config()
  eval_config.vocab_size = eval_config.vocab_size if eval_config.vocab_size < vocab_size else vocab_size
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  if config.is_char_model:
    seed_for_sample = [c for c in FLAGS.seed_for_sample.replace(' ', '_')]
  else:
    seed_for_sample = FLAGS.seed_for_sample.split()

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.name_scope("Train"):
      with tf.variable_scope("Model", initializer=initializer):
        mGenLM = Generator(is_training=True, config=config, reuse=False, mode="LM")
        mGenGAN = Generator(is_training=True, config=config, reuse=True, mode="GAN")
        mDesReal = Discriminator(is_training=True, config=config, reuse = False, d_arch=FLAGS.d_arch)
        mDesFake = Discriminator(is_training=True, config=config, probs=mGenGAN.gumbels_softmax, reuse = True, d_arch=FLAGS.d_arch)        
        tf.summary.scalar("Training Loss", mGenLM.lm_cost)
        tf.summary.scalar("Learning Rate", mGenLM.lr)

    with tf.name_scope("Valid"):
      with tf.variable_scope("Model", initializer=initializer):
        mvalidGen = Generator(is_training=False, config=config, reuse=True, mode="LM")
        tf.summary.scalar("Validation Loss", mvalidGen.lm_cost)

    with tf.name_scope("Test"):
      with tf.variable_scope("Model", initializer=initializer):
        mtestGen = Generator(is_training=False, config=eval_config, reuse=True, mode="LM")
        
    # Create GAN losses
    dLossReal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(mDesReal.logit),
                                                                       logits=mDesReal.logit), name="dLossReal")
    dLossFake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(mDesFake.logit),
                                                                       logits=mDesFake.logit), name="dLossFake")
    dLoss = tf.add(dLossReal, dLossFake, name="dLoss")
    gLoss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(mDesFake.logit),
                                                                   logits=mDesFake.logit), name="gLoss")
    
    # Get trainable vars for discriminator and generator
    tVars = tf.trainable_variables()
    dVars = [var for var in tVars if var.name.startswith('Model/Discriminator')]
    gVars = [var for var in tVars if var.name.startswith('Model/Generator')]
    
    # Clipping gradient
    gLossGrads, _ = tf.clip_by_global_norm(tf.gradients(gLoss, gVars), 
                                           config.max_grad_norm)
    dLossGrads, _ = tf.clip_by_global_norm(tf.gradients(dLoss, dVars), 
                                           config.max_grad_norm)
    gPerplexityGrads, _ = tf.clip_by_global_norm(tf.gradients(mGenLM.lm_cost, gVars), 
                                           config.max_grad_norm)
    
    # Create optimizers
    optimizer = tf.train.AdamOptimizer(learning_rate=config.gan_learning_rate)
    
    gLossOpt = optimizer.apply_gradients(zip(gLossGrads, gVars),
                                     global_step=tf.contrib.framework.get_or_create_global_step())
    gPerplexityOpt = optimizer.apply_gradients(zip(gPerplexityGrads, gVars),
                                     global_step=tf.contrib.framework.get_or_create_global_step())
    gOptDual = tf.group(gPerplexityOpt, gLossOpt)
    dOpt = optimizer.apply_gradients(zip(dLossGrads, dVars),
                                     global_step=tf.contrib.framework.get_or_create_global_step())

    saver = tf.train.Saver(name='saver', write_version=tf.train.SaverDef.V2)
    sv = tf.train.Supervisor(logdir=FLAGS.save_path, save_model_secs=0, save_summaries_secs=0, saver=saver)

    old_valid_perplexity = 10000000000.0
    #sessconfig = tf.ConfigProto(allow_soft_placement=True)
    #sessconfig.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with sv.managed_session() as session:
      if FLAGS.sample_mode:
        
        # Load ngrams
        with open(os.path.join(FLAGS.data_path, FLAGS.file_prefix + ".test.txt")) as f:
            lines = f.readlines()   
        unigrams, bigrams, trigrams, quadgrams, quintagrams, sextagrams = get_grams(lines)

        while True:
          inpt = raw_input("Enter your sample prefix: ")
          cnt = int(raw_input("Sample size: "))

          if config.is_char_model:
            seed_for_sample = [c for c in inpt.replace(' ', '_')]
          else:
            seed_for_sample = inpt.split()
            
          tot_unigrams = 0.0
          tot_bigrams = 0.0
          tot_trigrams = 0.0
          tot_quadgrams = 0.0
          tot_quintagrams = 0.0
          tot_sextagrams = 0.0
          max_sum_ngrams = 0.0
          best_ngrams_sentence = None
          max_sextagrams = 0.0
          best_sextagram_sentence = "< No sextagram sentence was generated >"
            
          N = 1000
          
          for ii in range(N):
              samples = str(inpt) + " " + str(pretty_print(do_sample(session, mtestGen, [word_to_id[word] for word in seed_for_sample], cnt), config.is_char_model, id_2_word))
              samples_unigrams, samples_bigrams, samples_trigrams, samples_quadgrams, samples_quintagrams, samples_sextagrams = get_grams([samples])
              
              # Compute current score
              cur_unigrms = percentage_real(samples_unigrams, unigrams)
              cur_bigrms = percentage_real(samples_bigrams, bigrams)
              cur_trigrms = percentage_real(samples_trigrams, trigrams)
              cur_quadgrms = percentage_real(samples_quadgrams, quadgrams)
              cur_quintagrms = percentage_real(samples_quintagrams, quintagrams)
              cur_sextagrms = percentage_real(samples_sextagrams, sextagrams)
              
              # Add to total sum
              tot_unigrams += cur_unigrms
              tot_bigrams += cur_bigrms
              tot_trigrams += cur_trigrms
              tot_quadgrams += cur_quadgrms
              tot_quintagrams += cur_quintagrms
              tot_sextagrams += cur_sextagrms
              
              sum_ngrams = cur_unigrms + cur_bigrms + cur_trigrms + cur_quadgrms + cur_quintagrms + cur_sextagrms
              
              # Save sentence with max ngrams sum
              if (sum_ngrams > max_sum_ngrams):
                  max_sum_ngrams = sum_ngrams
                  best_ngrams_sentence = samples
                  
              # Save sentence with best sextagram score
              if (cur_sextagrms > max_sextagrams):
                  max_sextagrams = cur_sextagrms
                  best_sextagram_sentence = samples
              
              # Print sentence every 10 iterations
              if (ii % 10 == 0):
                  print (samples + ". Sum nGrams: " + str(sum_ngrams))
          
          print ("")
          print ("Averaging over " + str(N) + " iterations:")
          print ("----------------------------------------")
          print ("Unigrams %.3f" % (tot_unigrams/N))
          print ("Bigrams %.3f" % (tot_bigrams/N))
          print ("Trigrams %.3f" % (tot_trigrams/N))
          print ("Quadgrams %.3f" % (tot_quadgrams/N))
          print ("Quintagrams %.3f" % (tot_quintagrams/N))
          print ("Sextagrams %.3f" % (tot_sextagrams/N))
          print ("Best nGrams sum sentence: " + best_ngrams_sentence)
          print ("Best sextagram sentence: " + best_sextagram_sentence)
          
      if (FLAGS.train_gan or FLAGS.train_lm):
          for i in range(config.max_max_epoch):
    
            print("Seed: %s" % pretty_print([word_to_id[x] for x in seed_for_sample], config.is_char_model, id_2_word))
            print("Sample: %s" % pretty_print(do_sample(session, mtestGen, [word_to_id[word] for word in seed_for_sample],
                                                        max(5 * (len(seed_for_sample) + 1), 10)), config.is_char_model, id_2_word))
    
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            mGenLM.assign_lr(session, config.learning_rate * lr_decay)
            if (FLAGS.train_lm and not FLAGS.train_gan):
                print("Epoch: %d Learning rate: %.3f" % ((i + 1), session.run(mGenLM.lr)))
                train_perplexity = run_lm_epoch(session, 
                                                mGenLM, 
                                                train_data, 
                                                is_train=True, 
                                                verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            elif (FLAGS.train_gan and not FLAGS.train_lm):
                print("Epoch: %d" % ((i + 1)))
                dLosReal, dLosFake, dLos, gLos = run_gan_epoch(session, 
                                                               mDesReal,
                                                               train_data,
                                                               config,
                                                               dLossReal=dLossReal,
                                                               dLossFake=dLossFake,
                                                               dLoss=dLoss,
                                                               gLoss=gLoss,
                                                               dOpt=dOpt,
                                                               gOpt=gLossOpt,
                                                               is_train=True, 
                                                               verbose=True)
            elif (FLAGS.train_gan and FLAGS.train_lm):
                print("Epoch: %d" % ((i + 1)))
                dLosReal, dLosFake, dLos, gLos, train_perplexity = run_dual_epoch(session,
                                                                                  mDesReal,
                                                                                  mGenLM,
                                                                                  train_data,
                                                                                  config,
                                                                                  dLossReal=dLossReal,
                                                                                  dLossFake=dLossFake,
                                                                                  dLoss=dLoss,
                                                                                  gLoss=gLoss,
                                                                                  dOpt=dOpt,
                                                                                  gOpt=gOptDual,
                                                                                  epoch_num=i,
                                                                                  is_train=True,
                                                                                  verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_lm_epoch(session, mvalidGen, valid_data)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
            if valid_perplexity < old_valid_perplexity:
              if (FLAGS.save_embeddings):
                  np.save(os.path.join(FLAGS.data_path, FLAGS.file_prefix + "_embeddings"), 
                          session.run(mGenLM.embedding))
              old_valid_perplexity = valid_perplexity
              sv.saver.save(session, FLAGS.save_path, i)
            elif valid_perplexity >= 1.3*old_valid_perplexity:
              if len(sv.saver.last_checkpoints)>0:
                sv.saver.restore(session, sv.saver.last_checkpoints[-1])
              break
            else:
              if len(sv.saver.last_checkpoints)>0:
                sv.saver.restore(session, sv.saver.last_checkpoints[-1])
              lr_decay *=0.5

      test_perplexity = run_lm_epoch(session, mtestGen, test_data)
      print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
  tf.app.run()
