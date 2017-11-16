

## Goal

- Component in Tensorflow
  - RNN: LSTM / GRU ([RNN](./ecnunlp-tensorflow-tutorial-ii.ipynb), [slides](./cs224n-2017-rnn.pdf))
  - Attention Mechanism
  - GAN ([GAN](./gan.ipynb), [DCGAN](./dcgan.ipynb)) 

Next: 
- Pointer-Generator Networks
- Reinforcement Learning


## Recap
- data
    - [784] -> [10]

- model [LR/SVM/etc]
  - train
  - predict

- evaluation

## Recap (Cont.)

- Homework: Text Classification Task
  - [model](https://github.com/rgtjf/tf-classification/blob/master/src/models/NBoW.py)
  - [data](https://github.com/rgtjf/tf-classification/blob/master/src/data.py)
  - [main](https://github.com/rgtjf/tf-classification/blob/master/src/main.py)

- EASY to change to LSTM, HOW?

```python
 def BiLSTM(input_x, input_x_len, hidden_size, num_layers=1, dropout_rate=None, return_sequence=True):
    """
    BiLSTM Layer
    Args:
      input_x: [batch, sent_len, emb_size]
      input_x_len: [batch, ]
      hidden_size: int
      num_layers: int
      dropout_rate: float
      return_sequence: True/False
    Returns:
      if return_sequence=True:
          outputs: [batch, sent_len, hidden_size*2]
      else:
          output: [batch, hidden_size*2]
    """
    # cell = tf.contrib.rnn.GRUCell(hidden_size)
    cell_fw = tf.contrib.rnn.BasicLSTMCell(hidden_size)
    cell_bw = tf.contrib.rnn.BasicLSTMCell(hidden_size)

    if num_layers > 1:
        # Warning! Please consider that whether the cell to stack are the same
        cell_fw = tf.contrib.rnn.MultiRNNCell([cell_fw for _ in range(num_layers)])
        cell_bw = tf.contrib.rnn.MultiRNNCell([cell_bw for _ in range(num_layers)])

    if dropout_rate:
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=(1 - dropout_rate))
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=(1 - dropout_rate))

    b_outputs, b_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_x,
                                                            sequence_length=input_x_len, dtype=tf.float32)
    if return_sequence:
        outputs = tf.concat(b_outputs, axis=2)
    else:
        # states: [c, h]
        outputs = tf.concat([b_states[0][1], b_outputs[1][1]], axis=-1)
    return outputs

with tf.variable_scope("bilstm") as s:
    lstm_x = BiLSTM(embedded_x, self.input_x_len, self.lstm_size, return_sequence=True)
```


## Details

- BasicRNN / GRUCell / LSTMCell
  - [BasicRNN](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py#L293-L347)
  - [GRUCell](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py#L350-L441)
  - [LSTMCell](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py#L590-L811)

```python
class BasicRNNCell(RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, input_size=None, activation=tanh):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    with vs.variable_scope(scope or "basic_rnn_cell"):
      output = self._activation(
          _linear([inputs, state], self._num_units, True, scope=scope))
    return output, output
```

```python
class GRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None, activation=tanh):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or "gru_cell"):
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        r, u = array_ops.split(
            value=_linear(
                [inputs, state], 2 * self._num_units, True, 1.0, scope=scope),
            num_or_size_splits=2,
            axis=1)
        r, u = sigmoid(r), sigmoid(u)
      with vs.variable_scope("candidate"):
        c = self._activation(_linear([inputs, r * state],
                                     self._num_units, True,
                                     scope=scope))
      new_h = u * state + (1 - u) * c
    return new_h, new_h
```

```python
class BasicLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.
  The implementation is based on: http://arxiv.org/abs/1409.2329.
  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.
  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.
  For advanced models, please use the full LSTMCell that follows.
  """

  def __init__(self, num_units, forget_bias=1.0, input_size=None,
               state_is_tuple=True, activation=tanh):
    """Initialize the basic LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or "basic_lstm_cell"):
      # Parameters of gates are concatenated into one multiply for efficiency.
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)
      concat = _linear([inputs, h], 4 * self._num_units, True, scope=scope)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

      new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * sigmoid(o)

      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = array_ops.concat([new_c, new_h], 1)
      return new_h, new_state

```

## Attention
$\alpha_i = softmax_i q^T W p_i$
$o  = \sum_i \alpha_i p_i$

```python
def bilinear_attention(question_rep, passage_repres, passage_mask):
    """
    Attention bilinear
    ref: https://arxiv.org/pdf/1606.02858v2.pdf
         https://github.com/danqi/rc-cnn-dailymail/blob/master/code/nn_layers.py
    $\alpha_i = softmax_i q^T W p_i$
    $ o  = \sum_i \alpha_i p_i $

      Args:
        question_rep: [batch_size, hidden_size]
        passage_repres: [batch_size, sequence_length, hidden_size]
        passage_mask: [batch_size, sequence_length]
      Returns:
        passage_rep: [batch_size, hidden_size]
    """
    hidden_size = question_rep.get_shape()[1]
    # [hidden_size, hidden_size]
    W_bilinear = tf.get_variable("W_bilinear", shape=[hidden_size, hidden_size], dtype=tf.float32)

    # [batch_size, hidden_size]
    question_rep = tf.matmul(question_rep, W_bilinear)
    
    # [batch_size, 1, hidden_size]
    question_rep = tf.expand_dims(question_rep, 1)
    
    # [batch_size, seq_length]
    alpha = tf.nn.softmax(tf.reduce_sum(question_rep * passage_repres, axis=2))
    alpha = alpha * passage_mask
    alpha = alpha / tf.reduce_sum(alpha, axis=-1, keep_dims=True)

    # [batch_size, hidden_size]
    passage_rep = tf.reduce_sum(passage_repres * tf.expand_dims(alpha, axis=-1), axis=1)
    return passage_rep
```

## GAN


## Refs
- RNN / LSTM
    - *[lecture8](http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture8.pdf)
        - [html](http://web.stanford.edu/class/cs224n/lectures/vanishing_grad_example.html)
        - [ipynb](http://web.stanford.edu/class/cs224n/lectures/vanishing_grad_example.ipynb)
        - [notes](http://web.stanford.edu/class/cs224n/lecture_notes/cs224n-2017-notes5.pdf)
    - [lecture9](http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture9.pdf)
    - [LSTM backprop](https://wiseodd.github.io/techblog/2016/08/12/lstm-backprop/)
    - *[VanishingGradient]
    (https://github.com/harinisuresh/VanishingGradient/blob/master/Vanishing%20Gradient%20Example.ipynb)
    - [bcs-lstm](https://github.com/nicholaslocascio/bcs-lstm)
    - *[Recurrent nets and LSTM](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/lecture11.pdf)[University of Oxford]
    - [zhihu](https://www.zhihu.com/question/29411132)

- Pytorch
    - [DeepNLP](https://github.com/DSKSD/DeepNLP-models-Pytorch)
    - [RNN](https://nbviewer.jupyter.org/github/DSKSD/DeepNLP-models-Pytorch/blob/master/notebooks/06.RNN-Language-Model.ipynb)
    - [Dynamic-Memory-Network-for-Question-Answering](https://nbviewer.jupyter.org/github/DSKSD/DeepNLP-models-Pytorch/blob/master/notebooks/10.Dynamic-Memory-Network-for-Question-Answering.ipynb)

- Attention
    - [A Thorough Examination]https://arxiv.org/pdf/1606.02858v2.pdf
    - [Theano]https://github.com/danqi/rc-cnn-dailymail/blob/master/code/nn_layers.py


- [GAN]
    - []()