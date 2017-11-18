# ecnunlp-tensorflow-tutorial

Tensorflow Tutorial, especially in NLP.

- ppt, paper, code

## Refs
- http://adventuresinmachinelearning.com/python-tensorflow-tutorial/
- https://www.tensorflow.org/tutorials/layers
- [TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples)


## Outline

## How to install
- [Install CUDA](https://www.tensorflow.org/install/install_linux)
- install tensorflow
  ```bash
  pip install tensorflow-gpu==1.0.1
  # or
  pip install tensorflow-gpu==1.2.0
  ```

## Background

- computation graph
- sess, feed_dict, functions

## Run a model
  - CNN/LSTM/attention
  - objective function
  - seq2seq
  
## Hyper-parameters
  - optimizer
  - learning rate
  - dropout

## Homework
  - text classifcation task

## More
  - reinforcement learning
  - gan
  - [resnet](https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py)

## Pytorch
  - [pytorch nlp](http://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html)
  

---
Update 2017.10.30
## Homework
- [tf-classifcation](https://github.com/rgtjf/tf-classification)


--- 
Update 2017.11.18
## Lecture 1
- Tensorflow, Computation Graph, Placeholder, Variables, Operations
- data(x, y), model(train, predict), main(feed_dict, session)
- broadcast, conv2d
- Tensorflow in NLP, word embeddings, padding, mask
- Task, nlpcc2017\_news\_headline\_categorization

## Lecture 2
- BiLSTM (BasicLSTMCell, bidirectional\_dynamic\_rnn, outputs, state)
- RNN (vanish gradient) 
  - clipper
  - carefully init, sigmoid -> tanh/relu
  - add memory cell: LSTM (input gate, forget gate, output gate), GRU (reset gate, update gate)
- Attention 
- GAN
  - generator, discriminator
