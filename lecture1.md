

## Goal

1. MNIST
2. Homework: Text Classification Task


## Framework
- data
    - [784] -> [10]

- model [LR/SVM/etc]
  - train
  - predict

- evaluation



## Tensorflow

- Big Idea: express a numeric computation as a ``graph``
- Graph nodes are ``operations`` which have any number of inputs and outputs
- Graph edges are ``tensors`` which flow between nodes

$$ 
h = ReLU(Wx+b) 
$$

```python 
class Data(object):
    def __init__(self):
        pass
    
    def batch_iter(self):
        batch = None  
        yield batch

class Model(object):

    def __init__(self, params):
        # define the placeholder

        # define the variables

        # build the model graph
        # including: predict, loss, and train_op

    def train_model(sess, batch):
        feed_dict = {}
        to_return = {
            'train_op': self.train_op,
            'loss' : self. loss
        }
        return sess.run(to_return, feed_dict)
    
    def test_model(sess, batch):
        feed_dict = {}
        to_return = {
            'pred': self.pred
        }
        return sess.run(to_return, feed_dict)


if __name__ == '__main__':

    data = Data()
    train_data = data.train_data
    dev_data = data.dev_data

    model = Model()
    with tf.Session() as sess:
        for batch_data in train_data.batch_iter():
            loss = model.train_model(sess, batch_data)
    

        preds = []
        for batch_data in test_data.batch_iter():
            batch_preds = model.test_model(sess, batch_data)
            preds.append(batch_preds)

        acc = evalution(gold_labels, predicts)
```



## More
    - load embeddings
    - visualize
    - dropout in train/test
    - early stop
    - mask
    - attention


## TODO
- nlpcc data
  - precision: 
   ```
   cd /data/expr
   cp -r nlpcc/ YOUR_EXPR_PATH
  ```
- https://github.com/FudanNLP/nlpcc2017_news_headline_categorization
- 
```
.
|-- embed
├── char
│   ├── dev.txt
│   ├── id2tag.txt
│   ├── test.txt
│   ├── train.txt
│   └── vocab.txt
└── word
    ├── dev.txt
    ├── id2tag.txt
    ├── test.txt
    ├── train.txt
    ├── vocab.100k
    └── vocab.all
```
<!-- <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script> -->


- Refs:

- http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-tensorflow.pdf
- github.com/FudanNLP/nlpcc2017_news_headline_categorization
- https://github.com/aymericdamien/TensorFlow-Examples

