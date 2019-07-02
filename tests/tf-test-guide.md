# Tensorflow Code Testing Guide
This guide provides inspirations on testing machine learning code written in tensorflow.

## Tensorflow Testing Framework
Import the tensorflow built-in testing framework as follows:
```python
from tensorflow.python.platform import test
```
Or simply `import tensorflow as tf` and use `tf.test`. 
For the rest of this document, we will refer to the framework as `tf.test`.

### Introduction to `tf.test`
1. Recommended structure for the test code.

Similar to unittest, write your code as classes inherited from `tf.test.TestCase`. 
Then in the `main` scope, run `tf.test.main()`.

Example:
```python
import tensorflow as tf

class SeriousTest(tf.test.TestCase):
    def test_method_1(self):
        actual_value = expected_value = 1
        self.assertEqual(actual_value, expected_value)
    def test_method_2(self):
        actual_value = 1
        expected_value = 2
        self.assertNotEqual(actual_value, expected_value)

if __name__ == "__main__":
    tf.test.main()
```


2. Tensorflow unit test class [`tf.test.TestCase`](https://www.tensorflow.org/api_docs/python/tf/test/TestCase#test_session).

The `tf.test.TestCase` is built upon the standard python `unittest` with many additional methods.

Checkout the test code in [TF Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim/python/slim) for good examples.



### Useful Tips
#### 1. Use `test_session()`
When using `tf.test.TestCase`, you should almost always use 
the [`self.test_session()`](https://www.tensorflow.org/api_docs/python/tf/test/TestCase#test_session) method. 
It returns a TensorFlow Session for use in executing tests.

See example in the [AlexNet tests](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/alexnet_test.py) 
as part of the TF Slim code base.

#### 2. Test that a graph is built correctly

The method [`tf.test.assert_equal_graph_def`](https://www.tensorflow.org/api_docs/python/tf/test/assert_equal_graph_def) 
asserts two `GraphDef` objects are the same, ignoring versions and ordering of nodes, attrs, and control inputs.

Note: you can turn a `Graph` object to `GraphDef` object using `graph.as_graph_def()`.

Example:
```python
import tensorflow as tf

graph_actual = tf.Graph()
with graph_actual.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    variable_name = tf.Variable(initial_value=tf.random_normal(shape=[3, 5]))

graph_expected = tf.Graph()
with graph_expected.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    variable_name = tf.Variable(initial_value=tf.random_normal(shape=[3, 5]))

tf.test.assert_equal_graph_def(graph_actual.as_graph_def(), graph_expected.as_graph_def())
# test will pass.
```