---
layout: post
title:  "Asymmetric Loss Functions: How and Why in TensorFlow"
date:   2018-09-07 17:42:13
categories: jekyll update
---

# Asymmetric Loss Functions

This short blog is intended to be a reference for myself, but also to help explain why we would ever want a asymmetric loss function. In machine learning, we are never able to perfectly model a natural process. We can only ever accept the best model as an estimate. What is that famous quote that encapsulates this concept? It was Box who said "*all models are wrong, but some are useful.*"

Well, what makes a model more useful? What is the learning process that enforces a particular structure to be learned? The __loss function__. The loss function is analagous to the objective of the machine learning problem. The simple point that always nails it home for me is that the machine learning algorithm might train and reduce the loss function to zero, but does that mean it will perfectly represent the underlying natural process? No.

Thus, in order to be a machine learning expert and to craft the machine learning algorithm to your specific problem's needs, we can craft a better loss function!

## Context/Need
I ran into a problem where I wanted to penalize underpredictions of a regression model. Basically, we were afraid that if we underpredict this value for a user that they would default back to their own self declared overestimate of what they actually need. For brief context, the model is built to suggest to a user what energy needs they use as opposed to letting them self declare much more than they typically use. 

In order to do this, we are going to have to use a gradient descent approach. So, let's use tensorflow since it offers automatic differentiation, simplicity in defining loss functions, and simplicity in specifying optimizers to perform gradient descent.


```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
```

## Let's Make some data


```python
f = lambda x: x + np.random.normal(0,.1)
X_data = np.arange(0,100,.4) / 100
y_data = [f(x) for x in X_data]
```


```python
# plot
plt.figure(figsize=(20,10))
plt.plot(X_data,y_data, 'bo')
plt.show()
```


![png](https://raw.githubusercontent.com/ljstrnadiii/ljstrnadiii.github.io/master/images/2018-09-07-asymmetric_loss_function_4_0.png)


## The Loss Function

Let's use the typical squared error function, but rotated a bit. The function is 

$$l(x,y) = (x-y)^2$$

and transformed it could be

$$l(x,y,a) = (x-y)^2 * (sign(x-y)+a)^2$$



```python
l = lambda a: ((a)**2)*(np.sign(a)-.5)**2
c = lambda a: ((a)**2)
```


```python
A = np.arange(-3,3,.01)
l_data = [l(x) for x in A]
c_data = [c(x) for x in A]
```


```python
plt.figure(figsize=(20,10))
plt.plot(A, l_data,'r')
plt.plot(A, c_data,'b')

plt.xlim(-3,3)
plt.ylim(0,10)
plt.show()
```


![png](https://github.com/ljstrnadiii/ljstrnadiii.github.io/blob/master/images/2018-09-07-asymmetric_loss_function_8_0.png?raw=true)


So, we see that the new loss function in red will penalize errors that are less than zero greater than the errors greater than zero.

## Let's build a model

It is pretty straight forward to extend this to higher dimensional data, multivariate regression, or even to build a neural network with many hidden layers and nonlinear activation functions. 

For now, we are just going to demonstrate the asymmetric loss function in the simple regression setting. 


```python
# specify dimensions of the data
x_dim = [1,1]
y_dim = [1,1]

X = tf.placeholder("float64", shape=x_dim) # create symbolic variables
Y = tf.placeholder("float64",shape=y_dim)

# create your firt layer weights
w = tf.Variable(tf.zeros(
[x_dim[1],y_dim[1]],
dtype=tf.float64,
name="coef"
))

# create the bias
b = tf.Variable(tf.zeros(
y_dim,
dtype=tf.float64,
name="intcp"
))

y_model = tf.matmul(X, w) + b

cost = tf.pow(y_model-Y, 2) # use sqr error for cost function
def acost(a): return tf.pow(y_model-Y, 2) * tf.pow(tf.sign(y_model-Y) + a, 2)

#train_op = tf.train.AdamOptimizer().minimize(cost)
train_op1 = tf.train.AdamOptimizer().minimize(cost)

train_op2 = tf.train.AdamOptimizer().minimize(acost(-0.5))

sess = tf.Session()

# Run the asymmtric function
init = tf.global_variables_initializer()
sess.run(init)

for i in tqdm_notebook(range(500)):
    for (xi, yi) in zip(X_data, y_data): 
        xi = np.reshape(xi, x_dim)
        yi = np.reshape(yi, y_dim)
        sess.run(train_op2, feed_dict={X: xi, Y: yi})
        
# get the preds
x_t = np.reshape(X_data, [len(X_data),1])
preds_a = sess.run(tf.matmul(x_t,w)+b)


# Run the symmtric function
init = tf.global_variables_initializer()
sess.run(init)

for i in tqdm_notebook(range(500)):
    for (xi, yi) in zip(X_data, y_data): 
        xi = np.reshape(xi, x_dim)
        yi = np.reshape(yi, y_dim)
        sess.run(train_op1, feed_dict={X: xi, Y: yi})
    
# get the preds
x_t = np.reshape(X_data, [len(X_data),1])
preds = sess.run(tf.matmul(x_t,w)+b)

```


## Let's plot the predictions


```python
# plot
plt.figure(figsize=(20,10))
plt.plot(X_data,y_data, 'bo')
plt.plot(X_data,preds, 'b')
plt.plot(X_data,preds_a, 'r')

plt.show()
```


![png](https://github.com/ljstrnadiii/ljstrnadiii.github.io/blob/master/images/2018-09-07-asymmetric_loss_function_13_0.png?raw=true)


And voila! We see that the model slightly overpredicts the response. 
