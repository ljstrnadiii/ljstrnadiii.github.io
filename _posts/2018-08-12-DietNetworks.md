---
layout: post
title:  "Diet Networks: Neural Networks and the p >> n problem "
date:   2018-08-12 17:42:13
categories: jekyll update
---

# *Diet Networks: Thin Parameters for Fat Genomics*[^fn1]
*Diet Networks* is a deep learning approach to predicting ancestry using genomic data. The number of free parameters in a neural network depends on the input dimension. The dimension of genomic data tends to be greater than the number of observations by three orders of magnitude. The model proposes an alternative approach to a fully connected network that reduces the number of free parameters significantly. 

Summary:
* Discuss Neural Networks and the Deep Learning
* Discuss genomic data and motivate the approach of *Diet Networks*
* Discuss the *Diet Network* architecture
* Discuss the TensorFlow implementation and results

## Neural Network and Deep Learning

* Neural Networks are represented as graphical structures

![Nets_ex](https://github.com/ljstrnadiii/ljstrnadiii.github.io/blob/master/images/net_ex.png?raw=true)

* The weights, $$w_i$$, are the free parameters and are learned through maximum likelihood estimation and back propagation. 
* This structure can be used to represent: Linear Regression, Multivariate Regression, Binomial Regression, Softmax Regression

![neuron](https://raw.githubusercontent.com/ljstrnadiii/genomicdl/master/images/neuron.png)
  
* Nodes following the input layer are computed with an activation function 
 

## What about the notion of Deep Learning?
* Adding hidden layers allows the model to learn a 'deeper' representation.
* __The Universal Approximation Theorem__:  a network with two hidden layers and non-linear activation functions can approximate any continuous function over a compact subset of $$R^n$$. 

![hidden](https://github.com/ljstrnadiii/ljstrnadiii.github.io/blob/master/images/neth_ex.png?raw=true)

* The parameters of the model can be represented as matrices.

## Representation Learning
* We want to learn a new representation of the data such that the new representations are linear in this new space.

### Example:
![example](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/img/simple2_0.png)
 (Image above borrowed from [here](http://colah.github.io/))

* Non-linear activation functions allow the model to learn this discriminating function as a linear function in a new feature space.

![example2](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/img/simple2_1.png)
(Image above borrowed from [here](http://colah.github.io/))

* Nodes in the hidden layers with non-linear activation functions are represented as $$h_j = \phi(x^t w_{(j, \cdot)})$$ where $$\phi$$ is the non-linear activation function.
* The new representation of $$\mathbf{x}$$ is then represented as $$\mathbf{h} = \phi(x^t \mathbf{W})$$.
* The algorithm essentially explores weight matrices, $$\mathbf{W}$$, that are in the path of gradient descent. 
* These weight matrices contruct the hypothesis space of functions considered in the function approximation task.

## Convolutional Layers
The beginning of "Deep" learning started with convolutional neural networks. The main idea is to convolve a single neural network about an image or audio. Navigate [here for arithmetic](http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic) or [here for visualization](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/). 

![conv](http://deeplearning.net/software/theano/_images/numerical_no_padding_no_strides.gif)
(Image borrowed from [here](http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic))

* Demonstrates the convolving of a kernel or neural network about the larger blue image to generate the "down-sampled" output in green.

![matrix](http://deeplearning.net/software/theano/_images/math/fbbfeb72879906b56649abed691e81b92fdc0a41.png)
(Image borrowed from [here](http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic)

* Expresses how a convolutional layer can be represented by a matrix. Notice the reduction in learnable parameters.

Unfortunately, genomic data does not have an obvious relationship with neighboring entries in its sequence like image or audio data.  

## Genomic Data
* The 1000 genomes project released the largest genomic data set among 26 different populations. 
* The data are roughly 150,000 single nucleotide polymorphisms (SNPs) for roughly 2500 people. 
* SNPs are essentially genetic variations of nucleotides that occur at a significant frequency between populations. 
* The goal is to classify the ancestry of an individual based on this SNP data. 

## Diet Networks Structure
* *Diet Networks* proposes a fully connected network with two auxiliary networks.
* The main use of the auxiliary network is to predict the weights of the first layer in the discriminative network. 

![DietNet](https://raw.githubusercontent.com/ljstrnadiii/ljstrnadiii.github.io/master/images/nets.png)
(Image taken from *Diet Networks*[^fn1]*)

* A fully connected network with $$p$$ dimensional data will have a $$(p \times n_h)$$ weight matrix in the first layer of the discriminative network. 
* If $$n_h=100$$, then we have 15,000,000 free parameters! 
* The method proposed to predict the weight matrix will reduce this number significantly.

### Auxiliary Network for Encoding

* The Auxiliary network for encoding predicts the weight matrix in the first layer of the discriminative network.
* __note__:
	* $$X$$ is of size $$(n \times p)$$
	* $$X^T$$ is of size $$(p \times n)$$
	* Let hidden layers have $$n_h$$ number of units
	* The first layer of the discrminative network is represented by the weight matrix, $$W_e$$, which is $$(p \times n_h)$$. 

* The first layer in the auxiliary network has a weight matrix, $$W_e'$$, with size $$(n \times n_h)$$.
* Then the output of the auxiliary network $$X^TW_e'=W_e$$.
* $$W_e$$ has size $$(p \times n) \times (n \times n_h) -> (p \times n_h)$$. 
* Thus, $$W_e$$ is the appropriate size for the first layer in the discriminate network.
* The final number of learnable parameters to construct $$W_e$$ is $$n \times n_h$$

### Auxiliary Network for Decoding
* The same thing is happening for the decoding auxiliary network.
* __note__:	
	* $$W_d = W_e$$ which implies the transpose of $$W_d$$ gives a shape $$n_h \times n$$.
	* The output of the first MLP layer, $$H$$, in the discriminative is $$p \times n_h$$.
	* Thus, $$\hat{X} = HW_d^T$$ gives $$(p \times n_h) \times (n_h \times n) -> (p \times n)$$.
	* The reconstruction is used because it gives better results and helps with gradient flow. 

### The Embedding Layer

* This implementation focuses on the histogram embedding. 
* The histogram embedding is generated by calculating the frequency of each possible value {0,1,2} for each class {1,...,26} accross each SNP {1,...,$$p$$}. 
* This information is contained in a $$(p \times 78)$$ matrix since 3 input types $$\times$$ 26 classes gives 78.
* This embedding is the input to a hidden layer which has $$n_h$$ nodes. 
* Therefore, we will have a $$(78 \times n_h)$$ weight matrix to learn, but the corresponding output will be $$(p \times n_h)$$. 

## TensorFlow Implementation and Results

* My TensorFlow implementation can be found [here](https://github.com/ljstrnadiii/DietNet). 
* The goal is to replicate the results of the paper. 

* They provide information on the model such as
	* the number of hidden units and hidden layers
	* norm constraints on the gradients
	* using an adaptive learning rate stochastic gradient descent optimizer

* The paper does not specify 
	* exactly how they regularize the parameters 
	* if they used batch norm
	* if they used drop out
	* which activation functions were used 
	* how they initialized the weights of the hidden layers 
	* or which specific optimizers were used 

* The goal of this implementation is to be specific about the regularization, weight initialization, and optimizers used.

### Regularization
Regularization is a way of preventing our model from overfitting. It helps decrease the generalization error. 

* The paper specifies that they limit the norm of the gradients (gradient clipping).

* This implementation uses the following regularization techniques:
	* L2 norm on each matrix matrix (like ridge regression)
	* gradient clipping (only back propagate when gradient is less than threshold)
	* weight initialization (use distribution with mean of zero and small variance)

### Batch Norm
* A batch is a subset of data used for back propagation.
* Batch norm normalizes each batch when performing forward pass to calculate error.
* Prevents model parameters from drifting as a cause of scale issues. 
* This problem is known as covariate shift

### Drop out
* Drop out is the process of randomly turning off neurons in the model.
* It allows each neuron the opportunity to "vote" and prevents a subset of neurons from taking over.
* It is mathematically equivalent to ensemble learning and is computationally cheap.

### Activation Functions
* Each activation function has its own pros and cons.
* This implementation considers the tanh and relu non-linear activation functions.

### Optimizers
* *Diet Networks* simply specified they used an adaptive learning rate stochastic gradient descent back propagation learning algorithm.
* This implementation considers the ADAM and RMSprop optimizers in the model selection process.

### TensorFlow Implementation

* The following diagram illustrates the structure of this TensorFlow implementation

![Structure](https://github.com/ljstrnadiii/ljstrnadiii.github.io/blob/master/images/struct.png?raw=true)

*The left structure represent the auxiliary network. The right structure represents the discriminative network.*

* Everywhere there is a `act_fun` or `w_init` is left open for model selection. 

### Model Selection

TensorFlow has a feature called tensorboard which helps visualize learning. Tensorboard is a webapp that displays specified summary statistics. In order to perform model selection, many models are constructed. 

__Models Considered__:
* Weight initialization using the Normal and Uniform distribution with standard deviation of .1 and .01
* tanh and relu activation functions
* Adam and RMSprop optimizers
* learning rates of .001 and .0001 

![results](https://github.com/ljstrnadiii/ljstrnadiii.github.io/blob/master/images/results.png?raw=true)
*Test set accuracy of the 32 models*

The optimal model achieves about 93% accuracy which matches with the results o *Diet Networks*.

















 [^fn1]: Romero, Adriana, et al. "Diet Networks: Thin Parameters     for Fat Genomic." arXiv preprint arXiv:1611.09340 (2016)
