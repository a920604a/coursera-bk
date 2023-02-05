# Convolutional Neural Networks: Application

Welcome to Course 4's second assignment! In this notebook, you will:

- Create a mood classifer using the TF Keras Sequential API
- Build a ConvNet to identify sign language digits using the TF Keras Functional API

**After this assignment you will be able to:**

- Build and train a ConvNet in TensorFlow for a __binary__ classification problem
- Build and train a ConvNet in TensorFlow for a __multiclass__ classification problem
- Explain different use cases for the Sequential and Functional APIs

To complete this assignment, you should already be familiar with TensorFlow. If you are not, please refer back to the **TensorFlow Tutorial** of the third week of Course 2 ("**Improving deep neural networks**").

## Important Note on Submission to the AutoGrader

Before submitting your assignment to the AutoGrader, please make sure you are not doing the following:

1. You have not added any _extra_ `print` statement(s) in the assignment.
2. You have not added any _extra_ code cell(s) in the assignment.
3. You have not changed any of the function parameters.
4. You are not using any global variables inside your graded exercises. Unless specifically instructed to do so, please refrain from it and use the local variables instead.
5. You are not changing the assignment code where it is not required, like creating _extra_ variables.

If you do any of the following, you will get something like, `Grader not found` (or similarly unexpected) error upon submitting your assignment. Before asking for help/debugging the errors in your assignment, check for these first. If this is the case, and you don't remember the changes you have made, you can get a fresh copy of the assignment by following these [instructions](https://www.coursera.org/learn/convolutional-neural-networks/supplement/DS4yP/h-ow-to-refresh-your-workspace).

## Table of Contents

- [1 - Packages](#1)
    - [1.1 - Load the Data and Split the Data into Train/Test Sets](#1-1)
- [2 - Layers in TF Keras](#2)
- [3 - The Sequential API](#3)
    - [3.1 - Create the Sequential Model](#3-1)
        - [Exercise 1 - happyModel](#ex-1)
    - [3.2 - Train and Evaluate the Model](#3-2)
- [4 - The Functional API](#4)
    - [4.1 - Load the SIGNS Dataset](#4-1)
    - [4.2 - Split the Data into Train/Test Sets](#4-2)
    - [4.3 - Forward Propagation](#4-3)
        - [Exercise 2 - convolutional_model](#ex-2)
    - [4.4 - Train the Model](#4-4)
- [5 - History Object](#5)
- [6 - Bibliography](#6)

<a name='1'></a>
## 1 - Packages

As usual, begin by loading in the packages.


```python
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

%matplotlib inline
np.random.seed(1)
```

<a name='1-1'></a>
### 1.1 - Load the Data and Split the Data into Train/Test Sets

You'll be using the Happy House dataset for this part of the assignment, which contains images of peoples' faces. Your task will be to build a ConvNet that determines whether the people in the images are smiling or not -- because they only get to enter the house if they're smiling!  


```python
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 600
    number of test examples = 150
    X_train shape: (600, 64, 64, 3)
    Y_train shape: (600, 1)
    X_test shape: (150, 64, 64, 3)
    Y_test shape: (150, 1)


You can display the images contained in the dataset. Images are **64x64** pixels in RGB format (3 channels).


```python
index = 124
plt.imshow(X_train_orig[index]) #display sample training image
plt.show()
```


![png](output_7_0.png)


<a name='2'></a>
## 2 - Layers in TF Keras 

In the previous assignment, you created layers manually in numpy. In TF Keras, you don't have to write code directly to create layers. Rather, TF Keras has pre-defined layers you can use. 

When you create a layer in TF Keras, you are creating a function that takes some input and transforms it into an output you can reuse later. Nice and easy! 

<a name='3'></a>
## 3 - The Sequential API

In the previous assignment, you built helper functions using `numpy` to understand the mechanics behind convolutional neural networks. Most practical applications of deep learning today are built using programming frameworks, which have many built-in functions you can simply call. Keras is a high-level abstraction built on top of TensorFlow, which allows for even more simplified and optimized model creation and training. 

For the first part of this assignment, you'll create a model using TF Keras' Sequential API, which allows you to build layer by layer, and is ideal for building models where each layer has **exactly one** input tensor and **one** output tensor. 

As you'll see, using the Sequential API is simple and straightforward, but is only appropriate for simpler, more straightforward tasks. Later in this notebook you'll spend some time building with a more flexible, powerful alternative: the Functional API. 
 

<a name='3-1'></a>
### 3.1 - Create the Sequential Model

As mentioned earlier, the TensorFlow Keras Sequential API can be used to build simple models with layer operations that proceed in a sequential order. 

You can also add layers incrementally to a Sequential model with the `.add()` method, or remove them using the `.pop()` method, much like you would in a regular Python list.

Actually, you can think of a Sequential model as behaving like a list of layers. Like Python lists, Sequential layers are ordered, and the order in which they are specified matters.  If your model is non-linear or contains layers with multiple inputs or outputs, a Sequential model wouldn't be the right choice!

For any layer construction in Keras, you'll need to specify the input shape in advance. This is because in Keras, the shape of the weights is based on the shape of the inputs. The weights are only created when the model first sees some input data. Sequential models can be created by passing a list of layers to the Sequential constructor, like you will do in the next assignment.

<a name='ex-1'></a>
### Exercise 1 - happyModel

Implement the `happyModel` function below to build the following model: `ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Take help from [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) 

Also, plug in the following parameters for all the steps:

 - [ZeroPadding2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D): padding 3, input shape 64 x 64 x 3
 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 32 7x7 filters, stride 1
 - [BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization): for axis 3
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Using default parameters
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 1 neuron and a sigmoid activation. 
 
 
 **Hint:**
 
 Use **tfl** as shorthand for **tensorflow.keras.layers**


```python
# GRADED FUNCTION: happyModel

def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    model = tf.keras.Sequential([
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            
            ## Conv2D with 32 7x7 filters and stride of 1
            
            ## BatchNormalization for axis 3
            
            ## ReLU
            
            ## Max Pooling 2D with default parameters
            
            ## Flatten layer
            
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            
            # YOUR CODE STARTS HERE
            
        
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            tf.keras.layers.ZeroPadding2D( padding=(3, 3), input_shape = (64, 64, 3)),
            ## Conv2D with 32 7x7 filters and stride of 1
            tf.keras.layers.Conv2D(filters = 32, kernel_size = (7, 7), strides = (1, 1)),
            ## BatchNormalization for axis 3
            tf.keras.layers.BatchNormalization(axis = 3),
            ## ReLU
            tf.keras.layers.ReLU(),
            ## Max Pooling 2D with default parameters
            tf.keras.layers.MaxPool2D(),
            ## Flatten layer
            tf.keras.layers.Flatten(), 
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            tf.keras.layers.Dense(1, activation = 'sigmoid')
            
            # YOUR CODE ENDS HERE
        ])
    
    return model
```


```python
happy_model = happyModel()
# Print a summary for each layer
for layer in summary(happy_model):
    print(layer)
    
output = [['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
            ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform'],
            ['BatchNormalization', (None, 64, 64, 32), 128],
            ['ReLU', (None, 64, 64, 32), 0],
            ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid'],
            ['Flatten', (None, 32768), 0],
            ['Dense', (None, 1), 32769, 'sigmoid']]
    
comparator(summary(happy_model), output)
```

    ['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))]
    ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform']
    ['BatchNormalization', (None, 64, 64, 32), 128]
    ['ReLU', (None, 64, 64, 32), 0]
    ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid']
    ['Flatten', (None, 32768), 0]
    ['Dense', (None, 1), 32769, 'sigmoid']
    [32mAll tests passed![0m


#### Expected Output:

```
['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))]
['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform']
['BatchNormalization', (None, 64, 64, 32), 128]
['ReLU', (None, 64, 64, 32), 0]
['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid']
['Flatten', (None, 32768), 0]
['Dense', (None, 1), 32769, 'sigmoid']
All tests passed!
```

Now that your model is created, you can compile it for training with an optimizer and loss of your choice. When the string `accuracy` is specified as a metric, the type of accuracy used will be automatically converted based on the loss function used. This is one of the many optimizations built into TensorFlow that make your life easier! If you'd like to read more on how the compiler operates, check the docs [here](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).


```python
happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
```

It's time to check your model's parameters with the `.summary()` method. This will display the types of layers you have, the shape of the outputs, and how many parameters are in each layer. 


```python
happy_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    zero_padding2d (ZeroPadding2 (None, 70, 70, 3)         0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 64, 64, 32)        4736      
    _________________________________________________________________
    batch_normalization (BatchNo (None, 64, 64, 32)        128       
    _________________________________________________________________
    re_lu (ReLU)                 (None, 64, 64, 32)        0         
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 32, 32, 32)        0         
    _________________________________________________________________
    flatten (Flatten)            (None, 32768)             0         
    _________________________________________________________________
    dense (Dense)                (None, 1)                 32769     
    =================================================================
    Total params: 37,633
    Trainable params: 37,569
    Non-trainable params: 64
    _________________________________________________________________


<a name='3-2'></a>
### 3.2 - Train and Evaluate the Model

After creating the model, compiling it with your choice of optimizer and loss function, and doing a sanity check on its contents, you are now ready to build! 

Simply call `.fit()` to train. That's it! No need for mini-batching, saving, or complex backpropagation computations. That's all been done for you, as you're using a TensorFlow dataset with the batches specified already. You do have the option to specify epoch number or minibatch size if you like (for example, in the case of an un-batched dataset).


```python
happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)
```

    Epoch 1/10
    38/38 [==============================] - 4s 100ms/step - loss: 0.6975 - accuracy: 0.7883
    Epoch 2/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.2290 - accuracy: 0.8950
    Epoch 3/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.1644 - accuracy: 0.9417
    Epoch 4/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.2002 - accuracy: 0.9217
    Epoch 5/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.2347 - accuracy: 0.9283
    Epoch 6/10
    38/38 [==============================] - 4s 98ms/step - loss: 0.1419 - accuracy: 0.9467
    Epoch 7/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.0721 - accuracy: 0.9767
    Epoch 8/10
    38/38 [==============================] - 4s 98ms/step - loss: 0.0758 - accuracy: 0.9683
    Epoch 9/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.0699 - accuracy: 0.9750
    Epoch 10/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.3125 - accuracy: 0.8983





    <tensorflow.python.keras.callbacks.History at 0x7fb2811aa7d0>



After that completes, just use `.evaluate()` to evaluate against your test set. This function will print the value of the loss function and the performance metrics specified during the compilation of the model. In this case, the `binary_crossentropy` and the `accuracy` respectively.


```python
happy_model.evaluate(X_test, Y_test)
```

    5/5 [==============================] - 0s 22ms/step - loss: 1.8608 - accuracy: 0.6733





    [1.8607701063156128, 0.6733333468437195]



Easy, right? But what if you need to build a model with shared layers, branches, or multiple inputs and outputs? This is where Sequential, with its beautifully simple yet limited functionality, won't be able to help you. 

Next up: Enter the Functional API, your slightly more complex, highly flexible friend.  

<a name='4'></a>
## 4 - The Functional API

Welcome to the second half of the assignment, where you'll use Keras' flexible [Functional API](https://www.tensorflow.org/guide/keras/functional) to build a ConvNet that can differentiate between 6 sign language digits. 

The Functional API can handle models with non-linear topology, shared layers, as well as layers with multiple inputs or outputs. Imagine that, where the Sequential API requires the model to move in a linear fashion through its layers, the Functional API allows much more flexibility. Where Sequential is a straight line, a Functional model is a graph, where the nodes of the layers can connect in many more ways than one. 

In the visual example below, the one possible direction of the movement Sequential model is shown in contrast to a skip connection, which is just one of the many ways a Functional model can be constructed. A skip connection, as you might have guessed, skips some layer in the network and feeds the output to a later layer in the network. Don't worry, you'll be spending more time with skip connections very soon! 

<img src="images/seq_vs_func.png" style="width:350px;height:200px;">

<a name='4-1'></a>
### 4.1 - Load the SIGNS Dataset

As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.


```python
# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
```

<img src="images/SIGNS.png" style="width:800px;height:300px;">

The next cell will show you an example of a labelled image in the dataset. Feel free to change the value of `index` below and re-run to see different examples. 


```python
# Example of an image from the dataset
index = 9
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
```

    y = 4



![png](output_29_1.png)


<a name='4-2'></a>
### 4.2 - Split the Data into Train/Test Sets

In Course 2, you built a fully-connected network for this dataset. But since this is an image dataset, it is more natural to apply a ConvNet to it.

To get started, let's examine the shapes of your data. 


```python
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 1080
    number of test examples = 120
    X_train shape: (1080, 64, 64, 3)
    Y_train shape: (1080, 6)
    X_test shape: (120, 64, 64, 3)
    Y_test shape: (120, 6)


<a name='4-3'></a>
### 4.3 - Forward Propagation

In TensorFlow, there are built-in functions that implement the convolution steps for you. By now, you should be familiar with how TensorFlow builds computational graphs. In the [Functional API](https://www.tensorflow.org/guide/keras/functional), you create a graph of layers. This is what allows such great flexibility.

However, the following model could also be defined using the Sequential API since the information flow is on a single line. But don't deviate. What we want you to learn is to use the functional API.

Begin building your graph of layers by creating an input node that functions as a callable object:

- **input_img = tf.keras.Input(shape=input_shape):** 

Then, create a new node in the graph of layers by calling a layer on the `input_img` object: 

- **tf.keras.layers.Conv2D(filters= ... , kernel_size= ... , padding='same')(input_img):** Read the full documentation on [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D).

- **tf.keras.layers.MaxPool2D(pool_size=(f, f), strides=(s, s), padding='same'):** `MaxPool2D()` downsamples your input using a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window.  For max pooling, you usually operate on a single example at a time and a single channel at a time. Read the full documentation on [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D).

- **tf.keras.layers.ReLU():** computes the elementwise ReLU of Z (which can be any shape). You can read the full documentation on [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU).

- **tf.keras.layers.Flatten()**: given a tensor "P", this function takes each training (or test) example in the batch and flattens it into a 1D vector.  

    * If a tensor P has the shape (batch_size,h,w,c), it returns a flattened tensor with shape (batch_size, k), where $k=h \times w \times c$.  "k" equals the product of all the dimension sizes other than the first dimension.
    
    * For example, given a tensor with dimensions [100, 2, 3, 4], it flattens the tensor to be of shape [100, 24], where 24 = 2 * 3 * 4.  You can read the full documentation on [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten).

- **tf.keras.layers.Dense(units= ... , activation='softmax')(F):** given the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation on [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense).

In the last function above (`tf.keras.layers.Dense()`), the fully connected layer automatically initializes weights in the graph and keeps on training them as you train the model. Hence, you did not need to initialize those weights when initializing the parameters.

Lastly, before creating the model, you'll need to define the output using the last of the function's compositions (in this example, a Dense layer): 

- **outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)**


#### Window, kernel, filter, pool

The words "kernel" and "filter" are used to refer to the same thing. The word "filter" accounts for the amount of "kernels" that will be used in a single convolution layer. "Pool" is the name of the operation that takes the max or average value of the kernels. 

This is why the parameter `pool_size` refers to `kernel_size`, and you use `(f,f)` to refer to the filter size. 

Pool size and kernel size refer to the same thing in different objects - They refer to the shape of the window where the operation takes place. 

<a name='ex-2'></a>
### Exercise 2 - convolutional_model

Implement the `convolutional_model` function below to build the following model: `CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Use the functions above! 

Also, plug in the following parameters for all the steps:

 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 8 4 by 4 filters, stride 1, padding is "SAME"
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
 - **Conv2D**: Use 16 2 by 2 filters, stride 1, padding is "SAME"
 - **ReLU**
 - **MaxPool2D**: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 6 neurons and a softmax activation. 


```python
# GRADED FUNCTION: convolutional_model

def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    # Z1 = None
    ## RELU
    # A1 = None
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    # P1 = None
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    # Z2 = None
    ## RELU
    # A2 = None
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    # P2 = None
    ## FLATTEN
    # F = None
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    # outputs = None
    # YOUR CODE STARTS HERE
    
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tf.keras.layers.Conv2D( filters = 8 , kernel_size = (4, 4), strides=(1, 1), padding = 'SAME')(input_img)
    ## RELU
    A1 = tf.keras.layers.ReLU()(Z1)
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.keras.layers.MaxPool2D(pool_size=(8, 8), strides=8, padding = 'SAME')(A1)
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tf.keras.layers.Conv2D( filters = 16, kernel_size = (2, 2), strides=(1, 1), padding = 'SAME')(P1)
    ## RELU
    A2 = tf.keras.layers.ReLU()(Z2)
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=4, padding = 'SAME')(A2)
    ## FLATTEN
    F = tf.keras.layers.Flatten()(P2)
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    outputs = tf.keras.layers.Dense(6, activation='softmax')(F)
    
    # YOUR CODE ENDS HERE
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
```


```python
conv_model = convolutional_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()
    
output = [['InputLayer', [(None, 64, 64, 3)], 0],
        ['Conv2D', (None, 64, 64, 8), 392, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 64, 64, 8), 0],
        ['MaxPooling2D', (None, 8, 8, 8), 0, (8, 8), (8, 8), 'same'],
        ['Conv2D', (None, 8, 8, 16), 528, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 8, 8, 16), 0],
        ['MaxPooling2D', (None, 2, 2, 16), 0, (4, 4), (4, 4), 'same'],
        ['Flatten', (None, 64), 0],
        ['Dense', (None, 6), 390, 'softmax']]
    
comparator(summary(conv_model), output)
```

    Model: "functional_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 64, 64, 3)]       0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 64, 64, 8)         392       
    _________________________________________________________________
    re_lu_1 (ReLU)               (None, 64, 64, 8)         0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 8, 8, 8)           0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 8, 8, 16)          528       
    _________________________________________________________________
    re_lu_2 (ReLU)               (None, 8, 8, 16)          0         
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 2, 2, 16)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 64)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 6)                 390       
    =================================================================
    Total params: 1,310
    Trainable params: 1,310
    Non-trainable params: 0
    _________________________________________________________________
    [32mAll tests passed![0m


Both the Sequential and Functional APIs return a TF Keras model object. The only difference is how inputs are handled inside the object model! 

<a name='4-4'></a>
### 4.4 - Train the Model


```python
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)
```

    Epoch 1/100
    17/17 [==============================] - 2s 117ms/step - loss: 1.7924 - accuracy: 0.1880 - val_loss: 1.7929 - val_accuracy: 0.1833
    Epoch 2/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.7836 - accuracy: 0.2259 - val_loss: 1.7872 - val_accuracy: 0.2167
    Epoch 3/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7765 - accuracy: 0.3269 - val_loss: 1.7822 - val_accuracy: 0.3167
    Epoch 4/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7699 - accuracy: 0.3481 - val_loss: 1.7761 - val_accuracy: 0.3250
    Epoch 5/100
    17/17 [==============================] - 2s 101ms/step - loss: 1.7616 - accuracy: 0.3657 - val_loss: 1.7679 - val_accuracy: 0.3417
    Epoch 6/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7500 - accuracy: 0.4019 - val_loss: 1.7585 - val_accuracy: 0.3667
    Epoch 7/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7367 - accuracy: 0.4111 - val_loss: 1.7461 - val_accuracy: 0.4000
    Epoch 8/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7210 - accuracy: 0.4269 - val_loss: 1.7305 - val_accuracy: 0.3917
    Epoch 9/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7000 - accuracy: 0.4287 - val_loss: 1.7159 - val_accuracy: 0.3917
    Epoch 10/100
    17/17 [==============================] - 2s 101ms/step - loss: 1.6726 - accuracy: 0.4407 - val_loss: 1.6935 - val_accuracy: 0.4167
    Epoch 11/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.6389 - accuracy: 0.4491 - val_loss: 1.6664 - val_accuracy: 0.4167
    Epoch 12/100
    17/17 [==============================] - 2s 101ms/step - loss: 1.5996 - accuracy: 0.4509 - val_loss: 1.6356 - val_accuracy: 0.4167
    Epoch 13/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.5564 - accuracy: 0.4713 - val_loss: 1.5998 - val_accuracy: 0.4167
    Epoch 14/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.5107 - accuracy: 0.4824 - val_loss: 1.5658 - val_accuracy: 0.4167
    Epoch 15/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.4648 - accuracy: 0.4972 - val_loss: 1.5292 - val_accuracy: 0.4250
    Epoch 16/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.4178 - accuracy: 0.5130 - val_loss: 1.4943 - val_accuracy: 0.4250
    Epoch 17/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.3741 - accuracy: 0.5259 - val_loss: 1.4606 - val_accuracy: 0.4417
    Epoch 18/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.3332 - accuracy: 0.5370 - val_loss: 1.4297 - val_accuracy: 0.4583
    Epoch 19/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.2939 - accuracy: 0.5565 - val_loss: 1.3966 - val_accuracy: 0.4667
    Epoch 20/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.2576 - accuracy: 0.5676 - val_loss: 1.3671 - val_accuracy: 0.4917
    Epoch 21/100
    17/17 [==============================] - 2s 101ms/step - loss: 1.2232 - accuracy: 0.5898 - val_loss: 1.3396 - val_accuracy: 0.5083
    Epoch 22/100
    17/17 [==============================] - 2s 107ms/step - loss: 1.1919 - accuracy: 0.5991 - val_loss: 1.3081 - val_accuracy: 0.5167
    Epoch 23/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.1609 - accuracy: 0.6083 - val_loss: 1.2875 - val_accuracy: 0.5167
    Epoch 24/100
    17/17 [==============================] - 2s 101ms/step - loss: 1.1317 - accuracy: 0.6176 - val_loss: 1.2597 - val_accuracy: 0.5333
    Epoch 25/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.1044 - accuracy: 0.6287 - val_loss: 1.2391 - val_accuracy: 0.5500
    Epoch 26/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.0789 - accuracy: 0.6435 - val_loss: 1.2152 - val_accuracy: 0.5583
    Epoch 27/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.0542 - accuracy: 0.6565 - val_loss: 1.1953 - val_accuracy: 0.5750
    Epoch 28/100
    17/17 [==============================] - 2s 102ms/step - loss: 1.0315 - accuracy: 0.6676 - val_loss: 1.1735 - val_accuracy: 0.5583
    Epoch 29/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.0096 - accuracy: 0.6741 - val_loss: 1.1555 - val_accuracy: 0.5750
    Epoch 30/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.9890 - accuracy: 0.6824 - val_loss: 1.1344 - val_accuracy: 0.5667
    Epoch 31/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.9688 - accuracy: 0.6852 - val_loss: 1.1175 - val_accuracy: 0.5667
    Epoch 32/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.9498 - accuracy: 0.6880 - val_loss: 1.0999 - val_accuracy: 0.5750
    Epoch 33/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.9320 - accuracy: 0.6963 - val_loss: 1.0834 - val_accuracy: 0.5833
    Epoch 34/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.9151 - accuracy: 0.7009 - val_loss: 1.0686 - val_accuracy: 0.5917
    Epoch 35/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8987 - accuracy: 0.7065 - val_loss: 1.0532 - val_accuracy: 0.6000
    Epoch 36/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.8835 - accuracy: 0.7111 - val_loss: 1.0398 - val_accuracy: 0.6083
    Epoch 37/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8689 - accuracy: 0.7120 - val_loss: 1.0275 - val_accuracy: 0.6250
    Epoch 38/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8552 - accuracy: 0.7185 - val_loss: 1.0151 - val_accuracy: 0.6417
    Epoch 39/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8418 - accuracy: 0.7231 - val_loss: 1.0029 - val_accuracy: 0.6500
    Epoch 40/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8286 - accuracy: 0.7278 - val_loss: 0.9920 - val_accuracy: 0.6500
    Epoch 41/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8156 - accuracy: 0.7343 - val_loss: 0.9805 - val_accuracy: 0.6583
    Epoch 42/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8032 - accuracy: 0.7370 - val_loss: 0.9702 - val_accuracy: 0.6583
    Epoch 43/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7914 - accuracy: 0.7380 - val_loss: 0.9613 - val_accuracy: 0.6667
    Epoch 44/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7803 - accuracy: 0.7444 - val_loss: 0.9517 - val_accuracy: 0.6583
    Epoch 45/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7694 - accuracy: 0.7454 - val_loss: 0.9421 - val_accuracy: 0.6667
    Epoch 46/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.7588 - accuracy: 0.7500 - val_loss: 0.9349 - val_accuracy: 0.6667
    Epoch 47/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7490 - accuracy: 0.7509 - val_loss: 0.9246 - val_accuracy: 0.6667
    Epoch 48/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7389 - accuracy: 0.7556 - val_loss: 0.9174 - val_accuracy: 0.6583
    Epoch 49/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7290 - accuracy: 0.7574 - val_loss: 0.9097 - val_accuracy: 0.6500
    Epoch 50/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.7200 - accuracy: 0.7648 - val_loss: 0.9024 - val_accuracy: 0.6583
    Epoch 51/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7108 - accuracy: 0.7639 - val_loss: 0.8957 - val_accuracy: 0.6750
    Epoch 52/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.7019 - accuracy: 0.7685 - val_loss: 0.8878 - val_accuracy: 0.6750
    Epoch 53/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6928 - accuracy: 0.7704 - val_loss: 0.8812 - val_accuracy: 0.6750
    Epoch 54/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6846 - accuracy: 0.7713 - val_loss: 0.8736 - val_accuracy: 0.6750
    Epoch 55/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6764 - accuracy: 0.7722 - val_loss: 0.8659 - val_accuracy: 0.6750
    Epoch 56/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6687 - accuracy: 0.7741 - val_loss: 0.8577 - val_accuracy: 0.6833
    Epoch 57/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6610 - accuracy: 0.7759 - val_loss: 0.8505 - val_accuracy: 0.7000
    Epoch 58/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6536 - accuracy: 0.7787 - val_loss: 0.8433 - val_accuracy: 0.7083
    Epoch 59/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.6464 - accuracy: 0.7796 - val_loss: 0.8364 - val_accuracy: 0.7167
    Epoch 60/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6394 - accuracy: 0.7806 - val_loss: 0.8302 - val_accuracy: 0.7167
    Epoch 61/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6328 - accuracy: 0.7806 - val_loss: 0.8239 - val_accuracy: 0.7167
    Epoch 62/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6262 - accuracy: 0.7824 - val_loss: 0.8174 - val_accuracy: 0.7167
    Epoch 63/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6198 - accuracy: 0.7870 - val_loss: 0.8110 - val_accuracy: 0.7167
    Epoch 64/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6137 - accuracy: 0.7917 - val_loss: 0.8048 - val_accuracy: 0.7167
    Epoch 65/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.6080 - accuracy: 0.7991 - val_loss: 0.7987 - val_accuracy: 0.7167
    Epoch 66/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6022 - accuracy: 0.8000 - val_loss: 0.7931 - val_accuracy: 0.7167
    Epoch 67/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5962 - accuracy: 0.8019 - val_loss: 0.7876 - val_accuracy: 0.7250
    Epoch 68/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5903 - accuracy: 0.8019 - val_loss: 0.7821 - val_accuracy: 0.7250
    Epoch 69/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5845 - accuracy: 0.8037 - val_loss: 0.7766 - val_accuracy: 0.7250
    Epoch 70/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5792 - accuracy: 0.8046 - val_loss: 0.7711 - val_accuracy: 0.7250
    Epoch 71/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5739 - accuracy: 0.8093 - val_loss: 0.7652 - val_accuracy: 0.7250
    Epoch 72/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5686 - accuracy: 0.8074 - val_loss: 0.7599 - val_accuracy: 0.7250
    Epoch 73/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.5634 - accuracy: 0.8083 - val_loss: 0.7548 - val_accuracy: 0.7250
    Epoch 74/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5583 - accuracy: 0.8111 - val_loss: 0.7501 - val_accuracy: 0.7250
    Epoch 75/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5533 - accuracy: 0.8120 - val_loss: 0.7449 - val_accuracy: 0.7250
    Epoch 76/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5486 - accuracy: 0.8120 - val_loss: 0.7403 - val_accuracy: 0.7250
    Epoch 77/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5440 - accuracy: 0.8148 - val_loss: 0.7354 - val_accuracy: 0.7333
    Epoch 78/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5395 - accuracy: 0.8204 - val_loss: 0.7305 - val_accuracy: 0.7333
    Epoch 79/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.5351 - accuracy: 0.8204 - val_loss: 0.7257 - val_accuracy: 0.7333
    Epoch 80/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5311 - accuracy: 0.8222 - val_loss: 0.7206 - val_accuracy: 0.7333
    Epoch 81/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.5268 - accuracy: 0.8241 - val_loss: 0.7156 - val_accuracy: 0.7333
    Epoch 82/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5225 - accuracy: 0.8241 - val_loss: 0.7109 - val_accuracy: 0.7250
    Epoch 83/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5186 - accuracy: 0.8241 - val_loss: 0.7062 - val_accuracy: 0.7333
    Epoch 84/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5142 - accuracy: 0.8241 - val_loss: 0.7017 - val_accuracy: 0.7333
    Epoch 85/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.5099 - accuracy: 0.8259 - val_loss: 0.6969 - val_accuracy: 0.7417
    Epoch 86/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5061 - accuracy: 0.8269 - val_loss: 0.6925 - val_accuracy: 0.7417
    Epoch 87/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5021 - accuracy: 0.8278 - val_loss: 0.6884 - val_accuracy: 0.7417
    Epoch 88/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4982 - accuracy: 0.8278 - val_loss: 0.6841 - val_accuracy: 0.7500
    Epoch 89/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4943 - accuracy: 0.8296 - val_loss: 0.6800 - val_accuracy: 0.7583
    Epoch 90/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4905 - accuracy: 0.8315 - val_loss: 0.6762 - val_accuracy: 0.7583
    Epoch 91/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4868 - accuracy: 0.8324 - val_loss: 0.6723 - val_accuracy: 0.7583
    Epoch 92/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4830 - accuracy: 0.8333 - val_loss: 0.6682 - val_accuracy: 0.7583
    Epoch 93/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4793 - accuracy: 0.8352 - val_loss: 0.6645 - val_accuracy: 0.7583
    Epoch 94/100
    17/17 [==============================] - 2s 105ms/step - loss: 0.4757 - accuracy: 0.8370 - val_loss: 0.6608 - val_accuracy: 0.7583
    Epoch 95/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4719 - accuracy: 0.8398 - val_loss: 0.6572 - val_accuracy: 0.7667
    Epoch 96/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.4681 - accuracy: 0.8398 - val_loss: 0.6534 - val_accuracy: 0.7750
    Epoch 97/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4645 - accuracy: 0.8417 - val_loss: 0.6496 - val_accuracy: 0.7750
    Epoch 98/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4608 - accuracy: 0.8398 - val_loss: 0.6460 - val_accuracy: 0.7750
    Epoch 99/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4572 - accuracy: 0.8435 - val_loss: 0.6419 - val_accuracy: 0.7750
    Epoch 100/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.4537 - accuracy: 0.8463 - val_loss: 0.6385 - val_accuracy: 0.7750


<a name='5'></a>
## 5 - History Object 

The history object is an output of the `.fit()` operation, and provides a record of all the loss and metric values in memory. It's stored as a dictionary that you can retrieve at `history.history`: 


```python
history.history
```




    {'loss': [1.7924435138702393,
      1.783560872077942,
      1.7765305042266846,
      1.7699077129364014,
      1.7615914344787598,
      1.7499537467956543,
      1.7366644144058228,
      1.7209876775741577,
      1.6999585628509521,
      1.6725867986679077,
      1.6388627290725708,
      1.5996382236480713,
      1.5563796758651733,
      1.5106760263442993,
      1.4648396968841553,
      1.4177783727645874,
      1.3740649223327637,
      1.3331630229949951,
      1.2939426898956299,
      1.257556676864624,
      1.2231746912002563,
      1.1919180154800415,
      1.1608538627624512,
      1.1316982507705688,
      1.1044249534606934,
      1.0788804292678833,
      1.0542352199554443,
      1.0314995050430298,
      1.0096038579940796,
      0.9889760613441467,
      0.9687619805335999,
      0.9497972130775452,
      0.9319614768028259,
      0.9151018857955933,
      0.8987249135971069,
      0.88346266746521,
      0.8689302206039429,
      0.8552284240722656,
      0.8417896032333374,
      0.8285745978355408,
      0.8156384229660034,
      0.8032481670379639,
      0.791424572467804,
      0.7802746295928955,
      0.7694114446640015,
      0.7588050961494446,
      0.7489767670631409,
      0.7388911247253418,
      0.7290228605270386,
      0.7199593186378479,
      0.7108444571495056,
      0.7018937468528748,
      0.6927608251571655,
      0.6846256852149963,
      0.6764065027236938,
      0.6686760187149048,
      0.6610132455825806,
      0.6536001563072205,
      0.6463605761528015,
      0.6393910050392151,
      0.6327883005142212,
      0.6261999607086182,
      0.6197887063026428,
      0.6137460470199585,
      0.6079983115196228,
      0.6022248268127441,
      0.5961777567863464,
      0.5903176665306091,
      0.5845237374305725,
      0.5791613459587097,
      0.5739421844482422,
      0.568617582321167,
      0.5634231567382812,
      0.5582690238952637,
      0.5533314943313599,
      0.5486015677452087,
      0.5439605712890625,
      0.5394819378852844,
      0.5350850820541382,
      0.5310565233230591,
      0.5268325209617615,
      0.5225147604942322,
      0.5186436176300049,
      0.5141851902008057,
      0.509918212890625,
      0.5061379671096802,
      0.502117395401001,
      0.49817460775375366,
      0.49434593319892883,
      0.490531861782074,
      0.48679283261299133,
      0.4830121695995331,
      0.479267418384552,
      0.47571808099746704,
      0.47186124324798584,
      0.4681488573551178,
      0.4644794464111328,
      0.4608304798603058,
      0.45719385147094727,
      0.453677773475647],
     'accuracy': [0.18796296417713165,
      0.22592592239379883,
      0.32685184478759766,
      0.3481481373310089,
      0.36574074625968933,
      0.4018518626689911,
      0.41111111640930176,
      0.4268518388271332,
      0.4287036955356598,
      0.4407407343387604,
      0.44907405972480774,
      0.45092591643333435,
      0.4712963104248047,
      0.48240742087364197,
      0.4972222149372101,
      0.5129629373550415,
      0.5259259343147278,
      0.5370370149612427,
      0.5564814805984497,
      0.5675926208496094,
      0.5898148417472839,
      0.5990740656852722,
      0.6083333492279053,
      0.6175925731658936,
      0.6287037134170532,
      0.6435185074806213,
      0.6564815044403076,
      0.6675925850868225,
      0.6740740537643433,
      0.6824073791503906,
      0.6851851940155029,
      0.6879629492759705,
      0.6962962746620178,
      0.7009259462356567,
      0.7064814567565918,
      0.7111111283302307,
      0.7120370268821716,
      0.7185184955596924,
      0.7231481671333313,
      0.7277777791023254,
      0.7342592477798462,
      0.7370370626449585,
      0.7379629611968994,
      0.7444444298744202,
      0.7453703880310059,
      0.75,
      0.7509258985519409,
      0.7555555701255798,
      0.7574074268341064,
      0.7648147940635681,
      0.7638888955116272,
      0.7685185074806213,
      0.770370364189148,
      0.7712963223457336,
      0.7722222208976746,
      0.7740740776062012,
      0.7759259343147278,
      0.7787036895751953,
      0.779629647731781,
      0.7805555462837219,
      0.7805555462837219,
      0.7824074029922485,
      0.7870370149612427,
      0.7916666865348816,
      0.7990740537643433,
      0.800000011920929,
      0.8018518686294556,
      0.8018518686294556,
      0.8037037253379822,
      0.8046296238899231,
      0.8092592358589172,
      0.8074073791503906,
      0.8083333373069763,
      0.8111110925674438,
      0.8120370507240295,
      0.8120370507240295,
      0.8148148059844971,
      0.8203703761100769,
      0.8203703761100769,
      0.8222222328186035,
      0.8240740895271301,
      0.8240740895271301,
      0.8240740895271301,
      0.8240740895271301,
      0.8259259462356567,
      0.8268518447875977,
      0.8277778029441833,
      0.8277778029441833,
      0.8296296000480652,
      0.8314814567565918,
      0.8324074149131775,
      0.8333333134651184,
      0.835185170173645,
      0.8370370268821716,
      0.8398148417472839,
      0.8398148417472839,
      0.8416666388511658,
      0.8398148417472839,
      0.8435184955596924,
      0.8462963104248047],
     'val_loss': [1.7928569316864014,
      1.7872329950332642,
      1.7821850776672363,
      1.7760637998580933,
      1.767872929573059,
      1.7584731578826904,
      1.746147632598877,
      1.7304941415786743,
      1.715915560722351,
      1.6934664249420166,
      1.6663661003112793,
      1.6355799436569214,
      1.5998389720916748,
      1.5657931566238403,
      1.5292048454284668,
      1.4942902326583862,
      1.460564374923706,
      1.4297205209732056,
      1.396594524383545,
      1.3671128749847412,
      1.339618444442749,
      1.3080756664276123,
      1.2874596118927002,
      1.2596797943115234,
      1.239137053489685,
      1.2152061462402344,
      1.1952519416809082,
      1.1734607219696045,
      1.1554985046386719,
      1.1343910694122314,
      1.1175270080566406,
      1.0998896360397339,
      1.0834378004074097,
      1.0686312913894653,
      1.053240180015564,
      1.0397523641586304,
      1.027541995048523,
      1.0150666236877441,
      1.0028574466705322,
      0.9919806122779846,
      0.9804831147193909,
      0.9702123403549194,
      0.9613366723060608,
      0.9517168402671814,
      0.9421018362045288,
      0.9348896741867065,
      0.9245650172233582,
      0.9173726439476013,
      0.909696638584137,
      0.9023913145065308,
      0.8957013487815857,
      0.8877943158149719,
      0.8811617493629456,
      0.873639702796936,
      0.8659130930900574,
      0.8577252626419067,
      0.8504587411880493,
      0.8433040976524353,
      0.8364195227622986,
      0.8301950097084045,
      0.8238813877105713,
      0.8174384236335754,
      0.8110410571098328,
      0.8047798275947571,
      0.7987457513809204,
      0.7930806279182434,
      0.7876442074775696,
      0.7821404933929443,
      0.7765754461288452,
      0.7710636854171753,
      0.7652472257614136,
      0.7599445581436157,
      0.7547951340675354,
      0.7500922083854675,
      0.7449482083320618,
      0.7402698993682861,
      0.7354003190994263,
      0.7305153608322144,
      0.7256664037704468,
      0.7205526828765869,
      0.7156012058258057,
      0.7108787298202515,
      0.7062394618988037,
      0.7017480134963989,
      0.6968802213668823,
      0.6925303936004639,
      0.6883664727210999,
      0.6840667724609375,
      0.6800113916397095,
      0.6761543154716492,
      0.6722772121429443,
      0.6681548953056335,
      0.6645100116729736,
      0.6608374118804932,
      0.6571986675262451,
      0.6533861756324768,
      0.649583101272583,
      0.645971953868866,
      0.6418580412864685,
      0.6384853720664978],
     'val_accuracy': [0.18333333730697632,
      0.21666666865348816,
      0.3166666626930237,
      0.32499998807907104,
      0.34166666865348816,
      0.36666667461395264,
      0.4000000059604645,
      0.3916666805744171,
      0.3916666805744171,
      0.4166666567325592,
      0.4166666567325592,
      0.4166666567325592,
      0.4166666567325592,
      0.4166666567325592,
      0.42500001192092896,
      0.42500001192092896,
      0.4416666626930237,
      0.4583333432674408,
      0.46666666865348816,
      0.49166667461395264,
      0.5083333253860474,
      0.5166666507720947,
      0.5166666507720947,
      0.5333333611488342,
      0.550000011920929,
      0.5583333373069763,
      0.574999988079071,
      0.5583333373069763,
      0.574999988079071,
      0.5666666626930237,
      0.5666666626930237,
      0.574999988079071,
      0.5833333134651184,
      0.5916666388511658,
      0.6000000238418579,
      0.6083333492279053,
      0.625,
      0.6416666507720947,
      0.6499999761581421,
      0.6499999761581421,
      0.6583333611488342,
      0.6583333611488342,
      0.6666666865348816,
      0.6583333611488342,
      0.6666666865348816,
      0.6666666865348816,
      0.6666666865348816,
      0.6583333611488342,
      0.6499999761581421,
      0.6583333611488342,
      0.675000011920929,
      0.675000011920929,
      0.675000011920929,
      0.675000011920929,
      0.675000011920929,
      0.6833333373069763,
      0.699999988079071,
      0.7083333134651184,
      0.7166666388511658,
      0.7166666388511658,
      0.7166666388511658,
      0.7166666388511658,
      0.7166666388511658,
      0.7166666388511658,
      0.7166666388511658,
      0.7166666388511658,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7250000238418579,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7250000238418579,
      0.7333333492279053,
      0.7333333492279053,
      0.7416666746139526,
      0.7416666746139526,
      0.7416666746139526,
      0.75,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7583333253860474,
      0.7666666507720947,
      0.7749999761581421,
      0.7749999761581421,
      0.7749999761581421,
      0.7749999761581421,
      0.7749999761581421]}



Now visualize the loss over time using `history.history`: 


```python
# The history.history["loss"] entry is a dictionary with as many values as epochs that the
# model was trained on. 
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
```




    [Text(0, 0.5, 'Accuracy'), Text(0.5, 0, 'Epoch')]




![png](output_42_1.png)



![png](output_42_2.png)


**Congratulations**! You've finished the assignment and built two models: One that recognizes  smiles, and another that recognizes SIGN language with almost 80% accuracy on the test set. In addition to that, you now also understand the applications of two Keras APIs: Sequential and Functional. Nicely done! 

By now, you know a bit about how the Functional API works and may have glimpsed the possibilities. In your next assignment, you'll really get a feel for its power when you get the opportunity to build a very deep ConvNet, using ResNets! 

<a name='6'></a>
## 6 - Bibliography

You're always encouraged to read the official documentation. To that end, you can find the docs for the Sequential and Functional APIs here: 

https://www.tensorflow.org/guide/keras/sequential_model

https://www.tensorflow.org/guide/keras/functional
