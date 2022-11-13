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
            tfl.ZeroPadding2D(padding=(3, 3),input_shape=(64,64,3)),
        
            ## Conv2D with 32 7x7 filters and stride of 1
            tfl.Conv2D(filters=32, kernel_size=7, strides=(1,1), input_shape=[64, 64, 3]),
        
            ## BatchNormalization for axis 3
            tfl.BatchNormalization(axis=3),    
            
            ## ReLU
            tfl.ReLU(max_value=None, negative_slope=0.0, threshold=0.0),
            ## Max Pooling 2D with default parameters
        
            tfl.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None),
            
            ## Flatten layer
        
            tfl.Flatten(),
        
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            tfl.Dense(1,activation="sigmoid")
            
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
    38/38 [==============================] - 4s 100ms/step - loss: 1.3072 - accuracy: 0.7083
    Epoch 2/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.1798 - accuracy: 0.9333
    Epoch 3/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1656 - accuracy: 0.9400
    Epoch 4/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1509 - accuracy: 0.9350
    Epoch 5/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1494 - accuracy: 0.9383
    Epoch 6/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.1009 - accuracy: 0.9717
    Epoch 7/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.0624 - accuracy: 0.9800
    Epoch 8/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.0629 - accuracy: 0.9783
    Epoch 9/10
    38/38 [==============================] - 4s 97ms/step - loss: 0.1065 - accuracy: 0.9667
    Epoch 10/10
    38/38 [==============================] - 4s 93ms/step - loss: 0.0762 - accuracy: 0.9717





    <tensorflow.python.keras.callbacks.History at 0x7f674866b250>



After that completes, just use `.evaluate()` to evaluate against your test set. This function will print the value of the loss function and the performance metrics specified during the compilation of the model. In this case, the `binary_crossentropy` and the `accuracy` respectively.


```python
happy_model.evaluate(X_test, Y_test)
```

    5/5 [==============================] - 0s 35ms/step - loss: 0.9330 - accuracy: 0.6733





    [0.9330052733421326, 0.6733333468437195]



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



![png](output_28_1.png)


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
    Z1 = tf.keras.layers.Conv2D(filters=8, kernel_size=4, padding="same")(input_img)
    ## RELU
    A1 = tf.keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(Z1)
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.keras.layers.MaxPool2D(pool_size=(8,8), strides=(8,8), padding='same')(A1)
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tf.keras.layers.Conv2D(filters=16, kernel_size=2,strides=(1,1), padding="same")(P1)
    ## RELU
    A2 = tf.keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(Z2)
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.keras.layers.MaxPool2D(pool_size=(4,4), strides=(4,4), padding='same')(A2)
    ## FLATTEN
    F = tf.keras.layers.Flatten()(P2)
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    outputs = tf.keras.layers.Dense(6,activation="softmax")(F)
    
    
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
    17/17 [==============================] - 2s 112ms/step - loss: 1.8049 - accuracy: 0.1759 - val_loss: 1.7893 - val_accuracy: 0.1500
    Epoch 2/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7879 - accuracy: 0.1815 - val_loss: 1.7860 - val_accuracy: 0.2250
    Epoch 3/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7844 - accuracy: 0.2491 - val_loss: 1.7830 - val_accuracy: 0.2917
    Epoch 4/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7808 - accuracy: 0.2759 - val_loss: 1.7791 - val_accuracy: 0.3000
    Epoch 5/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7771 - accuracy: 0.2861 - val_loss: 1.7744 - val_accuracy: 0.3083
    Epoch 6/100
    17/17 [==============================] - 2s 102ms/step - loss: 1.7729 - accuracy: 0.3157 - val_loss: 1.7688 - val_accuracy: 0.3000
    Epoch 7/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7674 - accuracy: 0.3361 - val_loss: 1.7623 - val_accuracy: 0.3083
    Epoch 8/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.7595 - accuracy: 0.3787 - val_loss: 1.7507 - val_accuracy: 0.3333
    Epoch 9/100
    17/17 [==============================] - 2s 101ms/step - loss: 1.7477 - accuracy: 0.3806 - val_loss: 1.7352 - val_accuracy: 0.4083
    Epoch 10/100
    17/17 [==============================] - 2s 101ms/step - loss: 1.7327 - accuracy: 0.4065 - val_loss: 1.7160 - val_accuracy: 0.4667
    Epoch 11/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.7125 - accuracy: 0.4074 - val_loss: 1.6966 - val_accuracy: 0.4333
    Epoch 12/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.6878 - accuracy: 0.4278 - val_loss: 1.6692 - val_accuracy: 0.4750
    Epoch 13/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.6589 - accuracy: 0.4444 - val_loss: 1.6363 - val_accuracy: 0.4833
    Epoch 14/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.6146 - accuracy: 0.4667 - val_loss: 1.5902 - val_accuracy: 0.4667
    Epoch 15/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.5654 - accuracy: 0.4898 - val_loss: 1.5496 - val_accuracy: 0.4750
    Epoch 16/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.5178 - accuracy: 0.4991 - val_loss: 1.5027 - val_accuracy: 0.4833
    Epoch 17/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.4675 - accuracy: 0.5157 - val_loss: 1.4552 - val_accuracy: 0.5000
    Epoch 18/100
    17/17 [==============================] - 2s 101ms/step - loss: 1.4185 - accuracy: 0.5296 - val_loss: 1.4133 - val_accuracy: 0.5000
    Epoch 19/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.3673 - accuracy: 0.5426 - val_loss: 1.3644 - val_accuracy: 0.5083
    Epoch 20/100
    17/17 [==============================] - 2s 100ms/step - loss: 1.3263 - accuracy: 0.5519 - val_loss: 1.3261 - val_accuracy: 0.5250
    Epoch 21/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.2861 - accuracy: 0.5639 - val_loss: 1.2926 - val_accuracy: 0.5167
    Epoch 22/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.2396 - accuracy: 0.5843 - val_loss: 1.2502 - val_accuracy: 0.5833
    Epoch 23/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.2054 - accuracy: 0.6037 - val_loss: 1.2195 - val_accuracy: 0.6083
    Epoch 24/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.1707 - accuracy: 0.6074 - val_loss: 1.1898 - val_accuracy: 0.6000
    Epoch 25/100
    17/17 [==============================] - 2s 101ms/step - loss: 1.1368 - accuracy: 0.6157 - val_loss: 1.1582 - val_accuracy: 0.6167
    Epoch 26/100
    17/17 [==============================] - 2s 105ms/step - loss: 1.1072 - accuracy: 0.6250 - val_loss: 1.1332 - val_accuracy: 0.6500
    Epoch 27/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.0790 - accuracy: 0.6278 - val_loss: 1.1092 - val_accuracy: 0.6583
    Epoch 28/100
    17/17 [==============================] - 2s 106ms/step - loss: 1.0526 - accuracy: 0.6343 - val_loss: 1.0879 - val_accuracy: 0.6417
    Epoch 29/100
    17/17 [==============================] - 2s 111ms/step - loss: 1.0258 - accuracy: 0.6426 - val_loss: 1.0644 - val_accuracy: 0.6500
    Epoch 30/100
    17/17 [==============================] - 2s 112ms/step - loss: 1.0029 - accuracy: 0.6519 - val_loss: 1.0459 - val_accuracy: 0.6583
    Epoch 31/100
    17/17 [==============================] - 2s 112ms/step - loss: 0.9806 - accuracy: 0.6556 - val_loss: 1.0264 - val_accuracy: 0.6667
    Epoch 32/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.9586 - accuracy: 0.6657 - val_loss: 1.0090 - val_accuracy: 0.6750
    Epoch 33/100
    17/17 [==============================] - 2s 111ms/step - loss: 0.9394 - accuracy: 0.6759 - val_loss: 0.9929 - val_accuracy: 0.6833
    Epoch 34/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.9212 - accuracy: 0.6806 - val_loss: 0.9775 - val_accuracy: 0.7000
    Epoch 35/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.9034 - accuracy: 0.6898 - val_loss: 0.9630 - val_accuracy: 0.6833
    Epoch 36/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8867 - accuracy: 0.7000 - val_loss: 0.9487 - val_accuracy: 0.7167
    Epoch 37/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8710 - accuracy: 0.7102 - val_loss: 0.9364 - val_accuracy: 0.7250
    Epoch 38/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8561 - accuracy: 0.7167 - val_loss: 0.9242 - val_accuracy: 0.7417
    Epoch 39/100
    17/17 [==============================] - 2s 102ms/step - loss: 0.8420 - accuracy: 0.7185 - val_loss: 0.9124 - val_accuracy: 0.7500
    Epoch 40/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8284 - accuracy: 0.7222 - val_loss: 0.9009 - val_accuracy: 0.7417
    Epoch 41/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.8157 - accuracy: 0.7259 - val_loss: 0.8908 - val_accuracy: 0.7417
    Epoch 42/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.8032 - accuracy: 0.7287 - val_loss: 0.8803 - val_accuracy: 0.7417
    Epoch 43/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7913 - accuracy: 0.7370 - val_loss: 0.8711 - val_accuracy: 0.7500
    Epoch 44/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7799 - accuracy: 0.7370 - val_loss: 0.8617 - val_accuracy: 0.7500
    Epoch 45/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7692 - accuracy: 0.7389 - val_loss: 0.8536 - val_accuracy: 0.7500
    Epoch 46/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7594 - accuracy: 0.7426 - val_loss: 0.8459 - val_accuracy: 0.7500
    Epoch 47/100
    17/17 [==============================] - 2s 100ms/step - loss: 0.7483 - accuracy: 0.7454 - val_loss: 0.8376 - val_accuracy: 0.7667
    Epoch 48/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7386 - accuracy: 0.7500 - val_loss: 0.8303 - val_accuracy: 0.7667
    Epoch 49/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7293 - accuracy: 0.7509 - val_loss: 0.8241 - val_accuracy: 0.7583
    Epoch 50/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.7191 - accuracy: 0.7574 - val_loss: 0.8162 - val_accuracy: 0.7667
    Epoch 51/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7107 - accuracy: 0.7583 - val_loss: 0.8102 - val_accuracy: 0.7667
    Epoch 52/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.7021 - accuracy: 0.7593 - val_loss: 0.8064 - val_accuracy: 0.7667
    Epoch 53/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6934 - accuracy: 0.7630 - val_loss: 0.8015 - val_accuracy: 0.7667
    Epoch 54/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6850 - accuracy: 0.7667 - val_loss: 0.7970 - val_accuracy: 0.7667
    Epoch 55/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6770 - accuracy: 0.7694 - val_loss: 0.7905 - val_accuracy: 0.7667
    Epoch 56/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6685 - accuracy: 0.7704 - val_loss: 0.7854 - val_accuracy: 0.7667
    Epoch 57/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.6596 - accuracy: 0.7731 - val_loss: 0.7806 - val_accuracy: 0.7583
    Epoch 58/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.6511 - accuracy: 0.7769 - val_loss: 0.7737 - val_accuracy: 0.7583
    Epoch 59/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6435 - accuracy: 0.7787 - val_loss: 0.7685 - val_accuracy: 0.7583
    Epoch 60/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.6367 - accuracy: 0.7787 - val_loss: 0.7634 - val_accuracy: 0.7667
    Epoch 61/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6294 - accuracy: 0.7824 - val_loss: 0.7590 - val_accuracy: 0.7667
    Epoch 62/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6231 - accuracy: 0.7815 - val_loss: 0.7526 - val_accuracy: 0.7667
    Epoch 63/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6166 - accuracy: 0.7833 - val_loss: 0.7484 - val_accuracy: 0.7667
    Epoch 64/100
    17/17 [==============================] - 2s 105ms/step - loss: 0.6101 - accuracy: 0.7852 - val_loss: 0.7438 - val_accuracy: 0.7667
    Epoch 65/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.6040 - accuracy: 0.7889 - val_loss: 0.7394 - val_accuracy: 0.7583
    Epoch 66/100
    17/17 [==============================] - 2s 102ms/step - loss: 0.5984 - accuracy: 0.7898 - val_loss: 0.7352 - val_accuracy: 0.7667
    Epoch 67/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5923 - accuracy: 0.7944 - val_loss: 0.7306 - val_accuracy: 0.7750
    Epoch 68/100
    17/17 [==============================] - 2s 107ms/step - loss: 0.5867 - accuracy: 0.7963 - val_loss: 0.7266 - val_accuracy: 0.7750
    Epoch 69/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5813 - accuracy: 0.8000 - val_loss: 0.7214 - val_accuracy: 0.7750
    Epoch 70/100
    17/17 [==============================] - 2s 101ms/step - loss: 0.5758 - accuracy: 0.8046 - val_loss: 0.7170 - val_accuracy: 0.7750
    Epoch 71/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5706 - accuracy: 0.8037 - val_loss: 0.7124 - val_accuracy: 0.7750
    Epoch 72/100
    17/17 [==============================] - 2s 106ms/step - loss: 0.5656 - accuracy: 0.8083 - val_loss: 0.7084 - val_accuracy: 0.7750
    Epoch 73/100
    16/17 [===========================>..] - ETA: 0s - loss: 0.5575 - accuracy: 0.8096

<a name='5'></a>
## 5 - History Object 

The history object is an output of the `.fit()` operation, and provides a record of all the loss and metric values in memory. It's stored as a dictionary that you can retrieve at `history.history`: 


```python
history.history
```

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

**Congratulations**! You've finished the assignment and built two models: One that recognizes  smiles, and another that recognizes SIGN language with almost 80% accuracy on the test set. In addition to that, you now also understand the applications of two Keras APIs: Sequential and Functional. Nicely done! 

By now, you know a bit about how the Functional API works and may have glimpsed the possibilities. In your next assignment, you'll really get a feel for its power when you get the opportunity to build a very deep ConvNet, using ResNets! 

<a name='6'></a>
## 6 - Bibliography

You're always encouraged to read the official documentation. To that end, you can find the docs for the Sequential and Functional APIs here: 

https://www.tensorflow.org/guide/keras/sequential_model

https://www.tensorflow.org/guide/keras/functional
