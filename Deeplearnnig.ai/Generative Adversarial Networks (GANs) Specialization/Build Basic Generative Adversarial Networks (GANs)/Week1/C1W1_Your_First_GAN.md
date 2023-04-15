# Your First GAN

### Goal
In this notebook, you're going to create your first generative adversarial network (GAN) for this course! Specifically, you will build and train a GAN that can generate hand-written images of digits (0-9). You will be using PyTorch in this specialization, so if you're not familiar with this framework, you may find the [PyTorch documentation](https://pytorch.org/docs/stable/index.html) useful. The hints will also often include links to relevant documentation.

### Learning Objectives
1.   Build the generator and discriminator components of a GAN from scratch.
2.   Create generator and discriminator loss functions.
3.   Train your GAN and visualize the generated images.


## Getting Started
You will begin by importing some useful packages and the dataset you will use to build and train your GAN. You are also provided with a visualizer function to help you investigate the images your GAN will create.



```python
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes, please do not change!

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
```

#### MNIST Dataset
The training images your discriminator will be using is from a dataset called [MNIST](http://yann.lecun.com/exdb/mnist/). It contains 60,000 images of handwritten digits, from 0 to 9, like these:

![MNIST Digits](MnistExamples.png)

You may notice that the images are quite pixelated -- this is because they are all only 28 x 28! The small size of its images makes MNIST ideal for simple training. Additionally, these images are also in black-and-white so only one dimension, or "color channel", is needed to represent them (more on this later in the course).

#### Tensor
You will represent the data using [tensors](https://pytorch.org/docs/stable/tensors.html). Tensors are a generalization of matrices: for example, a stack of three matrices with the amounts of red, green, and blue at different locations in a 64 x 64 pixel image is a tensor with the shape 3 x 64 x 64.

Tensors are easy to manipulate and supported by [PyTorch](https://pytorch.org/), the machine learning library you will be using. Feel free to explore them more, but you can imagine these as multi-dimensional matrices or vectors!

#### Batches
While you could train your model after generating one image, it is extremely inefficient and leads to less stable training. In GANs, and in machine learning in general, you will process multiple images per training step. These are called batches.

This means that your generator will generate an entire batch of images and receive the discriminator's feedback on each before updating the model. The same goes for the discriminator, it will calculate its loss on the entire batch of generated images as well as on the reals before the model is updated.

## Generator
The first step is to build the generator component.

You will start by creating a function to make a single layer/block for the generator's neural network. Each block should include a [linear transformation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) to map to another shape, a [batch normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) for stabilization, and finally a non-linear activation function (you use a [ReLU here](https://pytorch.org/docs/master/generated/torch.nn.ReLU.html)) so the output can be transformed in complex ways. You will learn more about activations and batch normalization later in the course.


```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_generator_block
def get_generator_block(input_dim, output_dim):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    return nn.Sequential(
        # Hint: Replace all of the "None" with the appropriate dimensions.
        # The documentation may be useful if you're less familiar with PyTorch:
        # https://pytorch.org/docs/stable/nn.html.
        #### START CODE HERE ####
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        #### END CODE HERE ####
        nn.ReLU(inplace=True)
    )
```


```python
# Verify the generator block function
def test_gen_block(in_features, out_features, num_test=1000):
    block = get_generator_block(in_features, out_features)

    # Check the three parts
    assert len(block) == 3
    assert type(block[0]) == nn.Linear
    assert type(block[1]) == nn.BatchNorm1d
    assert type(block[2]) == nn.ReLU
    
    # Check the output shape
    test_input = torch.randn(num_test, in_features)
    test_output = block(test_input)
    assert tuple(test_output.shape) == (num_test, out_features)
    assert test_output.std() > 0.55
    assert test_output.std() < 0.65

test_gen_block(25, 12)
test_gen_block(15, 28)
print("Success!")
```

    Success!


Now you can build the generator class. It will take 3 values:

*   The noise vector dimension
*   The image dimension
*   The initial hidden dimension

Using these values, the generator will build a neural network with 5 layers/blocks. Beginning with the noise vector, the generator will apply non-linear transformations via the block function until the tensor is mapped to the size of the image to be outputted (the same size as the real images from MNIST). You will need to fill in the code for final layer since it is different than the others. The final layer does not need a normalization or activation function, but does need to be scaled with a [sigmoid function](https://pytorch.org/docs/master/generated/torch.nn.Sigmoid.html). 

Finally, you are given a forward pass function that takes in a noise vector and generates an image of the output dimension using your neural network.

<details>

<summary>
<font size="3" color="green">
<b>Optional hints for <code><font size="4">Generator</font></code></b>
</font>
</summary>

1. The output size of the final linear transformation should be im_dim, but remember you need to scale the outputs between 0 and 1 using the sigmoid function.
2. [nn.Linear](https://pytorch.org/docs/master/generated/torch.nn.Linear.html) and [nn.Sigmoid](https://pytorch.org/docs/master/generated/torch.nn.Sigmoid.html) will be useful here. 
</details>



```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Generator
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        # Build the neural network
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            # There is a dropdown with hints if you need them! 
            #### START CODE HERE ####
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
            #### END CODE HERE ####
        )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)
    
    # Needed for grading
    def get_gen(self):
        '''
        Returns:
            the sequential model
        '''
        return self.gen
```


```python
# Verify the generator class
def test_generator(z_dim, im_dim, hidden_dim, num_test=10000):
    gen = Generator(z_dim, im_dim, hidden_dim).get_gen()
    
    # Check there are six modules in the sequential part
    assert len(gen) == 6
    test_input = torch.randn(num_test, z_dim)
    test_output = gen(test_input)

    # Check that the output shape is correct
    assert tuple(test_output.shape) == (num_test, im_dim)
    assert test_output.max() < 1, "Make sure to use a sigmoid"
    assert test_output.min() > 0, "Make sure to use a sigmoid"
    assert test_output.min() < 0.5, "Don't use a block in your solution"
    assert test_output.std() > 0.05, "Don't use batchnorm here"
    assert test_output.std() < 0.15, "Don't use batchnorm here"

test_generator(5, 10, 20)
test_generator(20, 8, 24)
print("Success!")
```

    Success!


## Noise
To be able to use your generator, you will need to be able to create noise vectors. The noise vector z has the important role of making sure the images generated from the same class don't all look the same -- think of it as a random seed. You will generate it randomly using PyTorch by sampling random numbers from the normal distribution. Since multiple images will be processed per pass, you will generate all the noise vectors at once.

Note that whenever you create a new tensor using torch.ones, torch.zeros, or torch.randn, you either need to create it on the target device, e.g. `torch.ones(3, 3, device=device)`, or move it onto the target device using `torch.ones(3, 3).to(device)`. You do not need to do this if you're creating a tensor by manipulating another tensor or by using a variation that defaults the device to the input, such as `torch.ones_like`. In general, use `torch.ones_like` and `torch.zeros_like` instead of `torch.ones` or `torch.zeros` where possible.

<details>

<summary>
<font size="3" color="green">
<b>Optional hint for <code><font size="4">get_noise</font></code></b>
</font>
</summary>

1. 
You will probably find [torch.randn](https://pytorch.org/docs/master/generated/torch.randn.html) useful here.
</details>


```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_noise
def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    # NOTE: To use this on GPU with device='cuda', make sure to pass the device 
    # argument to the function you use to generate the noise.
    #### START CODE HERE ####
    return torch.randn(n_samples, z_dim).to(device)
    #### END CODE HERE ####
```


```python
# Verify the noise vector function
def test_get_noise(n_samples, z_dim, device='cpu'):
    noise = get_noise(n_samples, z_dim, device)
    
    # Make sure a normal distribution was used
    assert tuple(noise.shape) == (n_samples, z_dim)
    assert torch.abs(noise.std() - torch.tensor(1.0)) < 0.01
    assert str(noise.device).startswith(device)

test_get_noise(1000, 100, 'cpu')
if torch.cuda.is_available():
    test_get_noise(1000, 32, 'cuda')
print("Success!")
```

    Success!


## Discriminator
The second component that you need to construct is the discriminator. As with the generator component, you will start by creating a function that builds a neural network block for the discriminator.

*Note: You use leaky ReLUs to prevent the "dying ReLU" problem, which refers to the phenomenon where the parameters stop changing due to consistently negative values passed to a ReLU, which result in a zero gradient. You will learn more about this in the following lectures!* 


REctified Linear Unit (ReLU) |  Leaky ReLU
:-------------------------:|:-------------------------:
![](./relu-graph.png)  |  ![](./lrelu-graph.png)






```python
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_discriminator_block
def get_discriminator_block(input_dim, output_dim):
    '''
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
          followed by an nn.LeakyReLU activation with negative slope of 0.2 
          (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
    '''
    return nn.Sequential(
        #### START CODE HERE ####
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2)
        #### END CODE HERE ####
    )
```


```python
# Verify the discriminator block function
def test_disc_block(in_features, out_features, num_test=10000):
    block = get_discriminator_block(in_features, out_features)

    # Check there are two parts
    assert len(block) == 2
    test_input = torch.randn(num_test, in_features)
    test_output = block(test_input)

    # Check that the shape is right
    assert tuple(test_output.shape) == (num_test, out_features)
    
    # Check that the LeakyReLU slope is about 0.2
    assert -test_output.min() / test_output.max() > 0.1
    assert -test_output.min() / test_output.max() < 0.3
    assert test_output.std() > 0.3
    assert test_output.std() < 0.5

test_disc_block(25, 12)
test_disc_block(15, 28)
print("Success!")
```

    Success!


Now you can use these blocks to make a discriminator! The discriminator class holds 2 values:

*   The image dimension
*   The hidden dimension

The discriminator will build a neural network with 4 layers. It will start with the image tensor and transform it until it returns a single number (1-dimension tensor) output. This output classifies whether an image is fake or real. Note that you do not need a sigmoid after the output layer since it is included in the loss function. Finally, to use your discrimator's neural network you are given a forward pass function that takes in an image tensor to be classified.



```python
# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: Discriminator
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            # Hint: You want to transform the final output into a single value,
            #       so add one more linear map.
            #### START CODE HERE ####
            nn.Linear(hidden_dim, 1)
            #### END CODE HERE ####
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        return self.disc(image)
    
    # Needed for grading
    def get_disc(self):
        '''
        Returns:
            the sequential model
        '''
        return self.disc
```


```python
# Verify the discriminator class
def test_discriminator(z_dim, hidden_dim, num_test=100):
    
    disc = Discriminator(z_dim, hidden_dim).get_disc()

    # Check there are three parts
    assert len(disc) == 4

    # Check the linear layer is correct
    test_input = torch.randn(num_test, z_dim)
    test_output = disc(test_input)
    assert tuple(test_output.shape) == (num_test, 1)
    
    # Don't use a block
    assert not isinstance(disc[-1], nn.Sequential)

test_discriminator(5, 10)
test_discriminator(20, 8)
print("Success!")
```

    Success!


## Training
Now you can put it all together!
First, you will set your parameters:
  *   criterion: the loss function
  *   n_epochs: the number of times you iterate through the entire dataset when training
  *   z_dim: the dimension of the noise vector
  *   display_step: how often to display/visualize the images
  *   batch_size: the number of images per forward/backward pass
  *   lr: the learning rate
  *   device: the device type, here using a GPU (which runs CUDA), not CPU

Next, you will load the MNIST dataset as tensors using a dataloader.




```python
# Set your parameters
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001

# Load MNIST dataset as tensors
dataloader = DataLoader(
    MNIST('.', download=False, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)

### DO NOT EDIT ###
device = 'cuda'
```

Now, you can initialize your generator, discriminator, and optimizers. Note that each optimizer only takes the parameters of one particular model, since we want each optimizer to optimize only one of the models.


```python
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
```

Before you train your GAN, you will need to create functions to calculate the discriminator's loss and the generator's loss. This is how the discriminator and generator will know how they are doing and improve themselves. Since the generator is needed when calculating the discriminator's loss, you will need to call .detach() on the generator result to ensure that only the discriminator is updated!

Remember that you have already defined a loss function earlier (`criterion`) and you are encouraged to use `torch.ones_like` and `torch.zeros_like` instead of `torch.ones` or `torch.zeros`. If you use `torch.ones` or `torch.zeros`, you'll need to pass `device=device` to them.


```python
# UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_disc_loss
def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch (num_images) of fake images. 
    #            Make sure to pass the device argument to the noise.
    #       2) Get the discriminator's prediction of the fake image 
    #            and calculate the loss. Don't forget to detach the generator!
    #            (Remember the loss function you set earlier -- criterion. You need a 
    #            'ground truth' tensor in order to calculate the loss. 
    #            For example, a ground truth tensor for a fake image is all zeros.)
    #       3) Get the discriminator's prediction of the real image and calculate the loss.
    #       4) Calculate the discriminator's loss by averaging the real and fake loss
    #            and set it to disc_loss.
    #     *Important*: You should NOT write your own loss function here - use criterion(pred, true)!
    #### START CODE HERE ####
    noise = get_noise(num_images, z_dim, device)
    fake_image = gen(noise).detach()
    y_pred = disc(fake_image)
    y_real = disc(real)
    loss_fake = criterion(y_pred, torch.zeros_like(y_pred))
    loss_real = criterion(y_real, torch.ones_like(y_real))
    disc_loss = (loss_fake + loss_real) /2
    #### END CODE HERE ####
    return disc_loss
```


```python
def test_disc_reasonable(num_images=10):
    # Don't use explicit casts to cuda - use the device argument
    import inspect, re
    lines = inspect.getsource(get_disc_loss)
    assert (re.search(r"to\(.cuda.\)", lines)) is None
    assert (re.search(r"\.cuda\(\)", lines)) is None
    
    z_dim = 64
    gen = torch.zeros_like
    disc = lambda x: x.mean(1)[:, None]
    criterion = torch.mul # Multiply
    real = torch.ones(num_images, z_dim)
    disc_loss = get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu')
    assert torch.all(torch.abs(disc_loss.mean() - 0.5) < 1e-5)
    
    gen = torch.ones_like
    criterion = torch.mul # Multiply
    real = torch.zeros(num_images, z_dim)
    assert torch.all(torch.abs(get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu')) < 1e-5)
    
    gen = lambda x: torch.ones(num_images, 10)
    disc = lambda x: x.mean(1)[:, None] + 10
    criterion = torch.mul # Multiply
    real = torch.zeros(num_images, 10)
    assert torch.all(torch.abs(get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu').mean() - 5) < 1e-5)

    gen = torch.ones_like
    disc = nn.Linear(64, 1, bias=False)
    real = torch.ones(num_images, 64) * 0.5
    disc.weight.data = torch.ones_like(disc.weight.data) * 0.5
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    criterion = lambda x, y: torch.sum(x) + torch.sum(y)
    disc_loss = get_disc_loss(gen, disc, criterion, real, num_images, z_dim, 'cpu').mean()
    disc_loss.backward()
    assert torch.isclose(torch.abs(disc.weight.grad.mean() - 11.25), torch.tensor(3.75))
    
def test_disc_loss(max_tests = 10):
    z_dim = 64
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device) 
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    num_steps = 0
    for real, _ in dataloader:
        cur_batch_size = len(real)
        real = real.view(cur_batch_size, -1).to(device)

        ### Update discriminator ###
        # Zero out the gradient before backpropagation
        disc_opt.zero_grad()

        # Calculate discriminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
        assert (disc_loss - 0.68).abs() < 0.05

        # Update gradients
        disc_loss.backward(retain_graph=True)

        # Check that they detached correctly
        assert gen.gen[0][0].weight.grad is None

        # Update optimizer
        old_weight = disc.disc[0][0].weight.data.clone()
        disc_opt.step()
        new_weight = disc.disc[0][0].weight.data
        
        # Check that some discriminator weights changed
        assert not torch.all(torch.eq(old_weight, new_weight))
        num_steps += 1
        if num_steps >= max_tests:
            break

test_disc_reasonable()
test_disc_loss()
print("Success!")

```

    Success!



```python
# UNQ_C7 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gen_loss
def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch of fake images. 
    #           Remember to pass the device argument to the get_noise function.
    #       2) Get the discriminator's prediction of the fake image.
    #       3) Calculate the generator's loss. Remember the generator wants
    #          the discriminator to think that its fake images are real
    #     *Important*: You should NOT write your own loss function here - use criterion(pred, true)!

    #### START CODE HERE ####
    noise = get_noise(num_images, z_dim, device)
    fake_image = gen(noise)
    y_pred = disc(fake_image)
    gen_loss = criterion(y_pred, torch.ones_like(y_pred))
    #### END CODE HERE ####
    return gen_loss
```


```python
def test_gen_reasonable(num_images=10):
    # Don't use explicit casts to cuda - use the device argument
    import inspect, re
    lines = inspect.getsource(get_gen_loss)
    assert (re.search(r"to\(.cuda.\)", lines)) is None
    assert (re.search(r"\.cuda\(\)", lines)) is None
    
    z_dim = 64
    gen = torch.zeros_like
    disc = nn.Identity()
    criterion = torch.mul # Multiply
    gen_loss_tensor = get_gen_loss(gen, disc, criterion, num_images, z_dim, 'cpu')
    assert torch.all(torch.abs(gen_loss_tensor) < 1e-5)
    #Verify shape. Related to gen_noise parametrization
    assert tuple(gen_loss_tensor.shape) == (num_images, z_dim)

    gen = torch.ones_like
    disc = nn.Identity()
    criterion = torch.mul # Multiply
    gen_loss_tensor = get_gen_loss(gen, disc, criterion, num_images, z_dim, 'cpu')
    assert torch.all(torch.abs(gen_loss_tensor - 1) < 1e-5)
    #Verify shape. Related to gen_noise parametrization
    assert tuple(gen_loss_tensor.shape) == (num_images, z_dim)
    

def test_gen_loss(num_images):
    z_dim = 64
    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device) 
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    
    gen_loss = get_gen_loss(gen, disc, criterion, num_images, z_dim, device)
    
    # Check that the loss is reasonable
    assert (gen_loss - 0.7).abs() < 0.1
    gen_loss.backward()
    old_weight = gen.gen[0][0].weight.clone()
    gen_opt.step()
    new_weight = gen.gen[0][0].weight
    assert not torch.all(torch.eq(old_weight, new_weight))


test_gen_reasonable(10)
test_gen_loss(18)
print("Success!")
```

    Success!


Finally, you can put everything together! For each epoch, you will process the entire dataset in batches. For every batch, you will need to update the discriminator and generator using their loss. Batches are sets of images that will be predicted on before the loss functions are calculated (instead of calculating the loss function after each image). Note that you may see a loss to be greater than 1, this is okay since binary cross entropy loss can be any positive number for a sufficiently confident wrong guess. 

It’s also often the case that the discriminator will outperform the generator, especially at the start, because its job is easier. It's important that neither one gets too good (that is, near-perfect accuracy), which would cause the entire model to stop learning. Balancing the two models is actually remarkably hard to do in a standard GAN and something you will see more of in later lectures and assignments.

After you've submitted a working version with the original architecture, feel free to play around with the architecture if you want to see how different architectural choices can lead to better or worse GANs. For example, consider changing the size of the hidden dimension, or making the networks shallower or deeper by changing the number of layers.

<!-- In addition, be warned that this runs very slowly on a CPU. One way to run this more quickly is to use Google Colab: 

1.   Download the .ipynb
2.   Upload it to Google Drive and open it with Google Colab
3.   Make the runtime type GPU (under “Runtime” -> “Change runtime type” -> Select “GPU” from the dropdown)
4.   Replace `device = "cpu"` with `device = "cuda"`
5.   Make sure your `get_noise` function uses the right device -->

But remember, don’t expect anything spectacular: this is only the first lesson. The results will get better with later lessons as you learn methods to help keep your generator and discriminator at similar levels.

You should roughly expect to see this progression. On a GPU, this should take about 15 seconds per 500 steps, on average, while on CPU it will take roughly 1.5 minutes:
![MNIST Digits](https://drive.google.com/uc?export=view&id=1BlfFNZACaieFrOjMv_o2kGqwAR6eiLmN)


```python
# OPTIONAL PART

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True # Whether the generator should be tested
gen_loss = False
error = False
for epoch in range(n_epochs):
  
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

        # Flatten the batch of real images from the dataset
        real = real.view(cur_batch_size, -1).to(device)

        ### Update discriminator ###
        # Zero out the gradients before backpropagation
        disc_opt.zero_grad()

        # Calculate discriminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)

        # Update gradients
        disc_loss.backward(retain_graph=True)

        # Update optimizer
        disc_opt.step()

        # For testing purposes, to keep track of the generator weights
        if test_generator:
            old_generator_weights = gen.gen[0][0].weight.detach().clone()

        ### Update generator ###
        #     Hint: This code will look a lot like the discriminator updates!
        #     These are the steps you will need to complete:
        #       1) Zero out the gradients.
        #       2) Calculate the generator loss, assigning it to gen_loss.
        #       3) Backprop through the generator: update the gradients and optimizer.
        #### START CODE HERE ####
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward()
        gen_opt.step()
        #### END CODE HERE ####

        # For testing purposes, to check that your code changes the generator weights
        if test_generator:
            try:
                assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
            except:
                error = True
                print("Runtime tests have failed")

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1

```


      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 500: Generator loss: 1.4519755979776383, discriminator loss: 0.4083315249085426



    
![png](output_31_3.png)
    



    
![png](output_31_4.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 1000: Generator loss: 1.7964482858181, discriminator loss: 0.2687458098828796



    
![png](output_31_7.png)
    



    
![png](output_31_8.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 1500: Generator loss: 2.0437490901947, discriminator loss: 0.16406640297174432



    
![png](output_31_11.png)
    



    
![png](output_31_12.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 2000: Generator loss: 1.6624373681545264, discriminator loss: 0.23227261656522732



    
![png](output_31_15.png)
    



    
![png](output_31_16.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 2500: Generator loss: 1.5424311947822575, discriminator loss: 0.23968650183081616



    
![png](output_31_19.png)
    



    
![png](output_31_20.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 3000: Generator loss: 1.776551018238068, discriminator loss: 0.19418992444872868



    
![png](output_31_23.png)
    



    
![png](output_31_24.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 3500: Generator loss: 2.283523863554002, discriminator loss: 0.13904768325388434



    
![png](output_31_27.png)
    



    
![png](output_31_28.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 4000: Generator loss: 2.6919237799644464, discriminator loss: 0.11343891544640064



    
![png](output_31_31.png)
    



    
![png](output_31_32.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 4500: Generator loss: 3.0122037825584442, discriminator loss: 0.1053589980900289



    
![png](output_31_35.png)
    



    
![png](output_31_36.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 5000: Generator loss: 3.2434593706130994, discriminator loss: 0.09934416046738628



    
![png](output_31_39.png)
    



    
![png](output_31_40.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 5500: Generator loss: 3.3327725567817676, discriminator loss: 0.08806551019847399



    
![png](output_31_43.png)
    



    
![png](output_31_44.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 6000: Generator loss: 3.6317390680313117, discriminator loss: 0.07637205016613005



    
![png](output_31_47.png)
    



    
![png](output_31_48.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 6500: Generator loss: 3.6937981548309344, discriminator loss: 0.0709947549626231



    
![png](output_31_51.png)
    



    
![png](output_31_52.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 7000: Generator loss: 3.5530053138732915, discriminator loss: 0.0761131272763014



    
![png](output_31_55.png)
    



    
![png](output_31_56.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 7500: Generator loss: 3.6413460488319407, discriminator loss: 0.0607084618732333



    
![png](output_31_59.png)
    



    
![png](output_31_60.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 8000: Generator loss: 3.877961936950682, discriminator loss: 0.05243231688439842



    
![png](output_31_64.png)
    



    
![png](output_31_65.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 8500: Generator loss: 3.9097374968528733, discriminator loss: 0.06128458502143628



    
![png](output_31_68.png)
    



    
![png](output_31_69.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 9000: Generator loss: 3.962013502597805, discriminator loss: 0.06558868592232465



    
![png](output_31_72.png)
    



    
![png](output_31_73.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 9500: Generator loss: 3.859972579479216, discriminator loss: 0.06334410699456931



    
![png](output_31_76.png)
    



    
![png](output_31_77.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 10000: Generator loss: 3.965089545249939, discriminator loss: 0.06710259929299352



    
![png](output_31_80.png)
    



    
![png](output_31_81.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 10500: Generator loss: 3.9525598621368374, discriminator loss: 0.07011856111139056



    
![png](output_31_84.png)
    



    
![png](output_31_85.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 11000: Generator loss: 3.845519155979157, discriminator loss: 0.07030728945881128



    
![png](output_31_88.png)
    



    
![png](output_31_89.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 11500: Generator loss: 3.7848402671814005, discriminator loss: 0.07602443809434771



    
![png](output_31_92.png)
    



    
![png](output_31_93.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 12000: Generator loss: 3.9612501707077064, discriminator loss: 0.07980685787647968



    
![png](output_31_96.png)
    



    
![png](output_31_97.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 12500: Generator loss: 3.9160759677886983, discriminator loss: 0.08547861894220113



    
![png](output_31_100.png)
    



    
![png](output_31_101.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 13000: Generator loss: 3.9659827804565455, discriminator loss: 0.09027228431776163



    
![png](output_31_104.png)
    



    
![png](output_31_105.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 13500: Generator loss: 3.5028753433227506, discriminator loss: 0.11579327483475207



    
![png](output_31_108.png)
    



    
![png](output_31_109.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 14000: Generator loss: 3.59356876039505, discriminator loss: 0.10438513927161684



    
![png](output_31_112.png)
    



    
![png](output_31_113.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 14500: Generator loss: 3.6777227864265423, discriminator loss: 0.10291390369087464



    
![png](output_31_116.png)
    



    
![png](output_31_117.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 15000: Generator loss: 3.681973438262942, discriminator loss: 0.1133099763467907



    
![png](output_31_120.png)
    



    
![png](output_31_121.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 15500: Generator loss: 3.527122836112975, discriminator loss: 0.12316123229265223



    
![png](output_31_125.png)
    



    
![png](output_31_126.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 16000: Generator loss: 3.537960205078127, discriminator loss: 0.127341512106359



    
![png](output_31_129.png)
    



    
![png](output_31_130.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 16500: Generator loss: 3.622261183261872, discriminator loss: 0.12530675105750555



    
![png](output_31_133.png)
    



    
![png](output_31_134.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 17000: Generator loss: 3.563381230354311, discriminator loss: 0.11970547382533547



    
![png](output_31_137.png)
    



    
![png](output_31_138.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 17500: Generator loss: 3.4593440995216413, discriminator loss: 0.1345961386114358



    
![png](output_31_141.png)
    



    
![png](output_31_142.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 18000: Generator loss: 3.4712172169685376, discriminator loss: 0.1352806159108877



    
![png](output_31_145.png)
    



    
![png](output_31_146.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 18500: Generator loss: 3.5650890092849696, discriminator loss: 0.12647391945868736



    
![png](output_31_149.png)
    



    
![png](output_31_150.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 19000: Generator loss: 3.3800440449714677, discriminator loss: 0.12737362552434212



    
![png](output_31_153.png)
    



    
![png](output_31_154.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 19500: Generator loss: 3.460266919612882, discriminator loss: 0.1213540537804364



    
![png](output_31_157.png)
    



    
![png](output_31_158.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 20000: Generator loss: 3.4338675618171663, discriminator loss: 0.13660963678360002



    
![png](output_31_161.png)
    



    
![png](output_31_162.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 20500: Generator loss: 3.3660872430801416, discriminator loss: 0.1463007941693067



    
![png](output_31_165.png)
    



    
![png](output_31_166.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 21000: Generator loss: 3.2584140596389792, discriminator loss: 0.15357837621867657



    
![png](output_31_169.png)
    



    
![png](output_31_170.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 21500: Generator loss: 3.308559121608734, discriminator loss: 0.1600074314177037



    
![png](output_31_173.png)
    



    
![png](output_31_174.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 22000: Generator loss: 3.04912155294418, discriminator loss: 0.17766419802606112



    
![png](output_31_177.png)
    



    
![png](output_31_178.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 22500: Generator loss: 2.991594411373138, discriminator loss: 0.172732798099518



    
![png](output_31_181.png)
    



    
![png](output_31_182.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 23000: Generator loss: 3.168151021480559, discriminator loss: 0.17640149301290517



    
![png](output_31_186.png)
    



    
![png](output_31_187.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 23500: Generator loss: 2.963632841587066, discriminator loss: 0.18662444584071647



    
![png](output_31_190.png)
    



    
![png](output_31_191.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 24000: Generator loss: 3.0435085635185275, discriminator loss: 0.17782675586640834



    
![png](output_31_194.png)
    



    
![png](output_31_195.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 24500: Generator loss: 3.1192234053611747, discriminator loss: 0.1597359870225189



    
![png](output_31_198.png)
    



    
![png](output_31_199.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 25000: Generator loss: 3.008556437969208, discriminator loss: 0.1869365185946226



    
![png](output_31_202.png)
    



    
![png](output_31_203.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 25500: Generator loss: 2.8671735596656798, discriminator loss: 0.20930757981538783



    
![png](output_31_206.png)
    



    
![png](output_31_207.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 26000: Generator loss: 2.9407764048576346, discriminator loss: 0.20130801752209673



    
![png](output_31_210.png)
    



    
![png](output_31_211.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 26500: Generator loss: 2.767311824798582, discriminator loss: 0.2166636807024481



    
![png](output_31_214.png)
    



    
![png](output_31_215.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 27000: Generator loss: 2.694809013843535, discriminator loss: 0.2194490643888712



    
![png](output_31_218.png)
    



    
![png](output_31_219.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 27500: Generator loss: 2.6054097466468846, discriminator loss: 0.23476020544767384



    
![png](output_31_222.png)
    



    
![png](output_31_223.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 28000: Generator loss: 2.5839996676444987, discriminator loss: 0.23131823414564132



    
![png](output_31_226.png)
    



    
![png](output_31_227.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 28500: Generator loss: 2.7257746272087093, discriminator loss: 0.2127324385643007



    
![png](output_31_230.png)
    



    
![png](output_31_231.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 29000: Generator loss: 2.6752124848365786, discriminator loss: 0.21464626087248306



    
![png](output_31_234.png)
    



    
![png](output_31_235.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 29500: Generator loss: 2.614641642093658, discriminator loss: 0.22566779503226284



    
![png](output_31_238.png)
    



    
![png](output_31_239.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 30000: Generator loss: 2.7450142412185663, discriminator loss: 0.21307070198655148



    
![png](output_31_242.png)
    



    
![png](output_31_243.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 30500: Generator loss: 2.5270496993064873, discriminator loss: 0.23849389672279347



    
![png](output_31_247.png)
    



    
![png](output_31_248.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 31000: Generator loss: 2.7893358993530275, discriminator loss: 0.18763435703515985



    
![png](output_31_251.png)
    



    
![png](output_31_252.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 31500: Generator loss: 2.8564955120086664, discriminator loss: 0.18994559049606322



    
![png](output_31_255.png)
    



    
![png](output_31_256.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 32000: Generator loss: 2.824802839279174, discriminator loss: 0.20032289609313012



    
![png](output_31_259.png)
    



    
![png](output_31_260.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 32500: Generator loss: 2.817141379833224, discriminator loss: 0.20586529269814471



    
![png](output_31_263.png)
    



    
![png](output_31_264.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 33000: Generator loss: 2.6802453141212483, discriminator loss: 0.2279499523937701



    
![png](output_31_267.png)
    



    
![png](output_31_268.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 33500: Generator loss: 2.5621453747749325, discriminator loss: 0.2324880624115466



    
![png](output_31_271.png)
    



    
![png](output_31_272.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 34000: Generator loss: 2.5729925861358627, discriminator loss: 0.22924914056062712



    
![png](output_31_275.png)
    



    
![png](output_31_276.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 34500: Generator loss: 2.535235621452331, discriminator loss: 0.23596013027429583



    
![png](output_31_279.png)
    



    
![png](output_31_280.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 35000: Generator loss: 2.275671793222427, discriminator loss: 0.2805737277865408



    
![png](output_31_283.png)
    



    
![png](output_31_284.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 35500: Generator loss: 2.3039899802207935, discriminator loss: 0.27679134687781337



    
![png](output_31_287.png)
    



    
![png](output_31_288.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 36000: Generator loss: 2.283299221992493, discriminator loss: 0.2657596974968911



    
![png](output_31_291.png)
    



    
![png](output_31_292.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 36500: Generator loss: 2.3069647009372716, discriminator loss: 0.2711677777767183



    
![png](output_31_295.png)
    



    
![png](output_31_296.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 37000: Generator loss: 2.3477455999851213, discriminator loss: 0.2538313695192337



    
![png](output_31_299.png)
    



    
![png](output_31_300.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 37500: Generator loss: 2.3856826758384715, discriminator loss: 0.25561197018623366



    
![png](output_31_303.png)
    



    
![png](output_31_304.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 38000: Generator loss: 2.269376292705537, discriminator loss: 0.2783611330687999



    
![png](output_31_308.png)
    



    
![png](output_31_309.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 38500: Generator loss: 2.283368401288987, discriminator loss: 0.27636298391222947



    
![png](output_31_312.png)
    



    
![png](output_31_313.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 39000: Generator loss: 2.223164829015732, discriminator loss: 0.2889422015845777



    
![png](output_31_316.png)
    



    
![png](output_31_317.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 39500: Generator loss: 2.1415970182418795, discriminator loss: 0.2895905257761478



    
![png](output_31_320.png)
    



    
![png](output_31_321.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 40000: Generator loss: 2.246453771114349, discriminator loss: 0.2688218008279801



    
![png](output_31_324.png)
    



    
![png](output_31_325.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 40500: Generator loss: 2.203869190216062, discriminator loss: 0.28818326613307



    
![png](output_31_328.png)
    



    
![png](output_31_329.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 41000: Generator loss: 2.120078879594805, discriminator loss: 0.29756669932603835



    
![png](output_31_332.png)
    



    
![png](output_31_333.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 41500: Generator loss: 2.152223057270053, discriminator loss: 0.2838140029609204



    
![png](output_31_336.png)
    



    
![png](output_31_337.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 42000: Generator loss: 2.1748252408504505, discriminator loss: 0.29849713626503954



    
![png](output_31_340.png)
    



    
![png](output_31_341.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 42500: Generator loss: 2.1151256163120284, discriminator loss: 0.2921436564028261



    
![png](output_31_344.png)
    



    
![png](output_31_345.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 43000: Generator loss: 2.1187556238174436, discriminator loss: 0.3108876097500323



    
![png](output_31_348.png)
    



    
![png](output_31_349.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 43500: Generator loss: 2.0162302560806276, discriminator loss: 0.32070032253861447



    
![png](output_31_352.png)
    



    
![png](output_31_353.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 44000: Generator loss: 1.982310005187987, discriminator loss: 0.32193479415774345



    
![png](output_31_356.png)
    



    
![png](output_31_357.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 44500: Generator loss: 2.0082954223155998, discriminator loss: 0.31089258688688276



    
![png](output_31_360.png)
    



    
![png](output_31_361.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 45000: Generator loss: 2.105613631725309, discriminator loss: 0.2832366302311419



    
![png](output_31_364.png)
    



    
![png](output_31_365.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 45500: Generator loss: 2.0831980895996094, discriminator loss: 0.2990978417396546



    
![png](output_31_369.png)
    



    
![png](output_31_370.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 46000: Generator loss: 1.9914432499408714, discriminator loss: 0.3118910972774029



    
![png](output_31_373.png)
    



    
![png](output_31_374.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 46500: Generator loss: 1.9596053893566137, discriminator loss: 0.3177729516625403



    
![png](output_31_377.png)
    



    
![png](output_31_378.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 47000: Generator loss: 1.9832961206436164, discriminator loss: 0.3172543453872202



    
![png](output_31_381.png)
    



    
![png](output_31_382.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 47500: Generator loss: 1.9314368324279796, discriminator loss: 0.3321290986537934



    
![png](output_31_385.png)
    



    
![png](output_31_386.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 48000: Generator loss: 1.9272209479808793, discriminator loss: 0.33750362822413454



    
![png](output_31_389.png)
    



    
![png](output_31_390.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 48500: Generator loss: 1.9482359752655027, discriminator loss: 0.3330512658059598



    
![png](output_31_393.png)
    



    
![png](output_31_394.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 49000: Generator loss: 1.9173851346969608, discriminator loss: 0.33514723673462854



    
![png](output_31_397.png)
    



    
![png](output_31_398.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 49500: Generator loss: 1.928641286849976, discriminator loss: 0.33326875385642046



    
![png](output_31_401.png)
    



    
![png](output_31_402.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 50000: Generator loss: 1.9489359428882589, discriminator loss: 0.3358621298670767



    
![png](output_31_405.png)
    



    
![png](output_31_406.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 50500: Generator loss: 1.90805536675453, discriminator loss: 0.3434236035346987



    
![png](output_31_409.png)
    



    
![png](output_31_410.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 51000: Generator loss: 1.8723535315990445, discriminator loss: 0.34667164939642003



    
![png](output_31_413.png)
    



    
![png](output_31_414.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 51500: Generator loss: 1.8798282546997054, discriminator loss: 0.3397968833744529



    
![png](output_31_417.png)
    



    
![png](output_31_418.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 52000: Generator loss: 1.9326277589797989, discriminator loss: 0.32631474328041077



    
![png](output_31_421.png)
    



    
![png](output_31_422.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 52500: Generator loss: 2.039347404718398, discriminator loss: 0.3175268545746805



    
![png](output_31_425.png)
    



    
![png](output_31_426.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 53000: Generator loss: 1.7721690895557418, discriminator loss: 0.3721744621396064



    
![png](output_31_430.png)
    



    
![png](output_31_431.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 53500: Generator loss: 1.718030970811845, discriminator loss: 0.3787341841459274



    
![png](output_31_434.png)
    



    
![png](output_31_435.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 54000: Generator loss: 1.9135735449790956, discriminator loss: 0.32786619988083804



    
![png](output_31_438.png)
    



    
![png](output_31_439.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 54500: Generator loss: 1.839790132045746, discriminator loss: 0.3431461332440379



    
![png](output_31_442.png)
    



    
![png](output_31_443.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 55000: Generator loss: 1.8291386106014276, discriminator loss: 0.3498588621616359



    
![png](output_31_446.png)
    



    
![png](output_31_447.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 55500: Generator loss: 1.7847232661247254, discriminator loss: 0.3612832996845245



    
![png](output_31_450.png)
    



    
![png](output_31_451.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 56000: Generator loss: 1.8189983024597174, discriminator loss: 0.3393006066679958



    
![png](output_31_454.png)
    



    
![png](output_31_455.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 56500: Generator loss: 1.8988943800926203, discriminator loss: 0.3284487442672251



    
![png](output_31_458.png)
    



    
![png](output_31_459.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 57000: Generator loss: 1.8129437012672427, discriminator loss: 0.3609837980866431



    
![png](output_31_462.png)
    



    
![png](output_31_463.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 57500: Generator loss: 1.755611210346221, discriminator loss: 0.36340717953443497



    
![png](output_31_466.png)
    



    
![png](output_31_467.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 58000: Generator loss: 1.7487610211372397, discriminator loss: 0.36521679112315136



    
![png](output_31_470.png)
    



    
![png](output_31_471.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 58500: Generator loss: 1.8261974825859077, discriminator loss: 0.33875518929958376



    
![png](output_31_474.png)
    



    
![png](output_31_475.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 59000: Generator loss: 1.8035872306823717, discriminator loss: 0.35497032463550543



    
![png](output_31_478.png)
    



    
![png](output_31_479.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 59500: Generator loss: 1.7606685762405414, discriminator loss: 0.36261354196071627



    
![png](output_31_482.png)
    



    
![png](output_31_483.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 60000: Generator loss: 1.701543086767197, discriminator loss: 0.38012831813097003



    
![png](output_31_486.png)
    



    
![png](output_31_487.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 60500: Generator loss: 1.7103163278102873, discriminator loss: 0.3635235675573351



    
![png](output_31_490.png)
    



    
![png](output_31_491.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 61000: Generator loss: 1.623123787641526, discriminator loss: 0.407767561197281



    
![png](output_31_495.png)
    



    
![png](output_31_496.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 61500: Generator loss: 1.6501539266109457, discriminator loss: 0.39159765499830224



    
![png](output_31_499.png)
    



    
![png](output_31_500.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 62000: Generator loss: 1.5739944827556611, discriminator loss: 0.4115478475093842



    
![png](output_31_503.png)
    



    
![png](output_31_504.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 62500: Generator loss: 1.5966340360641487, discriminator loss: 0.4031843383312229



    
![png](output_31_507.png)
    



    
![png](output_31_508.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 63000: Generator loss: 1.586186097621916, discriminator loss: 0.39390349352359805



    
![png](output_31_511.png)
    



    
![png](output_31_512.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 63500: Generator loss: 1.529742845535278, discriminator loss: 0.4191225972175602



    
![png](output_31_515.png)
    



    
![png](output_31_516.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 64000: Generator loss: 1.509330605506897, discriminator loss: 0.4326053431630135



    
![png](output_31_519.png)
    



    
![png](output_31_520.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 64500: Generator loss: 1.5253307085037229, discriminator loss: 0.4192695360183719



    
![png](output_31_523.png)
    



    
![png](output_31_524.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 65000: Generator loss: 1.423358156204224, discriminator loss: 0.442434999525547



    
![png](output_31_527.png)
    



    
![png](output_31_528.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 65500: Generator loss: 1.4192569477558141, discriminator loss: 0.4484421516656875



    
![png](output_31_531.png)
    



    
![png](output_31_532.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 66000: Generator loss: 1.4369800877571102, discriminator loss: 0.43692900788783984



    
![png](output_31_535.png)
    



    
![png](output_31_536.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 66500: Generator loss: 1.4472659385204307, discriminator loss: 0.4432972142696379



    
![png](output_31_539.png)
    



    
![png](output_31_540.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 67000: Generator loss: 1.4747446553707129, discriminator loss: 0.42481861478090266



    
![png](output_31_543.png)
    



    
![png](output_31_544.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 67500: Generator loss: 1.4075598037242898, discriminator loss: 0.4532960081100466



    
![png](output_31_547.png)
    



    
![png](output_31_548.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 68000: Generator loss: 1.3855331835746774, discriminator loss: 0.4527249490022663



    
![png](output_31_551.png)
    



    
![png](output_31_552.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 68500: Generator loss: 1.3821633167266831, discriminator loss: 0.45176227378845213



    
![png](output_31_556.png)
    



    
![png](output_31_557.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 69000: Generator loss: 1.3033752326965327, discriminator loss: 0.46847189158201225



    
![png](output_31_560.png)
    



    
![png](output_31_561.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 69500: Generator loss: 1.377162681102752, discriminator loss: 0.4391334266066553



    
![png](output_31_564.png)
    



    
![png](output_31_565.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 70000: Generator loss: 1.410618436098099, discriminator loss: 0.4375912361741066



    
![png](output_31_568.png)
    



    
![png](output_31_569.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 70500: Generator loss: 1.3445320518016812, discriminator loss: 0.4522381857037541



    
![png](output_31_572.png)
    



    
![png](output_31_573.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 71000: Generator loss: 1.373481007575989, discriminator loss: 0.4515402186512952



    
![png](output_31_576.png)
    



    
![png](output_31_577.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 71500: Generator loss: 1.4048119382858275, discriminator loss: 0.4365125617384912



    
![png](output_31_580.png)
    



    
![png](output_31_581.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 72000: Generator loss: 1.425702041387558, discriminator loss: 0.4399210355877876



    
![png](output_31_584.png)
    



    
![png](output_31_585.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 72500: Generator loss: 1.4579155290126797, discriminator loss: 0.4208141348361971



    
![png](output_31_588.png)
    



    
![png](output_31_589.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 73000: Generator loss: 1.382049505710603, discriminator loss: 0.43430117094516757



    
![png](output_31_592.png)
    



    
![png](output_31_593.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 73500: Generator loss: 1.4475671951770794, discriminator loss: 0.42260916304588325



    
![png](output_31_596.png)
    



    
![png](output_31_597.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 74000: Generator loss: 1.4484017267227165, discriminator loss: 0.4268757753968241



    
![png](output_31_600.png)
    



    
![png](output_31_601.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 74500: Generator loss: 1.3974838781356798, discriminator loss: 0.43637293577194197



    
![png](output_31_604.png)
    



    
![png](output_31_605.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 75000: Generator loss: 1.4281336755752558, discriminator loss: 0.44354641628265434



    
![png](output_31_608.png)
    



    
![png](output_31_609.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 75500: Generator loss: 1.3242149069309226, discriminator loss: 0.4763226152062419



    
![png](output_31_612.png)
    



    
![png](output_31_613.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 76000: Generator loss: 1.3270197205543528, discriminator loss: 0.4586185787320143



    
![png](output_31_617.png)
    



    
![png](output_31_618.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 76500: Generator loss: 1.3117838754653928, discriminator loss: 0.4621535071134571



    
![png](output_31_621.png)
    



    
![png](output_31_622.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 77000: Generator loss: 1.3311121463775615, discriminator loss: 0.455997407257557



    
![png](output_31_625.png)
    



    
![png](output_31_626.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 77500: Generator loss: 1.342749555110931, discriminator loss: 0.4663026410937309



    
![png](output_31_629.png)
    



    
![png](output_31_630.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 78000: Generator loss: 1.371066940069199, discriminator loss: 0.4368144698739054



    
![png](output_31_633.png)
    



    
![png](output_31_634.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 78500: Generator loss: 1.3327529888153073, discriminator loss: 0.4639427134990694



    
![png](output_31_637.png)
    



    
![png](output_31_638.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 79000: Generator loss: 1.2576579401493064, discriminator loss: 0.4829175068140033



    
![png](output_31_641.png)
    



    
![png](output_31_642.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 79500: Generator loss: 1.2793714666366574, discriminator loss: 0.4767866529226304



    
![png](output_31_645.png)
    



    
![png](output_31_646.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 80000: Generator loss: 1.2299753348827365, discriminator loss: 0.4979269800782206



    
![png](output_31_649.png)
    



    
![png](output_31_650.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 80500: Generator loss: 1.2903002877235425, discriminator loss: 0.4591439729928968



    
![png](output_31_653.png)
    



    
![png](output_31_654.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 81000: Generator loss: 1.3037826921939846, discriminator loss: 0.469360141336918



    
![png](output_31_657.png)
    



    
![png](output_31_658.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 81500: Generator loss: 1.2644740183353436, discriminator loss: 0.492924510240555



    
![png](output_31_661.png)
    



    
![png](output_31_662.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 82000: Generator loss: 1.2670238947868346, discriminator loss: 0.4818728513121603



    
![png](output_31_665.png)
    



    
![png](output_31_666.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 82500: Generator loss: 1.2974353358745585, discriminator loss: 0.4746529617905617



    
![png](output_31_669.png)
    



    
![png](output_31_670.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 83000: Generator loss: 1.27791477560997, discriminator loss: 0.47518393349647564



    
![png](output_31_673.png)
    



    
![png](output_31_674.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 83500: Generator loss: 1.2260546128749852, discriminator loss: 0.5019592450261116



    
![png](output_31_678.png)
    



    
![png](output_31_679.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 84000: Generator loss: 1.350738450288773, discriminator loss: 0.4458018347620964



    
![png](output_31_682.png)
    



    
![png](output_31_683.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 84500: Generator loss: 1.31092030954361, discriminator loss: 0.46754373759031287



    
![png](output_31_686.png)
    



    
![png](output_31_687.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 85000: Generator loss: 1.249203630447387, discriminator loss: 0.4933890671133994



    
![png](output_31_690.png)
    



    
![png](output_31_691.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 85500: Generator loss: 1.3058379535675062, discriminator loss: 0.4658535462021831



    
![png](output_31_694.png)
    



    
![png](output_31_695.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 86000: Generator loss: 1.2785598299503325, discriminator loss: 0.47174331778287887



    
![png](output_31_698.png)
    



    
![png](output_31_699.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 86500: Generator loss: 1.263381775140762, discriminator loss: 0.4823970453739167



    
![png](output_31_702.png)
    



    
![png](output_31_703.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 87000: Generator loss: 1.2081541230678567, discriminator loss: 0.4942623887062074



    
![png](output_31_706.png)
    



    
![png](output_31_707.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 87500: Generator loss: 1.2386380643844617, discriminator loss: 0.4820726540684698



    
![png](output_31_710.png)
    



    
![png](output_31_711.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 88000: Generator loss: 1.21773440361023, discriminator loss: 0.5062047605514528



    
![png](output_31_714.png)
    



    
![png](output_31_715.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 88500: Generator loss: 1.2645729637145993, discriminator loss: 0.4788381667137144



    
![png](output_31_718.png)
    



    
![png](output_31_719.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 89000: Generator loss: 1.2110509026050562, discriminator loss: 0.49281109899282455



    
![png](output_31_722.png)
    



    
![png](output_31_723.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 89500: Generator loss: 1.2468588757514958, discriminator loss: 0.49285411775112176



    
![png](output_31_726.png)
    



    
![png](output_31_727.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 90000: Generator loss: 1.1508352646827689, discriminator loss: 0.5173188896179202



    
![png](output_31_730.png)
    



    
![png](output_31_731.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 90500: Generator loss: 1.2135836167335503, discriminator loss: 0.4877723878026005



    
![png](output_31_734.png)
    



    
![png](output_31_735.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 91000: Generator loss: 1.264163510799408, discriminator loss: 0.4808746671676635



    
![png](output_31_739.png)
    



    
![png](output_31_740.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 91500: Generator loss: 1.220516323328018, discriminator loss: 0.49061464852094644



    
![png](output_31_743.png)
    



    
![png](output_31_744.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 92000: Generator loss: 1.1885137395858767, discriminator loss: 0.4847811607718465



    
![png](output_31_747.png)
    



    
![png](output_31_748.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 92500: Generator loss: 1.2209673624038697, discriminator loss: 0.47813724541664077



    
![png](output_31_751.png)
    



    
![png](output_31_752.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 93000: Generator loss: 1.2467120130062115, discriminator loss: 0.46857058680057495



    
![png](output_31_755.png)
    



    
![png](output_31_756.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 93500: Generator loss: 1.243904872179031, discriminator loss: 0.4882635320425038



    
![png](output_31_759.png)
    



    
![png](output_31_760.png)
    


If you don't get any runtime error, it means that your code works. We check that the weights are changing in each iteration within the function.

**Congratulations, you have trained your first GAN**


```python

```
