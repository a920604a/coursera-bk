# Build a Conditional GAN

### Goals
In this notebook, you're going to make a conditional GAN in order to generate hand-written images of digits, conditioned on the digit to be generated (the class vector). This will let you choose what digit you want to generate.

You'll then do some exploration of the generated images to visualize what the noise and class vectors mean.  

### Learning Objectives
1.   Learn the technical difference between a conditional and unconditional GAN.
2.   Understand the distinction between the class and noise vector in a conditional GAN.



## Getting Started

For this assignment, you will be using the MNIST dataset again, but there's nothing stopping you from applying this generator code to produce images of animals conditioned on the species or pictures of faces conditioned on facial characteristics.

Note that this assignment requires no changes to the architectures of the generator or discriminator, only changes to the data passed to both. The generator will no longer take `z_dim` as an argument, but  `input_dim` instead, since you need to pass in both the noise and class vectors. In addition to good variable naming, this also means that you can use the generator and discriminator code you have previously written with different parameters.

You will begin by importing the necessary libraries and building the generator and discriminator.

#### Packages and Visualization


```python
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for our testing purposes, please do not change!

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()
```

#### Generator and Noise


```python
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        input_dim: the dimension of the input vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, input_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)

def get_noise(n_samples, input_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, input_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        input_dim: the dimension of the input vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, input_dim, device=device)
```

#### Discriminator


```python
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
      im_chan: the number of channels in the images, fitted for the dataset used, a scalar
            (MNIST is black-and-white, so 1 channel is your default)
      hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of the DCGAN; 
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
```

## Class Input

In conditional GANs, the input vector for the generator will also need to include the class information. The class is represented using a one-hot encoded vector where its length is the number of classes and each index represents a class. The vector is all 0's and a 1 on the chosen class. Given the labels of multiple images (e.g. from a batch) and number of classes, please create one-hot vectors for each label. There is a class within the PyTorch functional library that can help you.

<details>

<summary>
<font size="3" color="green">
<b>Optional hints for <code><font size="4">get_one_hot_labels</font></code></b>
</font>
</summary>

1.   This code can be done in one line.
2.   The documentation for [F.one_hot](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.one_hot) may be helpful.

</details>



```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_one_hot_labels

import torch.nn.functional as F
def get_one_hot_labels(labels, n_classes):
    '''
    Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
    Parameters:
        labels: tensor of labels from the dataloader, size (?)
        n_classes: the total number of classes in the dataset, an integer scalar
    '''
    #### START CODE HERE ####
    return F.one_hot(labels, n_classes)
    #### END CODE HERE ####
```


```python
assert (
    get_one_hot_labels(
        labels=torch.Tensor([[0, 2, 1]]).long(),
        n_classes=3
    ).tolist() == 
    [[
      [1, 0, 0], 
      [0, 0, 1], 
      [0, 1, 0]
    ]]
)
# Check that the device of get_one_hot_labels matches the input device
if torch.cuda.is_available():
    assert str(get_one_hot_labels(torch.Tensor([[0]]).long().cuda(), 1).device).startswith("cuda")
    
print("Success!")
```

    Success!


Next, you need to be able to concatenate the one-hot class vector to the noise vector before giving it to the generator. You will also need to do this when adding the class channels to the discriminator.

To do this, you will need to write a function that combines two vectors. Remember that you need to ensure that the vectors are the same type: floats. Again, you can look to the PyTorch library for help.
<details>
<summary>
<font size="3" color="green">
<b>Optional hints for <code><font size="4">combine_vectors</font></code></b>
</font>
</summary>

1.   This code can also be written in one line.
2.   The documentation for [torch.cat](https://pytorch.org/docs/master/generated/torch.cat.html) may be helpful.
3.   Specifically, you might want to look at what the `dim` argument of `torch.cat` does.

</details>



```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: combine_vectors
def combine_vectors(x, y):
    '''
    Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
    Parameters:
      x: (n_samples, ?) the first vector. 
        In this assignment, this will be the noise vector of shape (n_samples, z_dim), 
        but you shouldn't need to know the second dimension's size.
      y: (n_samples, ?) the second vector.
        Once again, in this assignment this will be the one-hot class vector 
        with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
    '''
    # Note: Make sure this function outputs a float no matter what inputs it receives
    #### START CODE HERE ####
    combined = torch.cat((x.float(), y.float()),1)
    #### END CODE HERE ####
    return combined.float()
```


```python
combined = combine_vectors(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]]))
if torch.cuda.is_available():
    # Check that it doesn't break with cuda
    cuda_check = combine_vectors(torch.tensor([[1, 2], [3, 4]]).cuda(), torch.tensor([[5, 6], [7, 8]]).cuda())
    assert str(cuda_check.device).startswith("cuda")
# Check exact order of elements
assert torch.all(combined == torch.tensor([[1, 2, 5, 6], [3, 4, 7, 8]]))
# Tests that items are of float type
assert (type(combined[0][0].item()) == float)
# Check shapes
combined = combine_vectors(torch.randn(1, 4, 5), torch.randn(1, 8, 5));
assert tuple(combined.shape) == (1, 12, 5)
assert tuple(combine_vectors(torch.randn(1, 10, 12).long(), torch.randn(1, 20, 12).long()).shape) == (1, 30, 12)
# Check that the float transformation doesn't happen after the inputs are concatenated
assert tuple(combine_vectors(torch.randn(1, 10, 12).long(), torch.randn(1, 20, 12)).shape) == (1, 30, 12)
print("Success!")
```

    Success!


## Training
Now you can start to put it all together!
First, you will define some new parameters:

*   mnist_shape: the number of pixels in each MNIST image, which has dimensions 28 x 28 and one channel (because it's black-and-white) so 1 x 28 x 28
*   n_classes: the number of classes in MNIST (10, since there are the digits from 0 to 9)


```python
mnist_shape = (1, 28, 28)
n_classes = 10
```

And you also include the same parameters from previous assignments:

  *   criterion: the loss function
  *   n_epochs: the number of times you iterate through the entire dataset when training
  *   z_dim: the dimension of the noise vector
  *   display_step: how often to display/visualize the images
  *   batch_size: the number of images per forward/backward pass
  *   lr: the learning rate
  *   device: the device type



```python
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.0002
device = 'cuda'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST('.', download=False, transform=transform),
    batch_size=batch_size,
    shuffle=True)
```

Then, you can initialize your generator, discriminator, and optimizers. To do this, you will need to update the input dimensions for both models. For the generator, you will need to calculate the size of the input vector; recall that for conditional GANs, the generator's input is the noise vector concatenated with the class vector. For the discriminator, you need to add a channel for every class.


```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_input_dimensions
def get_input_dimensions(z_dim, mnist_shape, n_classes):
    '''
    Function for getting the size of the conditional input dimensions 
    from z_dim, the image shape, and number of classes.
    Parameters:
        z_dim: the dimension of the noise vector, a scalar
        mnist_shape: the shape of each MNIST image as (C, W, H), which is (1, 28, 28)
        n_classes: the total number of classes in the dataset, an integer scalar
                (10 for MNIST)
    Returns: 
        generator_input_dim: the input dimensionality of the conditional generator, 
                          which takes the noise and class vectors
        discriminator_im_chan: the number of input channels to the discriminator
                            (e.g. C x 28 x 28 for MNIST)
    '''
    #### START CODE HERE ####
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = mnist_shape[0] + n_classes
    #### END CODE HERE ####
    return generator_input_dim, discriminator_im_chan
```


```python
def test_input_dims():
    gen_dim, disc_dim = get_input_dimensions(23, (12, 23, 52), 9)
    assert gen_dim == 32
    assert disc_dim == 21
test_input_dims()
print("Success!")
```

    Success!



```python
generator_input_dim, discriminator_im_chan = get_input_dimensions(z_dim, mnist_shape, n_classes)

gen = Generator(input_dim=generator_input_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(im_chan=discriminator_im_chan).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)
```

Now to train, you would like both your generator and your discriminator to know what class of image should be generated. There are a few locations where you will need to implement code.

For example, if you're generating a picture of the number "1", you would need to:
  
1.   Tell that to the generator, so that it knows it should be generating a "1"
2.   Tell that to the discriminator, so that it knows it should be looking at a "1". If the discriminator is told it should be looking at a 1 but sees something that's clearly an 8, it can guess that it's probably fake

There are no explicit unit tests here -- if this block of code runs and you don't change any of the other variables, then you've done it correctly!


```python
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CELL
cur_step = 0
generator_losses = []
discriminator_losses = []

#UNIT TEST NOTE: Initializations needed for grading
noise_and_labels = False
fake = False

fake_image_and_labels = False
real_image_and_labels = False
disc_fake_pred = False
disc_real_pred = False

for epoch in range(n_epochs):
    # Dataloader returns the batches and the labels
    for real, labels in tqdm(dataloader):
        cur_batch_size = len(real)
        # Flatten the batch of real images from the dataset
        real = real.to(device)

        one_hot_labels = get_one_hot_labels(labels.to(device), n_classes)
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, mnist_shape[1], mnist_shape[2])

        ### Update discriminator ###
        # Zero out the discriminator gradients
        disc_opt.zero_grad()
        # Get noise corresponding to the current batch_size 
        fake_noise = get_noise(cur_batch_size, z_dim, device=device)
        
        # Now you can get the images from the generator
        # Steps: 1) Combine the noise vectors and the one-hot labels for the generator
        #        2) Generate the conditioned fake images
       
        #### START CODE HERE ####
        noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
        fake = gen(noise_and_labels)
        #### END CODE HERE ####
        
        # Make sure that enough images were generated
        assert len(fake) == len(real)
        # Check that correct tensors were combined
        assert tuple(noise_and_labels.shape) == (cur_batch_size, fake_noise.shape[1] + one_hot_labels.shape[1])
        # It comes from the correct generator
        assert tuple(fake.shape) == (len(real), 1, 28, 28)

        # Now you can get the predictions from the discriminator
        # Steps: 1) Create the input for the discriminator
        #           a) Combine the fake images with image_one_hot_labels, 
        #              remember to detach the generator (.detach()) so you do not backpropagate through it
        #           b) Combine the real images with image_one_hot_labels
        #        2) Get the discriminator's prediction on the fakes as disc_fake_pred
        #        3) Get the discriminator's prediction on the reals as disc_real_pred
        
        #### START CODE HERE ####
        fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
        real_image_and_labels = combine_vectors(real, image_one_hot_labels)
        disc_fake_pred = disc(fake_image_and_labels.detach())
        disc_real_pred = disc(real_image_and_labels)
        #### END CODE HERE ####
        
        # Make sure shapes are correct 
        assert tuple(fake_image_and_labels.shape) == (len(real), fake.detach().shape[1] + image_one_hot_labels.shape[1], 28 ,28)
        assert tuple(real_image_and_labels.shape) == (len(real), real.shape[1] + image_one_hot_labels.shape[1], 28 ,28)
        # Make sure that enough predictions were made
        assert len(disc_real_pred) == len(real)
        # Make sure that the inputs are different
        assert torch.any(fake_image_and_labels != real_image_and_labels)
        # Shapes must match
        assert tuple(fake_image_and_labels.shape) == tuple(real_image_and_labels.shape)
        assert tuple(disc_fake_pred.shape) == tuple(disc_real_pred.shape)
        
        
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        disc_loss.backward(retain_graph=True)
        disc_opt.step() 

        # Keep track of the average discriminator loss
        discriminator_losses += [disc_loss.item()]

        ### Update generator ###
        # Zero out the generator gradients
        gen_opt.zero_grad()

        fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
        # This will error if you didn't concatenate your labels to your image correctly
        disc_fake_pred = disc(fake_image_and_labels)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the generator losses
        generator_losses += [gen_loss.item()]
        #

        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            disc_mean = sum(discriminator_losses[-display_step:]) / display_step
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {gen_mean}, discriminator loss: {disc_mean}")
            show_tensor_images(fake)
            show_tensor_images(real)
            step_bins = 20
            x_axis = sorted([i * step_bins for i in range(len(generator_losses) // step_bins)] * step_bins)
            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Generator Loss"
            )
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(discriminator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Discriminator Loss"
            )
            plt.legend()
            plt.show()
        elif cur_step == 0:
            print("Congratulations! If you've gotten here, it's working. Please let this train until you're happy with how the generated numbers look, and then go on to the exploration!")
        cur_step += 1
```


      0%|          | 0/469 [00:00<?, ?it/s]


    Congratulations! If you've gotten here, it's working. Please let this train until you're happy with how the generated numbers look, and then go on to the exploration!



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 1, step 500: Generator loss: 2.365815122127533, discriminator loss: 0.23452139009162784



    
![png](output_24_4.png)
    



    
![png](output_24_5.png)
    



    
![png](output_24_6.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 2, step 1000: Generator loss: 4.210208248138428, discriminator loss: 0.03439730873890221



    
![png](output_24_9.png)
    



    
![png](output_24_10.png)
    



    
![png](output_24_11.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 3, step 1500: Generator loss: 4.710073138713836, discriminator loss: 0.03760864031128585



    
![png](output_24_14.png)
    



    
![png](output_24_15.png)
    



    
![png](output_24_16.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 4, step 2000: Generator loss: 3.3794925413131716, discriminator loss: 0.14948570676147938



    
![png](output_24_19.png)
    



    
![png](output_24_20.png)
    



    
![png](output_24_21.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 5, step 2500: Generator loss: 2.505614499092102, discriminator loss: 0.24794028165936471



    
![png](output_24_24.png)
    



    
![png](output_24_25.png)
    



    
![png](output_24_26.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 6, step 3000: Generator loss: 2.288471808671951, discriminator loss: 0.3177657506465912



    
![png](output_24_29.png)
    



    
![png](output_24_30.png)
    



    
![png](output_24_31.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 7, step 3500: Generator loss: 2.0368205959796906, discriminator loss: 0.32717025020718576



    
![png](output_24_34.png)
    



    
![png](output_24_35.png)
    



    
![png](output_24_36.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 8, step 4000: Generator loss: 2.0730760498046874, discriminator loss: 0.3561065137982368



    
![png](output_24_39.png)
    



    
![png](output_24_40.png)
    



    
![png](output_24_41.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 9, step 4500: Generator loss: 1.9229473404884339, discriminator loss: 0.33711373415589335



    
![png](output_24_44.png)
    



    
![png](output_24_45.png)
    



    
![png](output_24_46.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 10, step 5000: Generator loss: 1.917601254940033, discriminator loss: 0.3629780676662922



    
![png](output_24_49.png)
    



    
![png](output_24_50.png)
    



    
![png](output_24_51.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 11, step 5500: Generator loss: 1.8787400615215302, discriminator loss: 0.3962197976410389



    
![png](output_24_54.png)
    



    
![png](output_24_55.png)
    



    
![png](output_24_56.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 12, step 6000: Generator loss: 1.584790585398674, discriminator loss: 0.4241155064404011



    
![png](output_24_59.png)
    



    
![png](output_24_60.png)
    



    
![png](output_24_61.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 13, step 6500: Generator loss: 1.5379385780096053, discriminator loss: 0.42857546943426134



    
![png](output_24_64.png)
    



    
![png](output_24_65.png)
    



    
![png](output_24_66.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 14, step 7000: Generator loss: 1.4848047930002213, discriminator loss: 0.4499597182869911



    
![png](output_24_69.png)
    



    
![png](output_24_70.png)
    



    
![png](output_24_71.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 15, step 7500: Generator loss: 1.4547104108333588, discriminator loss: 0.4868628137111664



    
![png](output_24_74.png)
    



    
![png](output_24_75.png)
    



    
![png](output_24_76.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 17, step 8000: Generator loss: 1.3899183558225632, discriminator loss: 0.513661828815937



    
![png](output_24_80.png)
    



    
![png](output_24_81.png)
    



    
![png](output_24_82.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 18, step 8500: Generator loss: 1.3229641309976579, discriminator loss: 0.5237894829511642



    
![png](output_24_85.png)
    



    
![png](output_24_86.png)
    



    
![png](output_24_87.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 19, step 9000: Generator loss: 1.323524187207222, discriminator loss: 0.5264951602220536



    
![png](output_24_90.png)
    



    
![png](output_24_91.png)
    



    
![png](output_24_92.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 20, step 9500: Generator loss: 1.2578706163167954, discriminator loss: 0.5435332206487655



    
![png](output_24_95.png)
    



    
![png](output_24_96.png)
    



    
![png](output_24_97.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 21, step 10000: Generator loss: 1.2312291637659072, discriminator loss: 0.5402825145721436



    
![png](output_24_100.png)
    



    
![png](output_24_101.png)
    



    
![png](output_24_102.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 22, step 10500: Generator loss: 1.1742142330408096, discriminator loss: 0.5494932829737663



    
![png](output_24_105.png)
    



    
![png](output_24_106.png)
    



    
![png](output_24_107.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 23, step 11000: Generator loss: 1.1782477329969405, discriminator loss: 0.5511384994387627



    
![png](output_24_110.png)
    



    
![png](output_24_111.png)
    



    
![png](output_24_112.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 24, step 11500: Generator loss: 1.1908413441181183, discriminator loss: 0.5477168710231781



    
![png](output_24_115.png)
    



    
![png](output_24_116.png)
    



    
![png](output_24_117.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 25, step 12000: Generator loss: 1.175244507431984, discriminator loss: 0.5451523109078408



    
![png](output_24_120.png)
    



    
![png](output_24_121.png)
    



    
![png](output_24_122.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 26, step 12500: Generator loss: 1.2082747811079024, discriminator loss: 0.5542357576489448



    
![png](output_24_125.png)
    



    
![png](output_24_126.png)
    



    
![png](output_24_127.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 27, step 13000: Generator loss: 1.1401205087900161, discriminator loss: 0.5595799463391304



    
![png](output_24_130.png)
    



    
![png](output_24_131.png)
    



    
![png](output_24_132.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 28, step 13500: Generator loss: 1.1284845926761626, discriminator loss: 0.5621967079043388



    
![png](output_24_135.png)
    



    
![png](output_24_136.png)
    



    
![png](output_24_137.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 29, step 14000: Generator loss: 1.1076503278017045, discriminator loss: 0.5550114968419075



    
![png](output_24_140.png)
    



    
![png](output_24_141.png)
    



    
![png](output_24_142.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 30, step 14500: Generator loss: 1.1061525480747223, discriminator loss: 0.570388459444046



    
![png](output_24_145.png)
    



    
![png](output_24_146.png)
    



    
![png](output_24_147.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 31, step 15000: Generator loss: 1.1350401828289032, discriminator loss: 0.5662910235524178



    
![png](output_24_150.png)
    



    
![png](output_24_151.png)
    



    
![png](output_24_152.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 33, step 15500: Generator loss: 1.0893363620042802, discriminator loss: 0.5680054729580879



    
![png](output_24_156.png)
    



    
![png](output_24_157.png)
    



    
![png](output_24_158.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 34, step 16000: Generator loss: 1.096595168709755, discriminator loss: 0.5681150063276291



    
![png](output_24_161.png)
    



    
![png](output_24_162.png)
    



    
![png](output_24_163.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 35, step 16500: Generator loss: 1.0984941219091415, discriminator loss: 0.5702642297744751



    
![png](output_24_166.png)
    



    
![png](output_24_167.png)
    



    
![png](output_24_168.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 36, step 17000: Generator loss: 1.1051687115430833, discriminator loss: 0.5784681869745254



    
![png](output_24_171.png)
    



    
![png](output_24_172.png)
    



    
![png](output_24_173.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 37, step 17500: Generator loss: 1.052643830537796, discriminator loss: 0.5734514180421829



    
![png](output_24_176.png)
    



    
![png](output_24_177.png)
    



    
![png](output_24_178.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 38, step 18000: Generator loss: 1.0353193365335465, discriminator loss: 0.5762358205318451



    
![png](output_24_181.png)
    



    
![png](output_24_182.png)
    



    
![png](output_24_183.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 39, step 18500: Generator loss: 1.0589269555807113, discriminator loss: 0.5822852796316147



    
![png](output_24_186.png)
    



    
![png](output_24_187.png)
    



    
![png](output_24_188.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 40, step 19000: Generator loss: 1.0648753536939621, discriminator loss: 0.5788043640255928



    
![png](output_24_191.png)
    



    
![png](output_24_192.png)
    



    
![png](output_24_193.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 41, step 19500: Generator loss: 1.0443393121957778, discriminator loss: 0.5832441446185112



    
![png](output_24_196.png)
    



    
![png](output_24_197.png)
    



    
![png](output_24_198.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 42, step 20000: Generator loss: 1.060207589149475, discriminator loss: 0.5846630672216415



    
![png](output_24_201.png)
    



    
![png](output_24_202.png)
    



    
![png](output_24_203.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 43, step 20500: Generator loss: 1.0300032387971878, discriminator loss: 0.5788982088565826



    
![png](output_24_206.png)
    



    
![png](output_24_207.png)
    



    
![png](output_24_208.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 44, step 21000: Generator loss: 1.0429502416849137, discriminator loss: 0.583723005592823



    
![png](output_24_211.png)
    



    
![png](output_24_212.png)
    



    
![png](output_24_213.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 45, step 21500: Generator loss: 1.0443944412469863, discriminator loss: 0.5780628825426102



    
![png](output_24_216.png)
    



    
![png](output_24_217.png)
    



    
![png](output_24_218.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 46, step 22000: Generator loss: 1.028910937309265, discriminator loss: 0.5875479099154473



    
![png](output_24_221.png)
    



    
![png](output_24_222.png)
    



    
![png](output_24_223.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 47, step 22500: Generator loss: 1.051920481801033, discriminator loss: 0.5847301362156868



    
![png](output_24_226.png)
    



    
![png](output_24_227.png)
    



    
![png](output_24_228.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 49, step 23000: Generator loss: 1.0443498636484145, discriminator loss: 0.5825104997754097



    
![png](output_24_232.png)
    



    
![png](output_24_233.png)
    



    
![png](output_24_234.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 50, step 23500: Generator loss: 1.0226171932220458, discriminator loss: 0.5834506093859673



    
![png](output_24_237.png)
    



    
![png](output_24_238.png)
    



    
![png](output_24_239.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 51, step 24000: Generator loss: 1.0333442984819412, discriminator loss: 0.5827656745910644



    
![png](output_24_242.png)
    



    
![png](output_24_243.png)
    



    
![png](output_24_244.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 52, step 24500: Generator loss: 1.0335327694416045, discriminator loss: 0.5801970569491386



    
![png](output_24_247.png)
    



    
![png](output_24_248.png)
    



    
![png](output_24_249.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 53, step 25000: Generator loss: 1.0359509263038635, discriminator loss: 0.5891560164093971



    
![png](output_24_252.png)
    



    
![png](output_24_253.png)
    



    
![png](output_24_254.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 54, step 25500: Generator loss: 1.0665901299715042, discriminator loss: 0.5835553407669067



    
![png](output_24_257.png)
    



    
![png](output_24_258.png)
    



    
![png](output_24_259.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 55, step 26000: Generator loss: 1.0459110740423203, discriminator loss: 0.5834536117911339



    
![png](output_24_262.png)
    



    
![png](output_24_263.png)
    



    
![png](output_24_264.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 56, step 26500: Generator loss: 1.0266811985969544, discriminator loss: 0.5837144915461541



    
![png](output_24_267.png)
    



    
![png](output_24_268.png)
    



    
![png](output_24_269.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 57, step 27000: Generator loss: 1.0162328461408614, discriminator loss: 0.5941539855003357



    
![png](output_24_272.png)
    



    
![png](output_24_273.png)
    



    
![png](output_24_274.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 58, step 27500: Generator loss: 1.025822727560997, discriminator loss: 0.5841146700382233



    
![png](output_24_277.png)
    



    
![png](output_24_278.png)
    



    
![png](output_24_279.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 59, step 28000: Generator loss: 1.0343683532476424, discriminator loss: 0.5812875036001205



    
![png](output_24_282.png)
    



    
![png](output_24_283.png)
    



    
![png](output_24_284.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 60, step 28500: Generator loss: 1.0499597294330596, discriminator loss: 0.587694275200367



    
![png](output_24_287.png)
    



    
![png](output_24_288.png)
    



    
![png](output_24_289.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 61, step 29000: Generator loss: 1.0420405860543251, discriminator loss: 0.5846294738650322



    
![png](output_24_292.png)
    



    
![png](output_24_293.png)
    



    
![png](output_24_294.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 62, step 29500: Generator loss: 1.0299042001962662, discriminator loss: 0.5881598376631737



    
![png](output_24_297.png)
    



    
![png](output_24_298.png)
    



    
![png](output_24_299.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 63, step 30000: Generator loss: 1.0449653830528258, discriminator loss: 0.5905366259217262



    
![png](output_24_302.png)
    



    
![png](output_24_303.png)
    



    
![png](output_24_304.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 65, step 30500: Generator loss: 1.0341917291879654, discriminator loss: 0.589056861281395



    
![png](output_24_308.png)
    



    
![png](output_24_309.png)
    



    
![png](output_24_310.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 66, step 31000: Generator loss: 1.0194755026102067, discriminator loss: 0.5959141938090324



    
![png](output_24_313.png)
    



    
![png](output_24_314.png)
    



    
![png](output_24_315.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 67, step 31500: Generator loss: 1.018331073284149, discriminator loss: 0.5887036605477333



    
![png](output_24_318.png)
    



    
![png](output_24_319.png)
    



    
![png](output_24_320.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 68, step 32000: Generator loss: 1.0174207475185395, discriminator loss: 0.5882649145126343



    
![png](output_24_323.png)
    



    
![png](output_24_324.png)
    



    
![png](output_24_325.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 69, step 32500: Generator loss: 1.0165395605564118, discriminator loss: 0.5900672981142998



    
![png](output_24_328.png)
    



    
![png](output_24_329.png)
    



    
![png](output_24_330.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 70, step 33000: Generator loss: 0.999466784119606, discriminator loss: 0.5906392480731011



    
![png](output_24_333.png)
    



    
![png](output_24_334.png)
    



    
![png](output_24_335.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 71, step 33500: Generator loss: 0.9872718381881714, discriminator loss: 0.5933350747823716



    
![png](output_24_338.png)
    



    
![png](output_24_339.png)
    



    
![png](output_24_340.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 72, step 34000: Generator loss: 1.004344433784485, discriminator loss: 0.5937534298896789



    
![png](output_24_343.png)
    



    
![png](output_24_344.png)
    



    
![png](output_24_345.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 73, step 34500: Generator loss: 1.0437189137935639, discriminator loss: 0.5870669107437134



    
![png](output_24_348.png)
    



    
![png](output_24_349.png)
    



    
![png](output_24_350.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 74, step 35000: Generator loss: 1.0052263424396515, discriminator loss: 0.5883986117243767



    
![png](output_24_353.png)
    



    
![png](output_24_354.png)
    



    
![png](output_24_355.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 75, step 35500: Generator loss: 0.9943038059473037, discriminator loss: 0.5905092507600784



    
![png](output_24_358.png)
    



    
![png](output_24_359.png)
    



    
![png](output_24_360.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 76, step 36000: Generator loss: 1.012526062488556, discriminator loss: 0.590594360947609



    
![png](output_24_363.png)
    



    
![png](output_24_364.png)
    



    
![png](output_24_365.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 77, step 36500: Generator loss: 1.0246544107198716, discriminator loss: 0.5889714630842209



    
![png](output_24_368.png)
    



    
![png](output_24_369.png)
    



    
![png](output_24_370.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 78, step 37000: Generator loss: 1.0056278373003007, discriminator loss: 0.5991963036060334



    
![png](output_24_373.png)
    



    
![png](output_24_374.png)
    



    
![png](output_24_375.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 79, step 37500: Generator loss: 1.0055951414108277, discriminator loss: 0.5930276512503624



    
![png](output_24_378.png)
    



    
![png](output_24_379.png)
    



    
![png](output_24_380.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 81, step 38000: Generator loss: 1.0009749817848206, discriminator loss: 0.5871532409191131



    
![png](output_24_384.png)
    



    
![png](output_24_385.png)
    



    
![png](output_24_386.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 82, step 38500: Generator loss: 1.0327025705575943, discriminator loss: 0.5865820544958115



    
![png](output_24_389.png)
    



    
![png](output_24_390.png)
    



    
![png](output_24_391.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 83, step 39000: Generator loss: 1.0096050463914872, discriminator loss: 0.5920010585784912



    
![png](output_24_394.png)
    



    
![png](output_24_395.png)
    



    
![png](output_24_396.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 84, step 39500: Generator loss: 1.0045590969324112, discriminator loss: 0.5889269674420357



    
![png](output_24_399.png)
    



    
![png](output_24_400.png)
    



    
![png](output_24_401.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 85, step 40000: Generator loss: 1.0077891398668288, discriminator loss: 0.5906827850937844



    
![png](output_24_404.png)
    



    
![png](output_24_405.png)
    



    
![png](output_24_406.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 86, step 40500: Generator loss: 1.013230722308159, discriminator loss: 0.591272108912468



    
![png](output_24_409.png)
    



    
![png](output_24_410.png)
    



    
![png](output_24_411.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 87, step 41000: Generator loss: 1.003459321141243, discriminator loss: 0.5854612689018249



    
![png](output_24_414.png)
    



    
![png](output_24_415.png)
    



    
![png](output_24_416.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 88, step 41500: Generator loss: 0.9943310878276825, discriminator loss: 0.5888969164490699



    
![png](output_24_419.png)
    



    
![png](output_24_420.png)
    



    
![png](output_24_421.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 89, step 42000: Generator loss: 1.0187220697402954, discriminator loss: 0.594993172109127



    
![png](output_24_424.png)
    



    
![png](output_24_425.png)
    



    
![png](output_24_426.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 90, step 42500: Generator loss: 1.005424788236618, discriminator loss: 0.5937874577641488



    
![png](output_24_429.png)
    



    
![png](output_24_430.png)
    



    
![png](output_24_431.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 91, step 43000: Generator loss: 0.9916089347600937, discriminator loss: 0.5930819407701492



    
![png](output_24_434.png)
    



    
![png](output_24_435.png)
    



    
![png](output_24_436.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 92, step 43500: Generator loss: 1.0200039474964142, discriminator loss: 0.5869768711328507



    
![png](output_24_439.png)
    



    
![png](output_24_440.png)
    



    
![png](output_24_441.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 93, step 44000: Generator loss: 0.9921847386360169, discriminator loss: 0.5904656807780266



    
![png](output_24_444.png)
    



    
![png](output_24_445.png)
    



    
![png](output_24_446.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 94, step 44500: Generator loss: 1.0193623796701432, discriminator loss: 0.5886311857700348



    
![png](output_24_449.png)
    



    
![png](output_24_450.png)
    



    
![png](output_24_451.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 95, step 45000: Generator loss: 1.0027117849588394, discriminator loss: 0.5902761861681938



    
![png](output_24_454.png)
    



    
![png](output_24_455.png)
    



    
![png](output_24_456.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 97, step 45500: Generator loss: 0.9861012742519378, discriminator loss: 0.5927371917963028



    
![png](output_24_460.png)
    



    
![png](output_24_461.png)
    



    
![png](output_24_462.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 98, step 46000: Generator loss: 1.006313626050949, discriminator loss: 0.5930047160983085



    
![png](output_24_465.png)
    



    
![png](output_24_466.png)
    



    
![png](output_24_467.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 99, step 46500: Generator loss: 1.0056643015146256, discriminator loss: 0.5951471127867699



    
![png](output_24_470.png)
    



    
![png](output_24_471.png)
    



    
![png](output_24_472.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 100, step 47000: Generator loss: 1.0400250844955445, discriminator loss: 0.5817723087072373



    
![png](output_24_475.png)
    



    
![png](output_24_476.png)
    



    
![png](output_24_477.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 101, step 47500: Generator loss: 1.0216832772493363, discriminator loss: 0.5935589346885681



    
![png](output_24_480.png)
    



    
![png](output_24_481.png)
    



    
![png](output_24_482.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 102, step 48000: Generator loss: 1.0022087227106093, discriminator loss: 0.5932522151470184



    
![png](output_24_485.png)
    



    
![png](output_24_486.png)
    



    
![png](output_24_487.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 103, step 48500: Generator loss: 1.0246957265138625, discriminator loss: 0.5895025143027306



    
![png](output_24_490.png)
    



    
![png](output_24_491.png)
    



    
![png](output_24_492.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 104, step 49000: Generator loss: 1.0141848456859588, discriminator loss: 0.5876082508563996



    
![png](output_24_495.png)
    



    
![png](output_24_496.png)
    



    
![png](output_24_497.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 105, step 49500: Generator loss: 1.0262594512701035, discriminator loss: 0.5886790113449096



    
![png](output_24_500.png)
    



    
![png](output_24_501.png)
    



    
![png](output_24_502.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 106, step 50000: Generator loss: 1.0180598672628403, discriminator loss: 0.5921548217535019



    
![png](output_24_505.png)
    



    
![png](output_24_506.png)
    



    
![png](output_24_507.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 107, step 50500: Generator loss: 0.9962502681016921, discriminator loss: 0.593544638633728



    
![png](output_24_510.png)
    



    
![png](output_24_511.png)
    



    
![png](output_24_512.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 108, step 51000: Generator loss: 0.9939402619600296, discriminator loss: 0.5904114077091217



    
![png](output_24_515.png)
    



    
![png](output_24_516.png)
    



    
![png](output_24_517.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 109, step 51500: Generator loss: 1.0147280000448227, discriminator loss: 0.5884272117614746



    
![png](output_24_520.png)
    



    
![png](output_24_521.png)
    



    
![png](output_24_522.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 110, step 52000: Generator loss: 1.003916969180107, discriminator loss: 0.5902849839925766



    
![png](output_24_525.png)
    



    
![png](output_24_526.png)
    



    
![png](output_24_527.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 111, step 52500: Generator loss: 1.0228106182813645, discriminator loss: 0.5961541908979416



    
![png](output_24_530.png)
    



    
![png](output_24_531.png)
    



    
![png](output_24_532.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 113, step 53000: Generator loss: 1.0090177037715913, discriminator loss: 0.5880640907883644



    
![png](output_24_536.png)
    



    
![png](output_24_537.png)
    



    
![png](output_24_538.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 114, step 53500: Generator loss: 1.0200214551687241, discriminator loss: 0.5852308292984962



    
![png](output_24_541.png)
    



    
![png](output_24_542.png)
    



    
![png](output_24_543.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 115, step 54000: Generator loss: 1.010003294110298, discriminator loss: 0.5796048657298088



    
![png](output_24_546.png)
    



    
![png](output_24_547.png)
    



    
![png](output_24_548.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 116, step 54500: Generator loss: 1.0272772226333617, discriminator loss: 0.5845484637618065



    
![png](output_24_551.png)
    



    
![png](output_24_552.png)
    



    
![png](output_24_553.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 117, step 55000: Generator loss: 1.0195015397071838, discriminator loss: 0.5842510450482369



    
![png](output_24_556.png)
    



    
![png](output_24_557.png)
    



    
![png](output_24_558.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 118, step 55500: Generator loss: 1.0112336095571517, discriminator loss: 0.5928068357110023



    
![png](output_24_561.png)
    



    
![png](output_24_562.png)
    



    
![png](output_24_563.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 119, step 56000: Generator loss: 1.0164981350898743, discriminator loss: 0.5872695326209069



    
![png](output_24_566.png)
    



    
![png](output_24_567.png)
    



    
![png](output_24_568.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 120, step 56500: Generator loss: 1.0061778593063355, discriminator loss: 0.5889905136823654



    
![png](output_24_571.png)
    



    
![png](output_24_572.png)
    



    
![png](output_24_573.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 121, step 57000: Generator loss: 1.0219390313625336, discriminator loss: 0.5903721759319306



    
![png](output_24_576.png)
    



    
![png](output_24_577.png)
    



    
![png](output_24_578.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 122, step 57500: Generator loss: 1.0394862500429154, discriminator loss: 0.5832007101774216



    
![png](output_24_581.png)
    



    
![png](output_24_582.png)
    



    
![png](output_24_583.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 123, step 58000: Generator loss: 1.0200272674560547, discriminator loss: 0.5841398152709008



    
![png](output_24_586.png)
    



    
![png](output_24_587.png)
    



    
![png](output_24_588.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 124, step 58500: Generator loss: 1.0122340222597122, discriminator loss: 0.59067215102911



    
![png](output_24_591.png)
    



    
![png](output_24_592.png)
    



    
![png](output_24_593.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 125, step 59000: Generator loss: 1.0298925809860229, discriminator loss: 0.5849442155957222



    
![png](output_24_596.png)
    



    
![png](output_24_597.png)
    



    
![png](output_24_598.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 126, step 59500: Generator loss: 1.0153161140680314, discriminator loss: 0.5900761914849282



    
![png](output_24_601.png)
    



    
![png](output_24_602.png)
    



    
![png](output_24_603.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 127, step 60000: Generator loss: 1.0133797628879546, discriminator loss: 0.5867536931037903



    
![png](output_24_606.png)
    



    
![png](output_24_607.png)
    



    
![png](output_24_608.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 128, step 60500: Generator loss: 1.004065407037735, discriminator loss: 0.5877748013734817



    
![png](output_24_611.png)
    



    
![png](output_24_612.png)
    



    
![png](output_24_613.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 130, step 61000: Generator loss: 1.0164888522624969, discriminator loss: 0.5844663829803467



    
![png](output_24_617.png)
    



    
![png](output_24_618.png)
    



    
![png](output_24_619.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 131, step 61500: Generator loss: 1.0234656375646591, discriminator loss: 0.5823044225573539



    
![png](output_24_622.png)
    



    
![png](output_24_623.png)
    



    
![png](output_24_624.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 132, step 62000: Generator loss: 1.0547399146556855, discriminator loss: 0.5907284634709358



    
![png](output_24_627.png)
    



    
![png](output_24_628.png)
    



    
![png](output_24_629.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 133, step 62500: Generator loss: 1.0000185775756836, discriminator loss: 0.5926814821958541



    
![png](output_24_632.png)
    



    
![png](output_24_633.png)
    



    
![png](output_24_634.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 134, step 63000: Generator loss: 1.013117767214775, discriminator loss: 0.5875107944607735



    
![png](output_24_637.png)
    



    
![png](output_24_638.png)
    



    
![png](output_24_639.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 135, step 63500: Generator loss: 1.0144973140954971, discriminator loss: 0.5865109252333641



    
![png](output_24_642.png)
    



    
![png](output_24_643.png)
    



    
![png](output_24_644.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 136, step 64000: Generator loss: 1.0139959201812745, discriminator loss: 0.5841362013816833



    
![png](output_24_647.png)
    



    
![png](output_24_648.png)
    



    
![png](output_24_649.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 137, step 64500: Generator loss: 1.0295103254318236, discriminator loss: 0.5867808840870857



    
![png](output_24_652.png)
    



    
![png](output_24_653.png)
    



    
![png](output_24_654.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 138, step 65000: Generator loss: 1.0120015513896943, discriminator loss: 0.5918407678604126



    
![png](output_24_657.png)
    



    
![png](output_24_658.png)
    



    
![png](output_24_659.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 139, step 65500: Generator loss: 1.0256051408052445, discriminator loss: 0.5888157456517219



    
![png](output_24_662.png)
    



    
![png](output_24_663.png)
    



    
![png](output_24_664.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 140, step 66000: Generator loss: 1.0098456486463547, discriminator loss: 0.5869967831373215



    
![png](output_24_667.png)
    



    
![png](output_24_668.png)
    



    
![png](output_24_669.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 141, step 66500: Generator loss: 1.0337221087217332, discriminator loss: 0.5851074859499932



    
![png](output_24_672.png)
    



    
![png](output_24_673.png)
    



    
![png](output_24_674.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 142, step 67000: Generator loss: 1.0158808954954148, discriminator loss: 0.5877295281887055



    
![png](output_24_677.png)
    



    
![png](output_24_678.png)
    



    
![png](output_24_679.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 143, step 67500: Generator loss: 1.0113357504606246, discriminator loss: 0.5904845989942551



    
![png](output_24_682.png)
    



    
![png](output_24_683.png)
    



    
![png](output_24_684.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 144, step 68000: Generator loss: 0.9998085722923279, discriminator loss: 0.5909558398723602



    
![png](output_24_687.png)
    



    
![png](output_24_688.png)
    



    
![png](output_24_689.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 146, step 68500: Generator loss: 1.031954503417015, discriminator loss: 0.5858229861259461



    
![png](output_24_693.png)
    



    
![png](output_24_694.png)
    



    
![png](output_24_695.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 147, step 69000: Generator loss: 1.0094605860710144, discriminator loss: 0.5895318237543106



    
![png](output_24_698.png)
    



    
![png](output_24_699.png)
    



    
![png](output_24_700.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 148, step 69500: Generator loss: 1.0112643043994904, discriminator loss: 0.5885436331629753



    
![png](output_24_703.png)
    



    
![png](output_24_704.png)
    



    
![png](output_24_705.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 149, step 70000: Generator loss: 1.0380910954475402, discriminator loss: 0.5849322383403778



    
![png](output_24_708.png)
    



    
![png](output_24_709.png)
    



    
![png](output_24_710.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 150, step 70500: Generator loss: 1.0535721710920334, discriminator loss: 0.5926268690824509



    
![png](output_24_713.png)
    



    
![png](output_24_714.png)
    



    
![png](output_24_715.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 151, step 71000: Generator loss: 1.0160036910772323, discriminator loss: 0.5855192729830742



    
![png](output_24_718.png)
    



    
![png](output_24_719.png)
    



    
![png](output_24_720.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 152, step 71500: Generator loss: 1.0241395136117935, discriminator loss: 0.5914134519100189



    
![png](output_24_723.png)
    



    
![png](output_24_724.png)
    



    
![png](output_24_725.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 153, step 72000: Generator loss: 1.040040016889572, discriminator loss: 0.586236781835556



    
![png](output_24_728.png)
    



    
![png](output_24_729.png)
    



    
![png](output_24_730.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 154, step 72500: Generator loss: 1.0078105708360672, discriminator loss: 0.5867004963755608



    
![png](output_24_733.png)
    



    
![png](output_24_734.png)
    



    
![png](output_24_735.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 155, step 73000: Generator loss: 1.0111496396064759, discriminator loss: 0.59218505859375



    
![png](output_24_738.png)
    



    
![png](output_24_739.png)
    



    
![png](output_24_740.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 156, step 73500: Generator loss: 1.0210839964151381, discriminator loss: 0.5869369347691535



    
![png](output_24_743.png)
    



    
![png](output_24_744.png)
    



    
![png](output_24_745.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 157, step 74000: Generator loss: 0.9983591094017029, discriminator loss: 0.5940085656046867



    
![png](output_24_748.png)
    



    
![png](output_24_749.png)
    



    
![png](output_24_750.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 158, step 74500: Generator loss: 1.019736377596855, discriminator loss: 0.5895536653399468



    
![png](output_24_753.png)
    



    
![png](output_24_754.png)
    



    
![png](output_24_755.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 159, step 75000: Generator loss: 1.0321307941675186, discriminator loss: 0.5886060692071915



    
![png](output_24_758.png)
    



    
![png](output_24_759.png)
    



    
![png](output_24_760.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 160, step 75500: Generator loss: 1.0152143303155898, discriminator loss: 0.5894368638396263



    
![png](output_24_763.png)
    



    
![png](output_24_764.png)
    



    
![png](output_24_765.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 162, step 76000: Generator loss: 1.023012305021286, discriminator loss: 0.5920749220848084



    
![png](output_24_769.png)
    



    
![png](output_24_770.png)
    



    
![png](output_24_771.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 163, step 76500: Generator loss: 1.028085962176323, discriminator loss: 0.5917340404391289



    
![png](output_24_774.png)
    



    
![png](output_24_775.png)
    



    
![png](output_24_776.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 164, step 77000: Generator loss: 1.0013235046863556, discriminator loss: 0.5929797869920731



    
![png](output_24_779.png)
    



    
![png](output_24_780.png)
    



    
![png](output_24_781.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 165, step 77500: Generator loss: 1.013342220544815, discriminator loss: 0.5891543258428573



    
![png](output_24_784.png)
    



    
![png](output_24_785.png)
    



    
![png](output_24_786.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 166, step 78000: Generator loss: 1.0106794748306274, discriminator loss: 0.5965306566953659



    
![png](output_24_789.png)
    



    
![png](output_24_790.png)
    



    
![png](output_24_791.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 167, step 78500: Generator loss: 1.0177563719749452, discriminator loss: 0.5943069632053375



    
![png](output_24_794.png)
    



    
![png](output_24_795.png)
    



    
![png](output_24_796.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 168, step 79000: Generator loss: 1.0150098748207093, discriminator loss: 0.5881578136086464



    
![png](output_24_799.png)
    



    
![png](output_24_800.png)
    



    
![png](output_24_801.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 169, step 79500: Generator loss: 0.9912217171192169, discriminator loss: 0.5958423926234245



    
![png](output_24_804.png)
    



    
![png](output_24_805.png)
    



    
![png](output_24_806.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 170, step 80000: Generator loss: 0.9987383493185044, discriminator loss: 0.5932301429510116



    
![png](output_24_809.png)
    



    
![png](output_24_810.png)
    



    
![png](output_24_811.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 171, step 80500: Generator loss: 1.0287315063476563, discriminator loss: 0.5979215551018715



    
![png](output_24_814.png)
    



    
![png](output_24_815.png)
    



    
![png](output_24_816.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 172, step 81000: Generator loss: 1.0145618818998337, discriminator loss: 0.5977951684594154



    
![png](output_24_819.png)
    



    
![png](output_24_820.png)
    



    
![png](output_24_821.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 173, step 81500: Generator loss: 1.0077197346687317, discriminator loss: 0.5933164458274841



    
![png](output_24_824.png)
    



    
![png](output_24_825.png)
    



    
![png](output_24_826.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 174, step 82000: Generator loss: 1.0191299287080764, discriminator loss: 0.5889933585524559



    
![png](output_24_829.png)
    



    
![png](output_24_830.png)
    



    
![png](output_24_831.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 175, step 82500: Generator loss: 0.9985486156940461, discriminator loss: 0.5955166100263596



    
![png](output_24_834.png)
    



    
![png](output_24_835.png)
    



    
![png](output_24_836.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 176, step 83000: Generator loss: 1.027390226840973, discriminator loss: 0.5973404799699783



    
![png](output_24_839.png)
    



    
![png](output_24_840.png)
    



    
![png](output_24_841.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 178, step 83500: Generator loss: 1.0069014763832091, discriminator loss: 0.5960329038500786



    
![png](output_24_845.png)
    



    
![png](output_24_846.png)
    



    
![png](output_24_847.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 179, step 84000: Generator loss: 1.0090563267469406, discriminator loss: 0.5941139715313911



    
![png](output_24_850.png)
    



    
![png](output_24_851.png)
    



    
![png](output_24_852.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 180, step 84500: Generator loss: 1.025834595322609, discriminator loss: 0.5935550715327262



    
![png](output_24_855.png)
    



    
![png](output_24_856.png)
    



    
![png](output_24_857.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 181, step 85000: Generator loss: 0.9897033816576004, discriminator loss: 0.5961671417951584



    
![png](output_24_860.png)
    



    
![png](output_24_861.png)
    



    
![png](output_24_862.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 182, step 85500: Generator loss: 1.0207218356132508, discriminator loss: 0.5928991681933403



    
![png](output_24_865.png)
    



    
![png](output_24_866.png)
    



    
![png](output_24_867.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 183, step 86000: Generator loss: 1.0066205993890762, discriminator loss: 0.593928252518177



    
![png](output_24_870.png)
    



    
![png](output_24_871.png)
    



    
![png](output_24_872.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 184, step 86500: Generator loss: 1.007741155743599, discriminator loss: 0.6005500603318215



    
![png](output_24_875.png)
    



    
![png](output_24_876.png)
    



    
![png](output_24_877.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 185, step 87000: Generator loss: 1.012970511674881, discriminator loss: 0.5914468458294868



    
![png](output_24_880.png)
    



    
![png](output_24_881.png)
    



    
![png](output_24_882.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 186, step 87500: Generator loss: 0.9923973519802094, discriminator loss: 0.6039238372445106



    
![png](output_24_885.png)
    



    
![png](output_24_886.png)
    



    
![png](output_24_887.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 187, step 88000: Generator loss: 1.0058560127019882, discriminator loss: 0.5971718857288361



    
![png](output_24_890.png)
    



    
![png](output_24_891.png)
    



    
![png](output_24_892.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 188, step 88500: Generator loss: 0.9950532782077789, discriminator loss: 0.5960613418221473



    
![png](output_24_895.png)
    



    
![png](output_24_896.png)
    



    
![png](output_24_897.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 189, step 89000: Generator loss: 1.0235286303758622, discriminator loss: 0.5952605583071708



    
![png](output_24_900.png)
    



    
![png](output_24_901.png)
    



    
![png](output_24_902.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 190, step 89500: Generator loss: 1.0053212335109711, discriminator loss: 0.6011577704548836



    
![png](output_24_905.png)
    



    
![png](output_24_906.png)
    



    
![png](output_24_907.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 191, step 90000: Generator loss: 1.006945100903511, discriminator loss: 0.6016723192930221



    
![png](output_24_910.png)
    



    
![png](output_24_911.png)
    



    
![png](output_24_912.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 192, step 90500: Generator loss: 1.008920236825943, discriminator loss: 0.5978896546959876



    
![png](output_24_915.png)
    



    
![png](output_24_916.png)
    



    
![png](output_24_917.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 194, step 91000: Generator loss: 1.0227367652654649, discriminator loss: 0.5985640825033188



    
![png](output_24_921.png)
    



    
![png](output_24_922.png)
    



    
![png](output_24_923.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 195, step 91500: Generator loss: 1.0139871423244475, discriminator loss: 0.5960114967823028



    
![png](output_24_926.png)
    



    
![png](output_24_927.png)
    



    
![png](output_24_928.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 196, step 92000: Generator loss: 0.996145742058754, discriminator loss: 0.5992334386706353



    
![png](output_24_931.png)
    



    
![png](output_24_932.png)
    



    
![png](output_24_933.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 197, step 92500: Generator loss: 1.0041720930337905, discriminator loss: 0.5976423944830894



    
![png](output_24_936.png)
    



    
![png](output_24_937.png)
    



    
![png](output_24_938.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 198, step 93000: Generator loss: 0.9742332547903061, discriminator loss: 0.6012317734360695



    
![png](output_24_941.png)
    



    
![png](output_24_942.png)
    



    
![png](output_24_943.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Epoch 199, step 93500: Generator loss: 1.0215707128047944, discriminator loss: 0.5962182713150979



    
![png](output_24_946.png)
    



    
![png](output_24_947.png)
    



    
![png](output_24_948.png)
    


## Exploration
You can do a bit of exploration now!


```python
# Before you explore, you should put the generator
# in eval mode, both in general and so that batch norm
# doesn't cause you issues and is using its eval statistics
gen = gen.eval()
```

#### Changing the Class Vector
You can generate some numbers with your new model! You can add interpolation as well to make it more interesting.

So starting from a image, you will produce intermediate images that look more and more like the ending image until you get to the final image. Your're basically morphing one image into another. You can choose what these two images will be using your conditional GAN.


```python
import math

### Change me! ###
n_interpolation = 9 # Choose the interpolation: how many intermediate images you want + 2 (for the start and end image)
interpolation_noise = get_noise(1, z_dim, device=device).repeat(n_interpolation, 1)

def interpolate_class(first_number, second_number):
    first_label = get_one_hot_labels(torch.Tensor([first_number]).long(), n_classes)
    second_label = get_one_hot_labels(torch.Tensor([second_number]).long(), n_classes)

    # Calculate the interpolation vector between the two labels
    percent_second_label = torch.linspace(0, 1, n_interpolation)[:, None]
    interpolation_labels = first_label * (1 - percent_second_label) + second_label * percent_second_label

    # Combine the noise and the labels
    noise_and_labels = combine_vectors(interpolation_noise, interpolation_labels.to(device))
    fake = gen(noise_and_labels)
    show_tensor_images(fake, num_images=n_interpolation, nrow=int(math.sqrt(n_interpolation)), show=False)

### Change me! ###
start_plot_number = 1 # Choose the start digit
### Change me! ###
end_plot_number = 5 # Choose the end digit

plt.figure(figsize=(8, 8))
interpolate_class(start_plot_number, end_plot_number)
_ = plt.axis('off')

### Uncomment the following lines of code if you would like to visualize a set of pairwise class 
### interpolations for a collection of different numbers, all in a single grid of interpolations.
### You'll also see another visualization like this in the next code block!
# plot_numbers = [2, 3, 4, 5, 7]
# n_numbers = len(plot_numbers)
# plt.figure(figsize=(8, 8))
# for i, first_plot_number in enumerate(plot_numbers):
#     for j, second_plot_number in enumerate(plot_numbers):
#         plt.subplot(n_numbers, n_numbers, i * n_numbers + j + 1)
#         interpolate_class(first_plot_number, second_plot_number)
#         plt.axis('off')
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.1, wspace=0)
# plt.show()
# plt.close()
```

#### Changing the Noise Vector
Now, what happens if you hold the class constant, but instead you change the noise vector? You can also interpolate the noise vector and generate an image at each step.


```python
n_interpolation = 9 # How many intermediate images you want + 2 (for the start and end image)

# This time you're interpolating between the noise instead of the labels
interpolation_label = get_one_hot_labels(torch.Tensor([5]).long(), n_classes).repeat(n_interpolation, 1).float()

def interpolate_noise(first_noise, second_noise):
    # This time you're interpolating between the noise instead of the labels
    percent_first_noise = torch.linspace(0, 1, n_interpolation)[:, None].to(device)
    interpolation_noise = first_noise * percent_first_noise + second_noise * (1 - percent_first_noise)

    # Combine the noise and the labels again
    noise_and_labels = combine_vectors(interpolation_noise, interpolation_label.to(device))
    fake = gen(noise_and_labels)
    show_tensor_images(fake, num_images=n_interpolation, nrow=int(math.sqrt(n_interpolation)), show=False)

# Generate noise vectors to interpolate between
### Change me! ###
n_noise = 5 # Choose the number of noise examples in the grid
plot_noises = [get_noise(1, z_dim, device=device) for i in range(n_noise)]
plt.figure(figsize=(8, 8))
for i, first_plot_noise in enumerate(plot_noises):
    for j, second_plot_noise in enumerate(plot_noises):
        plt.subplot(n_noise, n_noise, i * n_noise + j + 1)
        interpolate_noise(first_plot_noise, second_plot_noise)
        plt.axis('off')
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.1, wspace=0)
plt.show()
plt.close()
```


    
![png](output_30_0.png)
    



    
![png](output_30_1.png)
    

