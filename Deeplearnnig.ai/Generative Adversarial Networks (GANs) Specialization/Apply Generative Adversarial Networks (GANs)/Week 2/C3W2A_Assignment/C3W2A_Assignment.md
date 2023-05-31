# U-Net

### Goals
In this notebook, you're going to implement a U-Net for a biomedical imaging segmentation task. Specifically, you're going to be labeling neurons, so one might call this a neural neural network! ;) 

Note that this is not a GAN, generative model, or unsupervised learning task. This is a supervised learning task, so there's only one correct answer (like a classifier!) You will see how this component underlies the Generator component of Pix2Pix in the next notebook this week.

### Learning Objectives
1.   Implement your own U-Net.
2.   Observe your U-Net's performance on a challenging segmentation task.


## Getting Started
You will start by importing libraries, defining a visualization function, and getting the neural dataset that you will be using.

#### Dataset
For this notebook, you will be using a dataset of electron microscopy
images and segmentation data. The information about the dataset you'll be using can be found [here](https://www.ini.uzh.ch/~acardona/data.html)! 

> Arganda-Carreras et al. "Crowdsourcing the creation of image
segmentation algorithms for connectomics". Front. Neuroanat. 2015. https://www.frontiersin.org/articles/10.3389/fnana.2015.00142/full

![dataset example](Neuraldatasetexample.png)



```python
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0)

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    # image_shifted = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
```

## U-Net Architecture
Now you can build your U-Net from its components. The figure below is from the paper, [*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597), by Ronneberger et al. 2015. It shows the U-Net architecture and how it contracts and then expands.

<!-- "[i]t consists of a contracting path (left side) and an expansive path (right side)" (Renneberger, 2015) -->

![Figure 1 from the paper, U-Net: Convolutional Networks for Biomedical Image Segmentation](https://drive.google.com/uc?export=view&id=1XgJRexE2CmsetRYyTLA7L8dsEwx7aQZY)

In other words, images are first fed through many convolutional layers which reduce height and width while increasing the channels, which the authors refer to as the "contracting path." For example, a set of two 2 x 2 convolutions with a stride of 2, will take a 1 x 28 x 28 (channels, height, width) grayscale image and result in a 2 x 14 x 14 representation. The "expanding path" does the opposite, gradually growing the image with fewer and fewer channels.

## Contracting Path
You will first implement the contracting blocks for the contracting path. This path is the encoder section of the U-Net, which has several downsampling steps as part of it. The authors give more detail of the remaining parts in the following paragraph from the paper (Renneberger, 2015):

>The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two 3 x 3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2 x 2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels.

<details>
<summary>
<font size="3" color="green">
<b>Optional hints for <code><font size="4">ContractingBlock</font></code></b>
</font>
</summary>

1.    Both convolutions should use 3 x 3 kernels.
2.    The max pool should use a 2 x 2 kernel with a stride 2.
</details>


```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: ContractingBlock
class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels):
        super(ContractingBlock, self).__init__()
        # You want to double the number of channels in the first convolution
        # and keep the same number of channels in the second.
        #### START CODE HERE ####
        self.conv1 = nn.Conv2d(input_channels, 2*input_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(2*input_channels, 2*input_channels, kernel_size=3)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #### END CODE HERE ####

    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock: 
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x
    
    # Required for grading
    def get_self(self):
        return self
```


```python
#UNIT TEST
def test_contracting_block(test_samples=100, test_channels=10, test_size=50):
    test_block = ContractingBlock(test_channels)
    test_in = torch.randn(test_samples, test_channels, test_size, test_size)
    test_out_conv1 = test_block.conv1(test_in)
    # Make sure that the first convolution has the right shape
    assert tuple(test_out_conv1.shape) == (test_samples, test_channels * 2, test_size - 2, test_size - 2)
    # Make sure that the right activation is used
    assert torch.all(test_block.activation(test_out_conv1) >= 0)
    assert torch.max(test_block.activation(test_out_conv1)) >= 1
    test_out_conv2 = test_block.conv2(test_out_conv1)
    # Make sure that the second convolution has the right shape
    assert tuple(test_out_conv2.shape) == (test_samples, test_channels * 2, test_size - 4, test_size - 4)
    test_out = test_block(test_in)
    # Make sure that the pooling has the right shape
    assert tuple(test_out.shape) == (test_samples, test_channels * 2, test_size // 2 - 2, test_size // 2 - 2)

test_contracting_block()
test_contracting_block(10, 9, 8)
print("Success!")
```

    Success!


## Expanding Path
Next, you will implement the expanding blocks for the expanding path. This is the decoding section of U-Net which has several upsampling steps as part of it. In order to do this, you'll also need to write a crop function. This is so you can crop the image from the *contracting path* and concatenate it to the current image on the expanding path—this is to form a skip connection. Again, the details are from the paper (Renneberger, 2015):

>Every step in the expanding path consists of an upsampling of the feature map followed by a 2 x 2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3 x 3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution.

<!-- so that the expanding block can resize the input from the contracting block can have the same size as the input from the previous layer -->

*Fun fact: later models based on this architecture often use padding in the convolutions to prevent the size of the image from changing outside of the upsampling / downsampling steps!*

<details>
<summary>
<font size="3" color="green">
<b>Optional hint for <code><font size="4">ExpandingBlock</font></code></b>
</font>
</summary>

1.    The concatenation means the number of channels goes back to being input_channels, so you need to halve it again for the next convolution.
</details>


```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: crop
def crop(image, new_shape):
    '''
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels.
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    '''
    # There are many ways to implement this crop function, but it's what allows
    # the skip connection to function as intended with two differently sized images!
    #### START CODE HERE ####
    _, _, h, w = image.shape
    _, _, h_new, w_new = new_shape
    ch, cw = h//2, w//2
    ch_new, cw_new = h_new//2, w_new//2
    x1 = int(cw - cw_new)
    y1 = int(ch - ch_new)
    x2 = int(x1 + w_new)
    y2 = int(y1 + h_new)
    return image[:, :, y1:y2, x1:x2]
    #### END CODE HERE ####
```


```python
#UNIT TEST
def test_expanding_block_crop(test_samples=100, test_channels=10, test_size=100):
    # Make sure that the crop function is the right shape
    skip_con_x = torch.randn(test_samples, test_channels, test_size + 6, test_size + 6)
    x = torch.randn(test_samples, test_channels, test_size, test_size)
    cropped = crop(skip_con_x, x.shape)
    assert tuple(cropped.shape) == (test_samples, test_channels, test_size, test_size)

    # Make sure that the crop function takes the right area
    test_meshgrid = torch.meshgrid([torch.arange(0, test_size), torch.arange(0, test_size)])
    test_meshgrid = test_meshgrid[0] + test_meshgrid[1]
    test_meshgrid = test_meshgrid[None, None, :, :].float()
    cropped = crop(test_meshgrid, torch.Size([1, 1, test_size // 2, test_size // 2]))
    assert cropped.max() == (test_size - 1) * 2 - test_size // 2
    assert cropped.min() == test_size // 2
    assert cropped.mean() == test_size - 1

    test_meshgrid = torch.meshgrid([torch.arange(0, test_size), torch.arange(0, test_size)])
    test_meshgrid = test_meshgrid[0] + test_meshgrid[1]
    crop_size = 5
    test_meshgrid = test_meshgrid[None, None, :, :].float()
    cropped = crop(test_meshgrid, torch.Size([1, 1, crop_size, crop_size]))
    assert cropped.max() <= (test_size + crop_size - 1) and cropped.max() >= test_size - 1
    assert cropped.min() >= (test_size - crop_size - 1) and cropped.min() <= test_size - 1
    assert abs(cropped.mean() - test_size) <= 2

test_expanding_block_crop()
print("Success!")
```

    Success!



```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: ExpandingBlock
class ExpandingBlock(nn.Module):
    '''
    ExpandingBlock Class
    Performs an upsampling, a convolution, a concatenation of its two inputs,
    followed by two more convolutions.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels):
        super(ExpandingBlock, self).__init__()
        # "Every step in the expanding path consists of an upsampling of the feature map"
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # "followed by a 2x2 convolution that halves the number of feature channels"
        # "a concatenation with the correspondingly cropped feature map from the contracting path"
        # "and two 3x3 convolutions"
        #### START CODE HERE ####
        self.conv1 = nn.Conv2d(input_channels, input_channels//2, kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels, input_channels//2, kernel_size=3)
        self.conv3 = nn.Conv2d(input_channels//2, input_channels//2, kernel_size=3)
        #### END CODE HERE ####
        self.activation = nn.ReLU() # "each followed by a ReLU"
 
    def forward(self, x, skip_con_x):
        '''
        Function for completing a forward pass of ExpandingBlock: 
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        return x
    
    # Required for grading
    def get_self(self):
        return self
```


```python
#UNIT TEST
def test_expanding_block(test_samples=100, test_channels=10, test_size=50):
    test_block = ExpandingBlock(test_channels)
    skip_con_x = torch.randn(test_samples, test_channels // 2, test_size * 2 + 6, test_size * 2 + 6)
    x = torch.randn(test_samples, test_channels, test_size, test_size)
    x = test_block.upsample(x)
    x = test_block.conv1(x)
    # Make sure that the first convolution produces the right shape
    assert tuple(x.shape) == (test_samples, test_channels // 2,  test_size * 2 - 1, test_size * 2 - 1)
    orginal_x = crop(skip_con_x, x.shape)
    x = torch.cat([x, orginal_x], axis=1)
    x = test_block.conv2(x)
    # Make sure that the second convolution produces the right shape
    assert tuple(x.shape) == (test_samples, test_channels // 2,  test_size * 2 - 3, test_size * 2 - 3)
    x = test_block.conv3(x)
    # Make sure that the final convolution produces the right shape
    assert tuple(x.shape) == (test_samples, test_channels // 2,  test_size * 2 - 5, test_size * 2 - 5)
    x = test_block.activation(x)

test_expanding_block()
print("Success!")
```

    Success!


## Final Layer
Now you will write the final feature mapping block, which takes in a tensor with arbitrarily many tensors and produces a tensor with the same number of pixels but with the correct number of output channels. From the paper (Renneberger, 2015):

>At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. In total the network has 23 convolutional layers.



```python
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: FeatureMapBlock
class FeatureMapBlock(nn.Module):
    '''
    FeatureMapBlock Class
    The final layer of a UNet - 
    maps each pixel to a pixel with the correct number of output dimensions
    using a 1x1 convolution.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        # "Every step in the expanding path consists of an upsampling of the feature map"
        #### START CODE HERE ####
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        #### END CODE HERE ####

    def forward(self, x):
        '''
        Function for completing a forward pass of FeatureMapBlock: 
        Given an image tensor, returns it mapped to the desired number of channels.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv(x)
        return x
```


```python
# UNIT TEST
assert tuple(FeatureMapBlock(10, 60)(torch.randn(1, 10, 10, 10)).shape) == (1, 60, 10, 10)
print("Success!")
```

    Success!


## U-Net

Now you can put it all together! Here, you'll write a `UNet` class which will combine a series of the three kinds of blocks you've implemented.


```python
# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CLASS: UNet
class UNet(nn.Module):
    '''
    UNet Class
    A series of 4 contracting blocks followed by 4 expanding blocks to 
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(UNet, self).__init__()
        # "Every step in the expanding path consists of an upsampling of the feature map"
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels)
        self.contract2 = ContractingBlock(hidden_channels * 2)
        self.contract3 = ContractingBlock(hidden_channels * 4)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.expand1 = ExpandingBlock(hidden_channels * 16)
        self.expand2 = ExpandingBlock(hidden_channels * 8)
        self.expand3 = ExpandingBlock(hidden_channels * 4)
        self.expand4 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)

    def forward(self, x):
        '''
        Function for completing a forward pass of UNet: 
        Given an image tensor, passes it through U-Net and returns the output.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
        # Keep in mind that the expand function takes two inputs, 
        # both with the same number of channels. 
        #### START CODE HERE ####
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.expand1(x4, x3)
        x6 = self.expand2(x5, x2)
        x7 = self.expand3(x6, x1)
        x8 = self.expand4(x7, x0)
        xn = self.downfeature(x8)
        #### END CODE HERE ####
        return xn
```


```python
#UNIT TEST
test_unet = UNet(1, 3)
assert tuple(test_unet(torch.randn(1, 1, 256, 256)).shape) == (1, 3, 117, 117)
print("Success!")
```

    Success!


## Training

Finally, you will put this into action!
Remember that these are your parameters:
  *   criterion: the loss function
  *   n_epochs: the number of times you iterate through the entire dataset when training
  *   input_dim: the number of channels of the input image
  *   label_dim: the number of channels of the output image
  *   display_step: how often to display/visualize the images
  *   batch_size: the number of images per forward/backward pass
  *   lr: the learning rate
  *   initial_shape: the size of the input image (in pixels)
  *   target_shape: the size of the output image (in pixels)
  *   device: the device type

This should take only a few minutes to train!


```python
import torch.nn.functional as F
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
input_dim = 1
label_dim = 1
display_step = 20
batch_size = 4
lr = 0.0002
initial_shape = 512
target_shape = 373
device = 'cuda'
```


```python
from skimage import io
import numpy as np
volumes = torch.Tensor(io.imread('train-volume.tif'))[:, None, :, :] / 255
labels = torch.Tensor(io.imread('train-labels.tif', plugin="tifffile"))[:, None, :, :] / 255
labels = crop(labels, torch.Size([len(labels), 1, target_shape, target_shape]))
dataset = torch.utils.data.TensorDataset(volumes, labels)
```


```python
def train():
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    unet = UNet(input_dim, label_dim).to(device)
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr)
    cur_step = 0

    for epoch in range(n_epochs):
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            # Flatten the image
            real = real.to(device)
            labels = labels.to(device)

            ### Update U-Net ###
            unet_opt.zero_grad()
            pred = unet(real)
            unet_loss = criterion(pred, labels)
            unet_loss.backward()
            unet_opt.step()

            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: U-Net loss: {unet_loss.item()}")
                show_tensor_images(
                    crop(real, torch.Size([len(real), 1, target_shape, target_shape])), 
                    size=(input_dim, target_shape, target_shape)
                )
                show_tensor_images(labels, size=(label_dim, target_shape, target_shape))
                show_tensor_images(torch.sigmoid(pred), size=(label_dim, target_shape, target_shape))
            cur_step += 1

train()
```


      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 0: Step 0: U-Net loss: 0.7121937870979309



    
![png](output_22_2.png)
    



    
![png](output_22_3.png)
    



    
![png](output_22_4.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 2: Step 20: U-Net loss: 0.5241668820381165



    
![png](output_22_8.png)
    



    
![png](output_22_9.png)
    



    
![png](output_22_10.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 5: Step 40: U-Net loss: 0.47498342394828796



    
![png](output_22_15.png)
    



    
![png](output_22_16.png)
    



    
![png](output_22_17.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 7: Step 60: U-Net loss: 0.3683202266693115



    
![png](output_22_21.png)
    



    
![png](output_22_22.png)
    



    
![png](output_22_23.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 10: Step 80: U-Net loss: 0.3633531630039215



    
![png](output_22_28.png)
    



    
![png](output_22_29.png)
    



    
![png](output_22_30.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 12: Step 100: U-Net loss: 0.39409294724464417



    
![png](output_22_34.png)
    



    
![png](output_22_35.png)
    



    
![png](output_22_36.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 15: Step 120: U-Net loss: 0.3476548194885254



    
![png](output_22_41.png)
    



    
![png](output_22_42.png)
    



    
![png](output_22_43.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 17: Step 140: U-Net loss: 0.35609349608421326



    
![png](output_22_47.png)
    



    
![png](output_22_48.png)
    



    
![png](output_22_49.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 20: Step 160: U-Net loss: 0.3048531413078308



    
![png](output_22_54.png)
    



    
![png](output_22_55.png)
    



    
![png](output_22_56.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 22: Step 180: U-Net loss: 0.3555775582790375



    
![png](output_22_60.png)
    



    
![png](output_22_61.png)
    



    
![png](output_22_62.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 25: Step 200: U-Net loss: 0.314405232667923



    
![png](output_22_67.png)
    



    
![png](output_22_68.png)
    



    
![png](output_22_69.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 27: Step 220: U-Net loss: 0.26302674412727356



    
![png](output_22_73.png)
    



    
![png](output_22_74.png)
    



    
![png](output_22_75.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 30: Step 240: U-Net loss: 0.29167670011520386



    
![png](output_22_80.png)
    



    
![png](output_22_81.png)
    



    
![png](output_22_82.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 32: Step 260: U-Net loss: 0.27725639939308167



    
![png](output_22_86.png)
    



    
![png](output_22_87.png)
    



    
![png](output_22_88.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 35: Step 280: U-Net loss: 0.31775686144828796



    
![png](output_22_93.png)
    



    
![png](output_22_94.png)
    



    
![png](output_22_95.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 37: Step 300: U-Net loss: 0.2989847958087921



    
![png](output_22_99.png)
    



    
![png](output_22_100.png)
    



    
![png](output_22_101.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 40: Step 320: U-Net loss: 0.25126680731773376



    
![png](output_22_106.png)
    



    
![png](output_22_107.png)
    



    
![png](output_22_108.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 42: Step 340: U-Net loss: 0.2428232729434967



    
![png](output_22_112.png)
    



    
![png](output_22_113.png)
    



    
![png](output_22_114.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 45: Step 360: U-Net loss: 0.2465704083442688



    
![png](output_22_119.png)
    



    
![png](output_22_120.png)
    



    
![png](output_22_121.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 47: Step 380: U-Net loss: 0.22647109627723694



    
![png](output_22_125.png)
    



    
![png](output_22_126.png)
    



    
![png](output_22_127.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 50: Step 400: U-Net loss: 0.219889298081398



    
![png](output_22_132.png)
    



    
![png](output_22_133.png)
    



    
![png](output_22_134.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 52: Step 420: U-Net loss: 0.21477098762989044



    
![png](output_22_138.png)
    



    
![png](output_22_139.png)
    



    
![png](output_22_140.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 55: Step 440: U-Net loss: 0.22318126261234283



    
![png](output_22_145.png)
    



    
![png](output_22_146.png)
    



    
![png](output_22_147.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 57: Step 460: U-Net loss: 0.19058632850646973



    
![png](output_22_151.png)
    



    
![png](output_22_152.png)
    



    
![png](output_22_153.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 60: Step 480: U-Net loss: 0.233653724193573



    
![png](output_22_158.png)
    



    
![png](output_22_159.png)
    



    
![png](output_22_160.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 62: Step 500: U-Net loss: 0.21280553936958313



    
![png](output_22_164.png)
    



    
![png](output_22_165.png)
    



    
![png](output_22_166.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 65: Step 520: U-Net loss: 0.19247037172317505



    
![png](output_22_171.png)
    



    
![png](output_22_172.png)
    



    
![png](output_22_173.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 67: Step 540: U-Net loss: 0.19753386080265045



    
![png](output_22_177.png)
    



    
![png](output_22_178.png)
    



    
![png](output_22_179.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 70: Step 560: U-Net loss: 0.15693819522857666



    
![png](output_22_184.png)
    



    
![png](output_22_185.png)
    



    
![png](output_22_186.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 72: Step 580: U-Net loss: 0.18084800243377686



    
![png](output_22_190.png)
    



    
![png](output_22_191.png)
    



    
![png](output_22_192.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 75: Step 600: U-Net loss: 0.16204871237277985



    
![png](output_22_197.png)
    



    
![png](output_22_198.png)
    



    
![png](output_22_199.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 77: Step 620: U-Net loss: 0.15913277864456177



    
![png](output_22_203.png)
    



    
![png](output_22_204.png)
    



    
![png](output_22_205.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 80: Step 640: U-Net loss: 0.15234796702861786



    
![png](output_22_210.png)
    



    
![png](output_22_211.png)
    



    
![png](output_22_212.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 82: Step 660: U-Net loss: 0.14264142513275146



    
![png](output_22_216.png)
    



    
![png](output_22_217.png)
    



    
![png](output_22_218.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 85: Step 680: U-Net loss: 0.12344925850629807



    
![png](output_22_223.png)
    



    
![png](output_22_224.png)
    



    
![png](output_22_225.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 87: Step 700: U-Net loss: 0.13199740648269653



    
![png](output_22_229.png)
    



    
![png](output_22_230.png)
    



    
![png](output_22_231.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 90: Step 720: U-Net loss: 0.12643560767173767



    
![png](output_22_236.png)
    



    
![png](output_22_237.png)
    



    
![png](output_22_238.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 92: Step 740: U-Net loss: 0.13207457959651947



    
![png](output_22_242.png)
    



    
![png](output_22_243.png)
    



    
![png](output_22_244.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 95: Step 760: U-Net loss: 0.12737613916397095



    
![png](output_22_249.png)
    



    
![png](output_22_250.png)
    



    
![png](output_22_251.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 97: Step 780: U-Net loss: 0.10927250981330872



    
![png](output_22_255.png)
    



    
![png](output_22_256.png)
    



    
![png](output_22_257.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 100: Step 800: U-Net loss: 0.11686728894710541



    
![png](output_22_262.png)
    



    
![png](output_22_263.png)
    



    
![png](output_22_264.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 102: Step 820: U-Net loss: 0.10815241187810898



    
![png](output_22_268.png)
    



    
![png](output_22_269.png)
    



    
![png](output_22_270.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 105: Step 840: U-Net loss: 0.10122904926538467



    
![png](output_22_275.png)
    



    
![png](output_22_276.png)
    



    
![png](output_22_277.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 107: Step 860: U-Net loss: 0.10206568241119385



    
![png](output_22_281.png)
    



    
![png](output_22_282.png)
    



    
![png](output_22_283.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 110: Step 880: U-Net loss: 0.10022618621587753



    
![png](output_22_288.png)
    



    
![png](output_22_289.png)
    



    
![png](output_22_290.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 112: Step 900: U-Net loss: 0.0950017124414444



    
![png](output_22_294.png)
    



    
![png](output_22_295.png)
    



    
![png](output_22_296.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 115: Step 920: U-Net loss: 0.09242583811283112



    
![png](output_22_301.png)
    



    
![png](output_22_302.png)
    



    
![png](output_22_303.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 117: Step 940: U-Net loss: 0.09846685081720352



    
![png](output_22_307.png)
    



    
![png](output_22_308.png)
    



    
![png](output_22_309.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 120: Step 960: U-Net loss: 0.08531367033720016



    
![png](output_22_314.png)
    



    
![png](output_22_315.png)
    



    
![png](output_22_316.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 122: Step 980: U-Net loss: 0.07917763292789459



    
![png](output_22_320.png)
    



    
![png](output_22_321.png)
    



    
![png](output_22_322.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 125: Step 1000: U-Net loss: 0.08374503999948502



    
![png](output_22_327.png)
    



    
![png](output_22_328.png)
    



    
![png](output_22_329.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 127: Step 1020: U-Net loss: 0.07394184917211533



    
![png](output_22_333.png)
    



    
![png](output_22_334.png)
    



    
![png](output_22_335.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 130: Step 1040: U-Net loss: 0.07997559010982513



    
![png](output_22_340.png)
    



    
![png](output_22_341.png)
    



    
![png](output_22_342.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 132: Step 1060: U-Net loss: 0.07902295887470245



    
![png](output_22_346.png)
    



    
![png](output_22_347.png)
    



    
![png](output_22_348.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 135: Step 1080: U-Net loss: 0.07417745143175125



    
![png](output_22_353.png)
    



    
![png](output_22_354.png)
    



    
![png](output_22_355.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 137: Step 1100: U-Net loss: 0.07410058379173279



    
![png](output_22_359.png)
    



    
![png](output_22_360.png)
    



    
![png](output_22_361.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 140: Step 1120: U-Net loss: 0.07316091656684875



    
![png](output_22_366.png)
    



    
![png](output_22_367.png)
    



    
![png](output_22_368.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 142: Step 1140: U-Net loss: 0.05912798270583153



    
![png](output_22_372.png)
    



    
![png](output_22_373.png)
    



    
![png](output_22_374.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 145: Step 1160: U-Net loss: 0.057397566735744476



    
![png](output_22_379.png)
    



    
![png](output_22_380.png)
    



    
![png](output_22_381.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 147: Step 1180: U-Net loss: 0.06582657992839813



    
![png](output_22_385.png)
    



    
![png](output_22_386.png)
    



    
![png](output_22_387.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 150: Step 1200: U-Net loss: 0.06557486951351166



    
![png](output_22_392.png)
    



    
![png](output_22_393.png)
    



    
![png](output_22_394.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 152: Step 1220: U-Net loss: 0.062375567853450775



    
![png](output_22_398.png)
    



    
![png](output_22_399.png)
    



    
![png](output_22_400.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 155: Step 1240: U-Net loss: 0.06731535494327545



    
![png](output_22_405.png)
    



    
![png](output_22_406.png)
    



    
![png](output_22_407.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 157: Step 1260: U-Net loss: 0.05443323403596878



    
![png](output_22_411.png)
    



    
![png](output_22_412.png)
    



    
![png](output_22_413.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 160: Step 1280: U-Net loss: 0.05661916360259056



    
![png](output_22_418.png)
    



    
![png](output_22_419.png)
    



    
![png](output_22_420.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 162: Step 1300: U-Net loss: 0.05655153468251228



    
![png](output_22_424.png)
    



    
![png](output_22_425.png)
    



    
![png](output_22_426.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 165: Step 1320: U-Net loss: 0.0590219721198082



    
![png](output_22_431.png)
    



    
![png](output_22_432.png)
    



    
![png](output_22_433.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 167: Step 1340: U-Net loss: 0.048821572214365005



    
![png](output_22_437.png)
    



    
![png](output_22_438.png)
    



    
![png](output_22_439.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 170: Step 1360: U-Net loss: 0.05500505119562149



    
![png](output_22_444.png)
    



    
![png](output_22_445.png)
    



    
![png](output_22_446.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 172: Step 1380: U-Net loss: 0.04902040213346481



    
![png](output_22_450.png)
    



    
![png](output_22_451.png)
    



    
![png](output_22_452.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 175: Step 1400: U-Net loss: 0.05654953047633171



    
![png](output_22_457.png)
    



    
![png](output_22_458.png)
    



    
![png](output_22_459.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 177: Step 1420: U-Net loss: 0.05042034387588501



    
![png](output_22_463.png)
    



    
![png](output_22_464.png)
    



    
![png](output_22_465.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 180: Step 1440: U-Net loss: 0.04726465418934822



    
![png](output_22_470.png)
    



    
![png](output_22_471.png)
    



    
![png](output_22_472.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 182: Step 1460: U-Net loss: 0.04395608976483345



    
![png](output_22_476.png)
    



    
![png](output_22_477.png)
    



    
![png](output_22_478.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 185: Step 1480: U-Net loss: 0.05461639165878296



    
![png](output_22_483.png)
    



    
![png](output_22_484.png)
    



    
![png](output_22_485.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 187: Step 1500: U-Net loss: 0.05021529272198677



    
![png](output_22_489.png)
    



    
![png](output_22_490.png)
    



    
![png](output_22_491.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 190: Step 1520: U-Net loss: 0.046015944331884384



    
![png](output_22_496.png)
    



    
![png](output_22_497.png)
    



    
![png](output_22_498.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]


    Epoch 192: Step 1540: U-Net loss: 0.044696494936943054



    
![png](output_22_502.png)
    



    
![png](output_22_503.png)
    



    
![png](output_22_504.png)
    



      0%|          | 0/8 [00:00<?, ?it/s]



      0%|          | 0/8 [00:00<?, ?it/s]



```python

```
