# Spectrally Normalized Generative Adversarial Networks (SN-GAN)

*Please note that this is an optional notebook, meant to introduce more advanced concepts if you're up for a challenge, so don't worry if you don't completely follow!*

**Goals**

In this notebook, you'll learn about and implement **spectral normalization**, a weight normalization technique to stabilize the training of the discriminator, as proposed in [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957) (Miyato et al. 2018).

**Background**

As its name suggests, SN-GAN normalizes the weight matrices in the discriminator by their corresponding [spectral norm](https://calculus.subwiki.org/wiki/Spectral_norm#:~:text=The%20spectral%20norm%20of%20a,where%20denotes%20the%20Euclidean%20norm.), which helps control the Lipschitz constant of the discriminator. As you have learned with WGAN, [Lipschitz continuity](https://en.wikipedia.org/wiki/Lipschitz_continuity) is important in ensuring the boundedness of the optimal discriminator. In the WGAN case, this makes it so that the underlying W-loss function for the discriminator (or more precisely, the critic) is valid.

As a result, spectral normalization helps improve stability and avoid vanishing gradient problems, such as mode collapse.

## Spectral Norm

Notationally, the spectral norm of a matrix $W$ is typically represented as $\sigma(W)$. For neural network purposes, this $W$ matrix represents a weight matrix in one of the network's layers. The spectral norm of a matrix is the matrix's largest singular value, which can be obtained via singular value decomposition (SVD).

**A Quick Refresher on SVD**

SVD is a generalization of [eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix) and is used to factorize a matrix as $W = U\Sigma V^\top$, where $U, V$ are orthogonal matrices and $\Sigma$ is a matrix of singular values on its diagonal. Note that $\Sigma$ doesn't have to be square.

\begin{align*}
    \Sigma = \begin{bmatrix}\sigma_1 & & \\ & \sigma_2 \\ & & \ddots \\ & & & \sigma_n\end{bmatrix}
\end{align*}

where $\sigma_1$ and $\sigma_n$ are the largest and smallest singular values, respectively. Intuitively, larger values correspond to larger amounts of stretching a matrix can apply to another vector. Following this notation, $\sigma(W) = \sigma_1$.

**Applying SVD to Spectral Normalization**

To spectrally normalize the weight matrix, you divide every value in the matrix by its spectral norm. As a result, a spectrally normalized matrix $\overline{W}_{SN}$ can be expressed as

\begin{align*}
  \overline{W}_{SN} = \dfrac{W}{\sigma(W)},
\end{align*}

In practice, computing the SVD of $W$ is expensive, so the authors of the SN-GAN paper do something very neat. They instead approximate the left and right singular vectors, $\tilde{u}$ and $\tilde{v}$ respectively, through power iteration such that $\sigma(W) \approx \tilde{u}^\top W\tilde{v}$.

Starting from randomly initialization, $\tilde{u}$ and $\tilde{v}$ are updated according to

\begin{align*}
  \tilde{u} &:= \dfrac{W^\top\tilde{u}}{||W^\top\tilde{u}||_2} \\
  \tilde{v} &:= \dfrac{W\tilde{v}}{||W\tilde{v}||_2}
\end{align*}

In practice, one round of iteration is sufficient to "achieve satisfactory performance" as per the authors.

Don't worry if you don't completely follow this! The algorithm is conveniently implemented as `torch.nn.utils.spectral_norm` in PyTorch, so as long as you get the general gist of how it might be useful and when to use it, then you're all set.


## A Bit of History on Spectral Normalization

This isn't the first time that spectral norm has been proposed in the context of deep learning models. There's a paper called [Spectral Norm Regularization for Improving the Generalizability of Deep Learning](https://arxiv.org/abs/1705.10941) (Yoshida et al. 2017) that proposes **spectral norm regularization**, which they showed to improve the generalizability of models by adding extra loss terms onto the loss function (just as L2 regularization and gradient penalty do!). These extra loss terms specifically penalize the spectral norm of the weights. You can think of this as *data-independent* regularization because the gradient with respect to $W$ isn't a function of the minibatch.

**Spectral normalization**, on the other hand, sets the spectral norm of the weight matrices to 1 -- it's a much harder constraint than adding a loss term, which is a form of "soft" regularization. As the authors show in the paper, you can think of spectral normalization as *data-dependent* regularization, since the gradient with respect to $W$ is dependent on the mini-batch statistics (shown in Section 2.1 of the [main paper](https://arxiv.org/pdf/1802.05957.pdf)). Spectral normalization essentially prevents the transformation of each layer
from becoming to sensitive in one direction and mitigates exploding gradients.

## DCGAN with Spectral Normalization

In rest of this notebook, you will walk through how to apply spectral normalization to DCGAN as an example, using your earlier DCGAN implementation. You can always add spectral normalization to your other models too.

Here, you start with the same setup and helper function, as you've seen before. 


```python
# Some setup
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for our testing purposes, please do not change!

'''
Function for visualizing images: Given a tensor of images, number of images, and
size per image, plots and prints the images in an uniform grid.
'''
def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
```

### DCGAN Generator

Since spectral normalization is only applied to the matrices in the discriminator, the generator implementation is the same as the original.


```python
class Generator(nn.Module):
    '''
    Generator Class
    Values:
    z_dim: the dimension of the noise vector, a scalar
    im_chan: the number of channels of the output image, a scalar
            MNIST is black-and-white, so that's our default
    hidden_dim: the inner dimension, a scalar
    '''
    
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of the DCGAN, 
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation
        Parameters:
        input_channels: how many channels the input feature representation has
        output_channels: how many channels the output feature representation should have
        kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride: the stride of the convolution
        final_layer: whether we're on the final layer (affects activation and batchnorm)
        '''
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else: # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def unsqueeze_noise(self, noise):
        '''
        Function for completing a forward pass of the Generator: Given a noise vector, 
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
        noise: a noise tensor with dimensions (batch_size, z_dim)
        '''
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        '''
        Function for completing a forward pass of the Generator: Given a noise vector, 
        returns a generated image.
        Parameters:
        noise: a noise tensor with dimensions (batch_size, z_dim)
        '''
        x = self.unsqueeze_noise(noise)
        return self.gen(x)

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating a noise vector: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
    n_samples: the number of samples in the batch, a scalar
    z_dim: the dimension of the noise vector, a scalar
    device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)
```

### DCGAN Discriminator

For the discriminator, you can wrap each `nn.Conv2d` with `nn.utils.spectral_norm`. In the backend, this introduces parameters for $\tilde{u}$ and $\tilde{v}$ in addition to $W$ so that the $W_{SN}$ can be computed as $\tilde{u}^\top W\tilde{v}$ in runtime.

Pytorch also provides a `nn.utils.remove_spectral_norm` function, which collapses the 3 separate parameters into a single explicit $\overline{W}_{SN} := \tilde{u}^\top W\tilde{v}$. You should only apply this to your convolutional layers during inference to improve runtime speed.

It is important note that spectral norm does not eliminate the need for batch norm. Spectral norm affects the weights of each layer, while batch norm affects the activations of each layer. You can see both in a discriminator architecture, but you can also see just one of them. Hope this is something you have fun experimenting with!


```python
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
    im_chan: the number of channels of the output image, a scalar
            MNIST is black-and-white (1 channel), so that's our default.
    hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of the DCGAN, 
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation
        Parameters:
        input_channels: how many channels the input feature representation has
        output_channels: how many channels the output feature representation should have
        kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
        stride: the stride of the convolution
        final_layer: whether we're on the final layer (affects activation and batchnorm)
        '''
        
        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(input_channels, output_channels, kernel_size, stride)),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else: # Final Layer
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(input_channels, output_channels, kernel_size, stride)),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the Discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
        image: a flattened image tensor with dimension (im_dim)
        '''
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
```

### Training SN-DCGAN

You can now put everything together and train a spectrally normalized DCGAN! Here are all your parameters for initialization and optimization. 


```python
criterion = nn.BCEWithLogitsLoss()
n_epochs = 50
z_dim = 64
display_step = 500
batch_size = 128
# A learning rate of 0.0002 works well on DCGAN
lr = 0.0002

# These parameters control the optimizer's momentum, which you can read more about here:
# https://distill.pub/2017/momentum/ but you don’t need to worry about it for this course
beta_1 = 0.5 
beta_2 = 0.999
device = 'cuda'

# We tranform our image values to be between -1 and 1 (the range of the tanh activation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST(".", download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True)
```

Now, initialize the generator, the discriminator, and the optimizers.


```python
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

# We initialize the weights to the normal distribution
# with mean 0 and standard deviation 0.02
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)
```

Finally, train the whole thing! And babysit those outputs :)


```python
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
for epoch in range(n_epochs):
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)

        ## Update Discriminator ##
        disc_opt.zero_grad()
        fake_noise = get_noise(cur_batch_size, z_dim, device=device)
        fake = gen(fake_noise)
        disc_fake_pred = disc(fake.detach())
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = disc(real)
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step
        # Update gradients
        disc_loss.backward(retain_graph=True)
        # Update optimizer
        disc_opt.step()

        ## Update Generator ##
        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
        fake_2 = gen(fake_noise_2)
        disc_fake_pred = disc(fake_2)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        ## Visualization code ##
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1
```


      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 500: Generator loss: 0.6941643460989002, discriminator loss: 0.6967705994844432



    
![png](output_14_3.png)
    



    
![png](output_14_4.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 1000: Generator loss: 0.6932537760734562, discriminator loss: 0.6931680802106858



    
![png](output_14_7.png)
    



    
![png](output_14_8.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 1500: Generator loss: 0.6931905971765515, discriminator loss: 0.6932025200128562



    
![png](output_14_11.png)
    



    
![png](output_14_12.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 2000: Generator loss: 0.6932471932172775, discriminator loss: 0.6931883269548417



    
![png](output_14_15.png)
    



    
![png](output_14_16.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 2500: Generator loss: 0.6933203892707823, discriminator loss: 0.6931734483242038



    
![png](output_14_19.png)
    



    
![png](output_14_20.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 3000: Generator loss: 0.6932222476005553, discriminator loss: 0.6931811637878413



    
![png](output_14_23.png)
    



    
![png](output_14_24.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 3500: Generator loss: 0.6931579736471166, discriminator loss: 0.6931520874500271



    
![png](output_14_27.png)
    



    
![png](output_14_28.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 4000: Generator loss: 0.693174110770226, discriminator loss: 0.6931516668796543



    
![png](output_14_31.png)
    



    
![png](output_14_32.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 4500: Generator loss: 0.6934900225400918, discriminator loss: 0.6932667939662933



    
![png](output_14_35.png)
    



    
![png](output_14_36.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 5000: Generator loss: 0.6931762549877162, discriminator loss: 0.693176806688309



    
![png](output_14_39.png)
    



    
![png](output_14_40.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 5500: Generator loss: 0.6932879742383959, discriminator loss: 0.6931519120931628



    
![png](output_14_43.png)
    



    
![png](output_14_44.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 6000: Generator loss: 0.6933463417291642, discriminator loss: 0.693222567081452



    
![png](output_14_47.png)
    



    
![png](output_14_48.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 6500: Generator loss: 0.6933241475820544, discriminator loss: 0.6931776897907255



    
![png](output_14_51.png)
    



    
![png](output_14_52.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 7000: Generator loss: 0.6931514507532118, discriminator loss: 0.6931546316146855



    
![png](output_14_55.png)
    



    
![png](output_14_56.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 7500: Generator loss: 0.6931601431369779, discriminator loss: 0.69315239417553



    
![png](output_14_59.png)
    



    
![png](output_14_60.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 8000: Generator loss: 0.6931571400165553, discriminator loss: 0.6931529202461242



    
![png](output_14_64.png)
    



    
![png](output_14_65.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 8500: Generator loss: 0.6931808847188956, discriminator loss: 0.6931500614881516



    
![png](output_14_68.png)
    



    
![png](output_14_69.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 9000: Generator loss: 0.6933453525304788, discriminator loss: 0.6931912213563914



    
![png](output_14_72.png)
    



    
![png](output_14_73.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 9500: Generator loss: 0.6931842290163039, discriminator loss: 0.6931706986427304



    
![png](output_14_76.png)
    



    
![png](output_14_77.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 10000: Generator loss: 0.6937996336221691, discriminator loss: 0.69337668633461



    
![png](output_14_80.png)
    



    
![png](output_14_81.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 10500: Generator loss: 0.6932606348991396, discriminator loss: 0.6931961631774897



    
![png](output_14_84.png)
    



    
![png](output_14_85.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 11000: Generator loss: 0.6934080917835237, discriminator loss: 0.6932327284812934



    
![png](output_14_88.png)
    



    
![png](output_14_89.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 11500: Generator loss: 0.6931626038551332, discriminator loss: 0.693149412035942



    
![png](output_14_92.png)
    



    
![png](output_14_93.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 12000: Generator loss: 0.6931908987760532, discriminator loss: 0.6931689337491997



    
![png](output_14_96.png)
    



    
![png](output_14_97.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 12500: Generator loss: 0.6931863358020782, discriminator loss: 0.6931602952480317



    
![png](output_14_100.png)
    



    
![png](output_14_101.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 13000: Generator loss: 0.6931699364185333, discriminator loss: 0.693158650159836



    
![png](output_14_104.png)
    



    
![png](output_14_105.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 13500: Generator loss: 0.6931706752777099, discriminator loss: 0.6931542218923564



    
![png](output_14_108.png)
    



    
![png](output_14_109.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 14000: Generator loss: 0.6932363721132281, discriminator loss: 0.6931678054332736



    
![png](output_14_112.png)
    



    
![png](output_14_113.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 14500: Generator loss: 0.693671117305756, discriminator loss: 0.6932303344011299



    
![png](output_14_116.png)
    



    
![png](output_14_117.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 15000: Generator loss: 0.6935198810100559, discriminator loss: 0.6931732662916185



    
![png](output_14_120.png)
    



    
![png](output_14_121.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 15500: Generator loss: 0.6939301556348804, discriminator loss: 0.6929111987352371



    
![png](output_14_125.png)
    



    
![png](output_14_126.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 16000: Generator loss: 0.7133436391353601, discriminator loss: 0.679645606875419



    
![png](output_14_129.png)
    



    
![png](output_14_130.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 16500: Generator loss: 0.7275565942525863, discriminator loss: 0.6733638910055155



    
![png](output_14_133.png)
    



    
![png](output_14_134.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 17000: Generator loss: 0.7086363095045088, discriminator loss: 0.6828448543548572



    
![png](output_14_137.png)
    



    
![png](output_14_138.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 17500: Generator loss: 0.7114701668024059, discriminator loss: 0.6848915455341336



    
![png](output_14_141.png)
    



    
![png](output_14_142.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 18000: Generator loss: 0.7089320507049559, discriminator loss: 0.6858939023017886



    
![png](output_14_145.png)
    



    
![png](output_14_146.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 18500: Generator loss: 0.7069256994724272, discriminator loss: 0.6872664549350734



    
![png](output_14_149.png)
    



    
![png](output_14_150.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 19000: Generator loss: 0.7026238349676126, discriminator loss: 0.68925052845478



    
![png](output_14_153.png)
    



    
![png](output_14_154.png)
    


    Step 20000: Generator loss: 0.6992605997323991, discriminator loss: 0.6915657229423521



    
![png](output_14_156.png)
    



    
![png](output_14_157.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 20500: Generator loss: 0.6969591895341875, discriminator loss: 0.6922729653120043



    
![png](output_14_160.png)
    



    
![png](output_14_161.png)
    



      0%|          | 0/469 [00:00<?, ?it/s]


    Step 21000: Generator loss: 0.6956607799530025, discriminator loss: 0.6936039599180227



    
![png](output_14_164.png)
    



```python

```
