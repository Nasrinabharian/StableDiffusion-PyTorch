''' 
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''
import nibabel as nib
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))

        self.data = []  # List to store data samples (image file paths and labels)
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                self.data.append((image_path, class_idx))

    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        assert os.path.exists(image_path), "images path {} does not exist".format(image_path)
        print(f"Found files: {image_path} in directory: {image_path}")
        if len(image_path) == 0:
            print(f"Warning: No NIfTI files found in {image_path}")   
        
        #with open(image_path, 'rb') as file:
        #    nifti_img = pickle.load(file)
        nifti_img = nib.load(image_path)
        nifti_img1 = nifti_img.get_fdata()
        
        #nifti_img1 = nifti_img
        #shape = (1, 16, 16, 16)

        # Generate the random 3D array
        #nifti_img1 = np.random.rand(*shape)

        #if nifti_img1.ndim != 3:
        #    raise ValueError(f"Expected 3D image, but got {nifti_img1.ndim}D array.")

        #min_val = nifti_img1.min()
        #max_val = nifti_img1.max()
        #normalized_image = nifti_img1
        
        #image_float = normalized_image.astype(np.float32)
        
        
        #nifti_img2 = nifti_img1

        #pkl nasrin added
    
        
        '''for key, value in nifti_img1.items():
            if hasattr(value, 'shape'):
                nifti_img1 = value'''
        #end of pkl

        nifti_img2 = torch.from_numpy(nifti_img1).float()  # Directly convert to tensor
        #nifti_img2 = torch.tensor(nifti_img1)
        #nifti_img2 = np.expand_dims(nifti_img2, axis=0)
        #nifti_img2 = nifti_img1
        min_val = nifti_img2.min()
        max_val = nifti_img2.max()
        #print(nifti_img2.shape)
        nifti_img2 = torch.unsqueeze(nifti_img2, 0)
        #nifti_img2 = nifti_img2.squeeze(0)
        
        ##nifti_img2 = nifti_img2.numpy()
        #print("nifti ", nifti_img1.shape)
        ims = nifti_img2
        return ims, label
        '''****
        image = Image.open(image_path).convert('L')  # Load and convert to RGB (adjust as needed)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label'''
    
class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm3d(out_channels), #BatchNorm3d because of the error 'If your batch size is 1, BatchNorm can be tricky because the batch statistics (mean and variance) cannot be computed properly for just a single example. If this is the case, you might want to switch to InstanceNorm3d or LayerNorm instead of BatchNorm3d when working with small batch sizes.
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm3d(out_channels), #BatchNorm3d
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool3d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose3d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=3): #hint_10to3
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose3d(2 * n_feat, 2 * n_feat, 8, 8), #4,4 instead of 1,1
            #nn.ConvTranspose3d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv3d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv3d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on
        
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask
        
        # embed context, time step
        # 3D change
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1, 1)
        '''cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)'''

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        #up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))

        
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,3).to(device) # context for us just cycles throught the mnist labels #hint_10to3
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))
        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1,1)
            t_is = t_is.repeat(2,1,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        
        return x_i, x_i_store, c_i

def plot_learning_curve(epochs, losses, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, label='Training Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'learning_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Saved learning curve at {plot_path}')

def train_mnist():

    # hardcoding these here
    n_epoch = 300
    batch_size = 1 #32 # do not change it because it affects output images : comment by Nasrin
    n_T = 500 # 500
    device = "cuda:0"
    n_classes = 3
    n_feat = 128 #256 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = True
    save_dir = '/Users/abharian/Documents/3D_conditional_MNIST/diffusion_generated_128fromoriginal/'
    ws_test = [0.5] # strength of generative guidance

    train_losses = []
    epoch_list = []

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    # optionally load a mode l
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    #tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    # Define data transformations (adjust as needed)
    #transforms.Resize((28, 28)),  
 

    transform = transforms.Compose([     
    #transforms.Resize((64, 64, 64)),# Resize the image to (28, 28)
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with one channel
    transforms.ToTensor(),            # Convert the image to a PyTorch tensor
    transforms.Normalize(0.5, 0.5, 0.5)   # Normalize the pixel values to range [-1, 1]
    ])
    custom_dataset_path = '/Users/abharian/Documents/3D_conditional_MNIST/128_input_of_diffusion_fromoriginal'

    # Create custom dataset instances for training and testing
    train_dataset = CustomDataset(os.path.join(custom_dataset_path, 'train'), transform=transform)
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)


        pbar = tqdm(dataloader)
        
        #tensor_squeezed = torch.squeeze(tensor, dim=0)
        loss_ema = None
        epoch_losses = []  # To store losses for this epoch

        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
            epoch_losses.append(loss.item())  # Store each batch loss

        # Calculate average loss for this epoch
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_epoch_loss)
        epoch_list.append(ep)
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4*n_classes
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store, c_i = ddpm.sample(n_sample, (1, 32, 32, 32), device, guide_w=w) #28to28
                #x_gen, x_gen_store = ddpm.sample(n_sample, (1, 16, 16, 16), device, guide_w=w) #28to28

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample/n_classes)):
                        try: 
                            z = (c == k)
                            result = z.nonzero()
                            idx = torch.squeeze(result)[j]
                        except:
                            idx = 0
                        x_real[k+(j*n_classes)] = x[idx]

                #x_all = torch.cat([x_gen, x_real]) #gen is top 12 and real is bottom 12 => totally 24 all
                x_all = torch.cat([x_gen])
                #print(type(x_all), x_all.shape, x_gen.shape, x_real.shape)
                #s = x_all*-1 + 1
                #print(s, s.shape)
                #grid = make_grid(x_all*-1 + 1, nrow=3) #hint_10to3
                print("shape of generated samples: ", x_all.shape)
                os.makedirs(save_dir, exist_ok=True)

                # Loop over the batch of images
                count = 0
                for idx in range(x_all.shape[0]):
                    # Get the 3D image (shape: 1, 16, 16, 16)
                    #Nasrin save pkl
                    latent_path = save_dir
                    if not os.path.exists(latent_path):
                        os.mkdir(latent_path)
                    
                    '''fname_latent_map = {}
                    part_count = 0
                    fname_latent_map[str(count)] = torch.unsqueeze((x_all[idx].cpu()), dim=0)
                        #fname_latent_map[im_dataset.images[idx]] = encoded_output.cpu()
                        # Save latents every 1000 images
                        #if (count+1) % 1000 == 0:
                        
                    pickle.dump(fname_latent_map, open(os.path.join(latent_path,
                                                                            'epoch{0}_class{1}_{2}.pkl'.format(ep, c_i[idx], count)), 'wb'))
                    part_count += 1
                    fname_latent_map = {}
                    count += 1
                    '''
                    
                    image_3d = x_all[idx, 0]  # Removing the channel dimension
                    
                    # Normalize the image to the range [0, 255] for saving
                    image_3d = (image_3d - image_3d.min()) / (image_3d.max() - image_3d.min())  # Normalize
                    image_3d = (image_3d * 255).byte().cpu().numpy()  # Scale to [0, 255] and convert to numpy array
                    
                    # Create a NIfTI image object
                    nifti_image = nib.Nifti1Image(image_3d, np.eye(4))  # Identity matrix as the affine
                    
                    # Save the 3D image as a .nii file
                    nifti_image.to_filename(os.path.join(save_dir, f'image_{idx+1}_ep{ep}_w{w}.nii'))
                print(f'NIfTI images saved in: {save_dir}')
                
                if ep%5==0 or ep == int(n_epoch-1):
                    plot_learning_curve(epoch_list, train_losses, save_dir)
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True, figsize=(8,3))
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample/n_classes)):
                            for col in range(n_classes):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                                plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                        return plots
                    #ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                    #ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    #print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
        # optionally save model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

if __name__ == "__main__":
    train_mnist()

