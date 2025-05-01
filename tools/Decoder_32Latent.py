import argparse
import glob
import os
import pickle
import numpy as np

import torch
torch.cuda.empty_cache()

import yaml
import nibabel as nib  # New import for NIfTI format
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataset.celeb_dataset import CelebDataset
from dataset.mnist_dataset import MnistDataset
from models.vqvae32_latent import VQVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infer(args):
    ######## Read the config file #######
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Create the dataset
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])
    
    # This is only used for saving latents. Which as of now
    # is not done in batches hence batch size 1
    data_loader = DataLoader(im_dataset,
                             batch_size=1,
                             shuffle=False)

    num_images = train_config['num_samples']
    idxs = torch.randint(0, len(im_dataset) - 1, (num_images,))
    ims = torch.cat([im_dataset[idx][None, :] for idx in idxs]).float()
    ims = ims.to(device)

    
    model = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']),
                                     map_location=device))
    model.eval()
    with torch.no_grad():
        encoded_output, _ = model.encode(ims)

        #Nasrin
        #z = ims
        z=encoded_output
        print("latents shape is: ", z.shape)
        batch_tensor = z.squeeze(0)

        # Loop through each volume in the batch and save as .nii
        for i in range(batch_tensor.size(0)):
            volume = batch_tensor[i]  # Select the i-th volume, shape [4, 4, 4]
            numpy_volume = volume.detach().cpu().numpy()  # Convert to NumPy array
                    
            # Create a NIfTI image
            nifti_image = nib.Nifti1Image(numpy_volume, affine=np.eye(4))
                
            # Save to a .nii file with a unique filename
            filename = f'/Users/abharian/LDM_Project/Origianl_New_DM_StableDiffusion-PyTorch/data/mnist/train/latents/0/latent_output_volume_{i+1}.nii'
            #filename = f'/Users/abharian/LDM_Project/StableDiffusion-PyTorch/data/mnist/train/latents/{class}/latent_output_volume_{i+1}.nii'
            nib.save(nifti_image, filename)
            print(f"Saved: {filename}")

        #end of Nasrin
        
        print("ims shape is ", ims.shape)
        encoded_output = ims 
        decoded_output = model.decode(encoded_output)
        decoded_output = torch.clamp(decoded_output, -1., 1.)
        decoded_output = (decoded_output + 1) / 2

        # Save 3D images in NIfTI format
        for i in range(num_images):
            image_data = decoded_output[i, 0].cpu().numpy()
            '''min_val = np.min(image_data)
            max_val = np.max(image_data)

            if max_val > min_val:  # Prevent division by zero
                image_data = (image_data - min_val) / (max_val - min_val)
                #image_data = image_data * 2
            else:
                image_data = np.zeros_like(image_data)  # In case all values are identical'''


            nifti_img = nib.Nifti1Image(image_data, affine=np.eye(4))
            print("saved samples in: ", (nifti_img, os.path.join(train_config['task_name'], f'reconstructed_image_{i}.nii')))
            nib.save(nifti_img, os.path.join(train_config['task_name'], f'reconstructed_image_{i}.nii'))
            #print("saved one: ", nifti_img.get_fdata())
            print("minsave is ", np.min(nifti_img.get_fdata()))
            print("maxsave is ", np.max(nifti_img.get_fdata()))
            

        if train_config['save_latents']:
            # save Latents (but in a very unoptimized way)
            latent_path = os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name'])
            latent_fnames = glob.glob(os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name'],
                                                   '*.pkl'))
            assert len(latent_fnames) == 0, 'Latents already present. Delete all latent files and re-run'
            if not os.path.exists(latent_path):
                os.mkdir(latent_path)
            print('Saving Latents for {}'.format(dataset_config['name']))
            
            fname_latent_map = {}
            part_count = 0
            count = 0
            for idx, im in enumerate(tqdm(data_loader)):
                encoded_output, _ , _ = model.encode(im.float().to(device))
                fname_latent_map[str(im_dataset.images[idx])] = encoded_output.cpu()
                #fname_latent_map[im_dataset.images[idx]] = encoded_output.cpu()
                # Save latents every 1000 images
                #if (count+1) % 1000 == 0:
                if (count+1) % 1 == 0:
                    pickle.dump(fname_latent_map, open(os.path.join(latent_path,
                                                                    '2_{}.pkl'.format(part_count)), 'wb'))
                    part_count += 1
                    fname_latent_map = {}
                count += 1
            if len(fname_latent_map) > 0:
                pickle.dump(fname_latent_map, open(os.path.join(latent_path,
                                                   '2_{}.pkl'.format(part_count)), 'wb'))
            print('Done saving latents')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for VQ-VAE inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    infer(args)
