import torch
import matplotlib.pyplot as plt
import os
from diffusion import ScalarDiffusionModel_VariancePreserving_LinearSchedule, UnconditionalScoreEstimator
from breast_utils import UNet, TimeEncoder, PositionEncoder 
from tqdm import tqdm
import sys
from PIL import Image
import numpy as np
from pydicom import dcmread
import imageio

np.random.seed(42)

device = torch.device('cuda:0')

# --------------------------
# define parameters
# --------------------------
model_path = './data/scalar/weights/diffusion_scalar_unconditional_500.pt'
save_path = './data/gen_images/scalar/bs200_500epochs'
img_shape = 224  
n_gen = 20000
batch_size = 400
# --------------------------

if __name__ == '__main__':

    time_encoder = TimeEncoder(out_channels=32).to(device)
    position_encoder = PositionEncoder(out_channels=32).to(device)
    denoiser = UNet(in_channels=65, out_channels=1, num_base_filters=32).to(device)

    class ScoreEstimator(UnconditionalScoreEstimator):
        def __init__(self, denoiser, time_encoder, position_encoder):
            super().__init__()
            self.denoiser = denoiser
            self.time_encoder = time_encoder
            self.position_encoder = position_encoder

        def forward(self, x_t, t):
            t_enc = self.time_encoder(t.unsqueeze(1))
            pos_enc = self.position_encoder().repeat(x_t.shape[0], 1, 1, 1)
            denoiser_input = torch.cat((x_t, t_enc, pos_enc), dim=1)
            return self.denoiser(denoiser_input)
        
    score_estimator = ScoreEstimator(denoiser, time_encoder, position_encoder).to(device)
    
    def sample_x_T_func(batch_size=None, y=None):
        if batch_size is None:
            raise ValueError('batch_size must be provided')
        return torch.randn(batch_size, 1, img_shape, img_shape).to(device)

    diffusion_model = ScalarDiffusionModel_VariancePreserving_LinearSchedule(score_estimator=score_estimator,
                                                                             sample_x_T_func=sample_x_T_func).to(device)

    if os.path.exists(model_path):
            diffusion_model.load_state_dict(torch.load(model_path))
            print(f'Loaded weights from {model_path}')
    else:
        raise(Exception("Error loading weights, check model_path."))
    # Calculate the number of batches needed
    n_batches = int(np.ceil(n_gen / batch_size))

    # Process in batches
    for batch_idx in tqdm(range(n_batches)):  # Using tqdm for a progress bar
        # Calculate start and end indices for the current batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_gen)

        # Generate batch of random noise
        x_T_batch = torch.randn((end_idx - start_idx, 1, img_shape, img_shape)).to(device)

        # Run the reverse process for the current batch
        x_t_all_batch = diffusion_model.reverse_process(
            x_T=x_T_batch,
            t_stop=0.0,
            num_steps=1024,
            returnFullProcess=False,
            verbose=False
        )
#         observe_every = 200
#         image_filenames = []
#         image_names = []
#         for img_idx, img_tensor in enumerate(x_t_all_batch):  
#             if img_idx % observe_every == 0 or img_idx == 8191:
#                 ###
# #                cur_img = img_tensor.squeeze(0).cpu().numpy()
#                 cur_img = img_tensor.cpu().numpy().squeeze(0).squeeze(0)
#                 img_min = cur_img.min()
#                 img_max = cur_img.max()
#                 cur_img = (cur_img - img_min) / (img_max - img_min)  # Normalize to [0, 1]
#                 cur_img = cur_img * 16122.4727  # Scale based on the max value in training
#                 cur_img = cur_img.astype(np.int16)
#                 cur_img_pil = Image.fromarray(cur_img, mode='I;16')
#                 img_name = f'{save_path}/gen_scalar_{start_idx + img_idx}.tiff'
#                 cur_img_pil.save(img_name)
#                 print(f'Saved image gen_scalar_{start_idx + img_idx}.tiff')
#                 image_names.append(img_name)
                
#                 # Generate and save the histogram for the current iteration
#                 plt.figure(figsize=(10, 4))
#                 hist_aux = img_tensor.cpu().ravel()
#                 plt.hist(hist_aux, bins=50, color='blue', alpha=0.7)
#                 plt.title(f'Histogram at Iteration {img_idx}')
#                 plt.xlabel('Pixel Value')
#                 plt.ylabel('Frequency')
#                 plt.xlim([-4, 4])
#                 plt.grid(True)

#                 # Save the plot as an image file
#                 filename = f'{save_path}scalar_hist_{img_idx:04d}.png'
#                 plt.savefig(filename)
#                 plt.close()  # Close the figure to free memory
#                 image_filenames.append(filename)

#         # After all iterations, create a GIF
#         with imageio.get_writer(f'{save_path}histograms.gif', mode='I') as writer:
#             for filename in image_filenames:
#                 image = imageio.imread(filename)
#                 writer.append_data(image)
#         with imageio.get_writer(f'{save_path}image_histograms.gif', mode='I') as writer:
#             for img in image_names:
#                 image = imageio.imread(img)
#                 writer.append_data(image)
        
        # Process and save each image in the current batch
        for img_idx, img_tensor in enumerate(x_t_all_batch):
            cur_img = img_tensor.squeeze(0).cpu().numpy()
            img_min = cur_img.min()
            img_max = cur_img.max()
            cur_img = (cur_img - img_min) / (img_max - img_min)  # Normalize to [0, 1]
            cur_img = cur_img * 16383  # Scale based on the max value in training
            cur_img = cur_img.astype(np.int16)
            cur_img_pil = Image.fromarray(cur_img, mode='I;16')
            cur_img_pil.save(f'{save_path}/gen_scalar_{start_idx + img_idx}.tiff')
            print(f'Saved image gen_scalar_{start_idx + img_idx}.tiff')