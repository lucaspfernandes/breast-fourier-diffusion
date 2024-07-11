import torch
import matplotlib.pyplot as plt
import os
from diffusion import ScalarDiffusionModel_VariancePreserving_LinearSchedule, UnconditionalScoreEstimator
from breast_utils import UNet, TimeEncoder, PositionEncoder 
from tqdm import tqdm
import sys
from PIL import Image
import numpy as np
import imageio

np.random.seed(42)

device = torch.device('cuda:0') #select your device

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