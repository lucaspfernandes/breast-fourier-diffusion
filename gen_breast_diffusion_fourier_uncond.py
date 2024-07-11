# gen images  using a breast diffusion model 
import torch
import matplotlib.pyplot as plt
import os
from diffusion import FourierDiffusionModel, UnconditionalScoreEstimator
from breast_utils import UNet, TimeEncoder, PositionEncoder
from PIL import Image
import numpy as np
from tqdm import tqdm

device = torch.device('cuda:0') #select your device

np.random.seed(42)

# --------------------------
# define parameters
# --------------------------
img_shape = 224
model_path = './data/fourier/param_3_6/weights/diffusion_fourier_unconditional_500.pt'
save_path = './data/gen_images/fourier/param_3_6/bs200_500epochs'
n_gen = 12000
batch_size = 400
# --------------------------

if __name__ == '__main__':
    

    time_encoder = TimeEncoder(out_channels=32).to(device)
    position_encoder = PositionEncoder(out_channels=32).to(device)
    denoiser = UNet(in_channels=65, out_channels=1, num_base_filters=32).to(device)

    class ScoreEstimator(UnconditionalScoreEstimator): #modified
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

    # now do it with torch
    def gaussian_blur_fourier_transfer_function(fwhm, size):
        """Generate a Gaussian blur transfer function with a given FWHM and size."""
        sigma = fwhm / (2.0 * torch.sqrt(2.0 * torch.log(torch.tensor(2.0))))
        xGrid = torch.linspace(-size // 2, size // 2, steps=size).to(device)
        yGrid = torch.linspace(-size // 2, size // 2, steps=size).to(device)
        xGrid, yGrid = torch.meshgrid(xGrid, yGrid)
        rGrid = torch.sqrt(xGrid**2 + yGrid**2)
        y = torch.exp(-rGrid**2 / (2 * sigma**2))
        y /= y.sum()  # Normalize
        y = torch.fft.fft2(y)
        y = torch.abs(y)
        return y
    
    #here you can set the filter parametrization size in mm (equations 25, 26 and 27)
    fourier_transfer_function_LPF = gaussian_blur_fourier_transfer_function(3.0, img_shape)
    fourier_transfer_function_BPF = gaussian_blur_fourier_transfer_function(1.0, img_shape) - fourier_transfer_function_LPF
    fourier_transfer_function_HPF = torch.ones(img_shape, img_shape).to(device) - fourier_transfer_function_BPF - fourier_transfer_function_LPF
    
    fourier_transfer_function_LPF = fourier_transfer_function_LPF.unsqueeze(0).unsqueeze(0)
    fourier_transfer_function_BPF = fourier_transfer_function_BPF.unsqueeze(0).unsqueeze(0)
    fourier_transfer_function_HPF = fourier_transfer_function_HPF.unsqueeze(0).unsqueeze(0)

    def modulation_transfer_function_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, img_shape, img_shape)
        LPF = fourier_transfer_function_LPF.repeat(t.shape[0], 1, 1, 1) * torch.exp(-5*_t*_t)
        BPF = fourier_transfer_function_BPF.repeat(t.shape[0], 1, 1, 1) * torch.exp(-7*_t*_t)
        HPF = fourier_transfer_function_HPF.repeat(t.shape[0], 1, 1, 1) * torch.exp(-9*_t*_t)
        return LPF + BPF + HPF

    def modulation_transfer_function_derivative_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, img_shape, img_shape)
        LPF = fourier_transfer_function_LPF.repeat(t.shape[0], 1, 1, 1) * (-10*_t * torch.exp(-5*_t*_t))
        BPF = fourier_transfer_function_BPF.repeat(t.shape[0], 1, 1, 1) * (-14*_t * torch.exp(-7*_t*_t))
        HPF = fourier_transfer_function_HPF.repeat(t.shape[0], 1, 1, 1) * (-18*_t * torch.exp(-9*_t*_t))
        return LPF + BPF + HPF + 1e-10

    def noise_power_spectrum_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, img_shape, img_shape)
        LPF = fourier_transfer_function_LPF.repeat(t.shape[0], 1, 1, 1) * (1.0 - torch.exp(-10*_t*_t))
        BPF = fourier_transfer_function_BPF.repeat(t.shape[0], 1, 1, 1) * (1.0 - torch.exp(-14*_t*_t))
        HPF = fourier_transfer_function_HPF.repeat(t.shape[0], 1, 1, 1) * (1.0 - torch.exp(-18*_t*_t))
        return LPF + BPF + HPF + 1e-10

    def noise_power_spectrum_derivative_func(t):
        _t = t.reshape(-1, 1, 1, 1).repeat(1, 1, img_shape, img_shape)
        LPF = fourier_transfer_function_LPF.repeat(t.shape[0], 1, 1, 1) *  (20*_t * torch.exp(-10*_t*_t))
        BPF = fourier_transfer_function_BPF.repeat(t.shape[0], 1, 1, 1) *  (28*_t * torch.exp(-14*_t*_t))
        HPF = fourier_transfer_function_HPF.repeat(t.shape[0], 1, 1, 1) *  (36*_t * torch.exp(-18*_t*_t))
        return LPF + BPF + HPF + 1e-10


    # Create the FourierDiffusionModel using the functions we've defined
    diffusion_model = FourierDiffusionModel(
        score_estimator=score_estimator,
        modulation_transfer_function_func=modulation_transfer_function_func,
        noise_power_spectrum_func=noise_power_spectrum_func,
        modulation_transfer_function_derivative_func=modulation_transfer_function_derivative_func,
        noise_power_spectrum_derivative_func=noise_power_spectrum_derivative_func
    )
    
    if os.path.exists(model_path):
        diffusion_model.load_state_dict(torch.load(model_path))
        print(f'Loaded weights from {model_path}')
    else:
        raise(Exception("Error loading weights, check model_path."))
    
    n_batches = int(np.ceil(n_gen / batch_size))
    
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
            cur_img_pil.save(f'{save_path}/gen_fourier_{start_idx + img_idx}.tiff')
            print(f'Saved image gen_fourier_{start_idx + img_idx}.tiff')


    
    