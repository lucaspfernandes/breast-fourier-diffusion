# load the breast dataset and train a scalar diffusion model on it
import torch
import matplotlib.pyplot as plt
import os
from diffusion import ScalarDiffusionModel_VariancePreserving_LinearSchedule, UnconditionalScoreEstimator
from breast_utils import UNet, TimeEncoder, PositionEncoder, breast_train_loader 
from tqdm import tqdm
import pandas as pd
import sys

device = torch.device('cuda:0') #select your device

# --------------------------
# define parameters
# --------------------------
verbose=True
loadPreviousWeights=False
runTraining=True #check train_dir in breast_utils
runTesting=False
n_epochs = 500
runReverseProcess=True
save_interval = 10
plot_interval = 5
img_shape = 224
#set batch_size on breast_utils.py !!!!

previous_checkpoint_path = './data/weights/diffusion_scalar_unconditional.pt'
animations_path = './data/animations_train'
model_path = './data/weights'
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


    if loadPreviousWeights:
        if os.path.exists(previous_checkpoint_path):
            diffusion_model.load_state_dict(torch.load(previous_checkpoint_path))
            print(f'Loaded weights from {previous_checkpoint_path}')

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-3)
    
    epoch_list = []
    loss_list = []
    for epoch in tqdm(range(1, n_epochs+1), desc='Epochs'):
        # run the training loop
        if runTraining:
            for i, x_0_batch in enumerate(breast_train_loader):
                optimizer.zero_grad()
                x_0_batch = x_0_batch.to(device)
                loss = diffusion_model.compute_loss(x_0_batch, loss_name='elbo')
                loss.backward()
                optimizer.step()
                if verbose:
                    print(f'Epoch {epoch}, iteration {i+1}, loss {loss.item()}')
                sys.stdout.flush()
            epoch_list.append(epoch)
            loss_list.append(loss.item())

        # run the test loop
#        if runTesting:
#            for i, (x_0_batch, _) in enumerate(mnist_test_loader):
#                x_0_batch = x_0_batch.to(device)
#                loss = diffusion_model.compute_loss(x_0_batch, loss_name='elbo')
#                if verbose:
#                    print(f'Epoch {epoch}, iteration {i}, test loss {loss.item()}')
        
        if runReverseProcess and (epoch % plot_interval == 0):
            x_T = torch.randn((4,1,img_shape,img_shape)).to(device)
            # run the reverse process loop
            x_t_all = diffusion_model.reverse_process(
                            x_T=x_T[0:4],
                            batch_size=4,
                            t_stop=0.0,
                            num_steps=1024,
                            returnFullProcess=True,
                            verbose=False
                            )

            # plot the results as an animation
            from matplotlib.animation import FuncAnimation

            fig, ax = plt.subplots(2, 2)
            ims = []
            x_t_all = x_t_all.cpu().numpy()
            for i in range(4):
                im = ax[i//2, i%2].imshow(x_t_all[0,i, 0, :, :], animated=True)
                im.set_clim(-1, 1)
                ims.append([im])
            print('Animating frames')
            def updatefig(frame):
                #print('Animating frame ', frame)
                for i in range(4):
                    if frame < 64:
                        ims[i][0].set_array(x_t_all[frame*16 + 15,i, 0, :, :])
                return [im[0] for im in ims]
            
            ani = FuncAnimation(fig, updatefig, frames=range(64), interval=50, blit=True)
            
            ani.save(f'{animations_path}/diffusion_scalar_unconditional_{epoch}.gif')
        if epoch % save_interval == 0:
            torch.save(diffusion_model.state_dict(), f'{model_path}/diffusion_scalar_unconditional_{epoch}.pt')
        sys.stdout.flush()
        
    # create a dataframe with loss values
    loss_per_epoch = pd.DataFrame({
        'epoch': epoch_list,
        'Loss': loss_list})

    # save dataframe as csv
    loss_path = f'{model_path}/loss_per_epoch.csv' 
    loss_per_epoch.to_csv(loss_path, index=False)
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, loss_list, linestyle='-', color='blue')
    plt.title('Loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
#    plt.xticks(epoch_list)
    plt.tight_layout()
    # save ploto as png
    plt.savefig(f'{model_path}/loss_per_epoch.png')


        