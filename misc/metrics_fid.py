import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import glob
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm 


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('L')
        image = image.convert('RGB')  # Convert grayscale to RGB
        if self.transform:
            image = self.transform(image)
        return image

def load_images(path):
    image_paths = glob.glob(f'{path}/*.tiff')
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = ImageDataset(image_paths, transform)
    return dataset

test_images_path = '../cropped_data_test'
#generated_images_path = './data/gen_images/scalar/bs200_500epochs'
generated_images_path = './data/gen_images/fourier/param_1_3/bs200_500epochs_16steps'
bs = 400
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on device = {device}')

# Load datasets
dataset1 = load_images(test_images_path)
dataset2 = load_images(generated_images_path)

# Create data loaders
loader1 = DataLoader(dataset1, batch_size=bs)
loader2 = DataLoader(dataset2, batch_size=bs)

# Initialize FID
fid = FrechetInceptionDistance(feature=64, normalize=True).to(device)

# Compute features from real images
for real_images in tqdm(loader1, desc="Processing Real Images"):
    fid.update(real_images.to(device), real=True)

# Compute features from generated images
for generated_images in tqdm(loader2, desc="Processing Generated Images"):
    fid.update(generated_images.to(device), real=False)

# Compute FID score
fid_value = fid.compute()
print("FID score:", fid_value)