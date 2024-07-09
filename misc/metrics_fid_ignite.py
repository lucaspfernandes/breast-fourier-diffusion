import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import glob
from ignite.metrics import FID
from tqdm import tqdm 


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image = image.convert('F')
        if self.transform:
            image = self.transform(image)
        return image

def load_images(path):
    image_paths = glob.glob(f'{path}/*.tiff')
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[16383, 16383, 16383])
    ])
    dataset = ImageDataset(image_paths, transform)
    return dataset

test_images_path = '../cropped_data_test'
generated_images_path = './data/gen_images/scalar/bs200_500epochs'
#generated_images_path = './data/gen_images/scalar/bs200_500epochs'
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
fid = FID(device=device)

def update_fid(loader, real):
    for images in tqdm(loader, desc="Computing FID"):
        images = images.to(device)
        fid.update((images, real))

# Compute FID
update_fid(loader1, real=True)
update_fid(loader2, real=False)

# Get the FID score
fid_score = fid.compute()
print("FID score:", fid_score)