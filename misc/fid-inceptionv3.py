from torchmetrics.image.fid import FrechetInceptionDistance
import tifffile
import os
import torch


gen_path = './breast-fourier-diff/data/gen_images/scalar/bs200_500epochs'
test_path = './cropped_data_test'

gen_files = os.listdir(gen_path)
test_files = os.listdir(test_path)

real_images = torch.tensor([])
fake_images = torch.tensor([])

for sample_path in sorted(gen_files):
    img_np = tifffile.imread(os.path.join(gen_path, sample_path))
    fake_images = torch.cat([fake_images, torch.tensor(img_np).unsqueeze(0)], dim=0)

for sample_path in sorted(test_files):
    img_np = tifffile.imread(os.path.join(test_path, sample_path))
    real_images = torch.cat([real_images, torch.tensor(img_np).unsqueeze(0)], dim=0)
            
print(f"{len(fake_images)} images found in {str(gen_path)}")
print(f"{len(real_images)} images found in {str(test_path)}")

real_images = real_images.unsqueeze(1)
fake_images = fake_images.unsqueeze(1)

real_images_rgb = torch.cat([real_images] * 3, dim=1)
fake_images_rgb = torch.cat([fake_images] * 3, dim=1)

fid = FrechetInceptionDistance(normalize=True)
fid.update(real_images_rgb, real=True)
fid.update(fake_images_rgb, real=False)

print(f"FID: {float(fid.compute())}")