import os
import glob
import numpy as np
import pydicom
from PIL import Image

def create_crops(image, crop_size=(224, 224)):
    crops = []
    img_height, img_width = image.shape

    # Loop through the image to generate crops
    for y in range(0, img_height - crop_size[0] + 1, crop_size[0]):
        for x in range(0, img_width - crop_size[1] + 1, crop_size[1]):
            crop = image[y:y+crop_size[0], x:x+crop_size[1]]
            crops.append(crop)

    return crops

# Load DICOM image
data_path = "../data" 
dicom_path_list = glob.glob(f"{data_path}/*/*.dcm")
output_dir = "../cropped_data"
os.makedirs(output_dir, exist_ok=True)

# Define bounding box limits, assuming x_max is the width of the image
x_min, y_min = 94, 26
x_max, y_max = 2816, 3560  
nb_imgs = 0

for dicom_path in dicom_path_list:
    dicom = pydicom.dcmread(dicom_path)
    image_array = dicom.pixel_array
    # Crop image within bounding box limits
    cropped_image = image_array[y_min:y_max, x_min:x_max]
    # Create 224x224 crops from the cropped image
    crops = create_crops(cropped_image)
    # Save the crops
    for i, crop in enumerate(crops):
        crop_image = Image.fromarray(crop)
        crop_image.save(os.path.join(output_dir, f"crop_{nb_imgs+i+1}.tiff"))
    nb_imgs += len(crops)

print(f"{nb_imgs} crops saved successfully.")
