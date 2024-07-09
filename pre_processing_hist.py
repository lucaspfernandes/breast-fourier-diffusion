import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread
import glob
from tqdm import tqdm

# Define the number of bins and the range for a 14-bit image
num_bins = 256  # You can adjust this for more or less granularity
bin_edges = np.linspace(0, 16383, num_bins+1)  # Creates evenly spaced bin edges from 0 to 16383

# Initialize an array to store the aggregated histogram
# The length is num_bins which is one less than the number of bin_edges
aggregated_histogram = np.zeros(num_bins)

# Loop through your dataset (adjust the path to your images)
with tqdm(glob.glob('../data/data_tcc/Flat_600.700.500_271_4-recon-5x2.dcm')) as img_path_list:
    for image_path in img_path_list: 
        ds = dcmread(image_path)
        img_array = ds.pixel_array
        # Calculate the histogram for this image
        # np.histogram returns the counts and the bin edges, but we only need the counts here
        image_histogram, _ = np.histogram(img_array, bins=bin_edges)
        # Add this image's histogram to the aggregated histogram
        aggregated_histogram += image_histogram

# Now that we have the aggregated histogram, let's plot it
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], aggregated_histogram, width=np.diff(bin_edges), edgecolor='black', align='edge')
plt.title('Example histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.xlim(0, 16383)  # Adjust the x-axis to match the 14-bit range
plt.savefig('./data/example_hist.png', dpi=300)
plt.show()

