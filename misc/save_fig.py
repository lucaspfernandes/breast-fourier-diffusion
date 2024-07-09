import matplotlib.pyplot as plt
from PIL import Image
import glob
from pydicom import dcmread
import numpy as np

img_files = sorted(glob.glob('../data/data_tcc/*.dcm'))[:16]
fig, ax = plt.subplots(2, 4)
for i, item in enumerate(img_files):
    ds = dcmread(item)
    img = Image.fromarray(ds.pixel_array)
    ax[i//4, i%4].imshow(img)

plt.show()    