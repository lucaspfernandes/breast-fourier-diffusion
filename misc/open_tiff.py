from PIL import Image
import matplotlib.pyplot as plt
import glob
from pathlib import Path

img_files = sorted(glob.glob('./data/gen_images/scalar/*.tiff'))
for item in img_files:
    img = Image.open(item)
    plt.title(Path(item).stem)
    plt.imshow(img)
    plt.show()


