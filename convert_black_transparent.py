from PIL import Image
import numpy as np

img = np.array(Image.open("test_files/images/interior_input.png").convert("RGBA"))
# Set alpha=0 where black, alpha=255 elsewhere
is_black = np.all(img[:,:,:3] < 10, axis=-1)
img[:,:,3] = np.where(is_black, 0, 255)
Image.fromarray(img).save("test_files/images/interior_input.png")  # overwrite same file