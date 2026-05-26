#test code for image processing
#Jenna Stocker 10/07/2025

from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance

img_input = "test_files/images/interior_input.png"
img_output = "test_files/images/interior_input_blurred.png" #red and blue image, can be bmp


def convert_to_BW(img_input):
    '''convert raw image to clear B&W'''
    im = Image.open(img_input).convert("RGBA")
    #image mode: red, green, blue, alpha    
    #alpha is opacity. ranges from 0 (transparent) to 1 (opaque)    

    #blur to clean up spaces
    im = im.filter(ImageFilter.GaussianBlur(radius=1))

    #turn up the contrast
    enh = ImageEnhance.Contrast(im)
    im = enh.enhance(10) #% more contrast, negative will invert

    #convert to pure black and white
    gs_im = im.convert("L") #grayscale
    threshold = 128

    im_bw = gs_im.point(lambda p: 255 if p > threshold else 0)
    im_bw = im_bw.convert("RGB")  # minimal edit to add RGB channels
    im_bw = im.filter(ImageFilter.GaussianBlur(radius=14))
    return im_bw

def get_img_info(im):
    '''get info about B&W image'''
    pixels = im.load() 
    width, height = im.size #dimensions in pixels
    return (pixels, width, height)
    

def change_colors(pixels, width, height): 
    '''get image data, convert from B&W to red and blue'''

    #color codes, old
    #black = (0,0,0)
    #white =(253, 253, 253)
    #color codes, new
    blue = (0, 0, 255)
    red = (255, 0, 0)
    
    #create new RGB image
    new_image = Image.new("RGB", (width, height))
    
    for i in range(width):
        for j in range(height):
            color = pixels[i,j]  
            if color == 0: #black to blue
                new_image.putpixel((i, j), blue)
            else: #white to red
                new_image.putpixel((i, j),red)
    #blur it a bit
    new_image = new_image.filter(ImageFilter.GaussianBlur(radius=4))
    return new_image

def main():
    #img_BW = convert_to_BW(img_input)
    #size, w, h = get_img_info(img_BW)
    #colored_image = change_colors(size, w, h)
    #colored_image.show()
    #colored_image.save(img_output)
    #img_BW.save(img_output)
    im = Image.open(img_input).convert("RGBA")
    #image mode: red, green, blue, alpha    
    #alpha is opacity. ranges from 0 (transparent) to 1 (opaque)    

    #blur to clean up spaces
    im = im.filter(ImageFilter.GaussianBlur(radius=6))
    im.save(img_output)

    print(f"Saved updated image as {img_output}")


if __name__ == "__main__":
    main()