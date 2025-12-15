#pull line composition from an image

import numpy as np
from PIL import Image, ImageDraw

#image

save_path = ".\generated_images\mesostructure_untitled.png"
copy_path = ".\generated_images\mesostructure_line.png" #image copy with line
trim_path = ".\generated_images\mesostructure_trimmed.png"
#untitled images set to 1024x1024 in previous script
#trimmed images 922x922

def line_comp(save_path, column_index):
    '''Pull a vertical line from an image and return % comp of AP, HTPB, and interface (red, blue, purple)'''
    
    img = Image.open(save_path).convert("RGB")
    arr = np.array(img)
    #pick a line
    line = arr[:, column_index, :] 
    
    #count pixels
    is_red = np.all(line == [255, 0, 0], axis=1)
    is_blue = np.all(line == [0, 0, 255], axis=1)

    # purple = anything that isn't red or blue
    is_purple = ~(is_red | is_blue)

    # counts
    AP_count = np.sum(is_red)
    HTPB_count = np.sum(is_blue)
    interface_count = np.sum(is_purple)
    total = line.shape[0]
    
    #convert to percentages
    ap = (AP_count / total) * 100
    htpb = (HTPB_count / total) * 100
    int = (interface_count / total) * 100
    
    return ap, htpb, int

def draw_line(save_path, column_index, copy_path):
    '''return image with line showing which column was taken'''
    img = Image.open(save_path).convert("RGB")
    img_with_line = img.copy()
    draw = ImageDraw.Draw(img_with_line)
    draw.line([(column_index, 0), (column_index, img_with_line.height)], fill='white', width=1)
    img_with_line.save(copy_path)
    return img_with_line

#find average composition of a vertical line by looping over whole thing
#probably will want to trim off edges to get to the middle, so we don't have areas where AP = 0
def vert_avg_fast(save_path):
    img = Image.open(save_path).convert("RGB")
    arr = np.array(img)
    tol = 10
    # Red = AP
    is_red = np.all(np.abs(arr - [255, 0, 0]) <= tol, axis=2)
    # Blue = HTPB
    is_blue = np.all(np.abs(arr - [0, 0, 255]) <= tol, axis=2)
    # Anything else = interface
    is_purple = ~(is_red | is_blue)

    
    total_pixels = arr.shape[0] * arr.shape[1]
    
    ap = np.sum(is_red) / total_pixels * 100
    htpb = np.sum(is_blue) / total_pixels * 100
    interface = np.sum(is_purple) / total_pixels * 100
    
    return ap, htpb, interface

def vert_avg(save_path):
    img = Image.open(save_path).convert("RGB")
    arr = np.array(img)
    comp_list = []
    for col in range(arr.shape[1]):
        comp_list.append(line_comp(save_path, col))
    comp_array = np.array(comp_list)
    ap_av, htpb_av, int_av = np.mean(comp_array, axis=0)
    return ap_av, htpb_av, int_av

#DON"T NEED THIS WITH UPDATED GEN_STRUCT
def trim_edges(save_path, trim_path):
    '''trims edges, takes 5% of square image's width off each edge, so 1x1 is now 0.9x0.9'''
    img = Image.open(save_path).convert("RGB")
    
    w, h = img.size
    trim_h = int(0.05*h)
    trim_w = int(0.05*w)
    #crop box, left, top, right, bottom
    crop_box = (trim_w, trim_h, w - trim_w, h - trim_h)
    trimmed_img = img.crop(crop_box)
    trimmed_img.save(trim_path)
    return trimmed_img


    
if __name__ == "__main__":
    trim_edges(save_path, trim_path)
    ap, htpb, int = line_comp(trim_path, 75)
    print(f'Column 75: AP: {ap:.2f} %, HTPB: {htpb:.2f} %, Interface: {int:.2f} %')
    draw_line(trim_path, 75, copy_path)
    ap, htpb, int = line_comp(trim_path, 225)
    print(f'Column 225: AP: {ap:.2f} %, HTPB: {htpb:.2f} %, Interface: {int:.2f} %')
    draw_line(copy_path, 225, copy_path)
    ap, htpb, int = line_comp(trim_path, 375)
    print(f'Column 375: AP: {ap:.2f} %, HTPB: {htpb:.2f} %, Interface: {int:.2f} %')
    draw_line(copy_path, 375, copy_path)
    ap, htpb, int = line_comp(trim_path, 525)
    print(f'Column 525: AP: {ap:.2f} %, HTPB: {htpb:.2f} %, Interface: {int:.2f} %')
    draw_line(copy_path, 525, copy_path)
    ap, htpb, int = line_comp(trim_path, 675)
    print(f'Column 675: AP: {ap:.2f} %, HTPB: {htpb:.2f} %, Interface: {int:.2f} %')
    draw_line(copy_path, 675, copy_path)
    ap, htpb, int = line_comp(trim_path, 825)
    print(f'Column 825: AP: {ap:.2f} %, HTPB: {htpb:.2f} %, Interface: {int:.2f} %')
    draw_line(copy_path, 825, copy_path)
    ap_av, htpb_av, int_av = vert_avg(trim_path)
    print(f'Vertical Avg: AP: {ap_av:.2f} %, HTPB: {htpb_av:.2f} %, Interface: {int_av:.2f} %')

