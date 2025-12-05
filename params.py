'''
Changing parameters to see correlation with AP/HTPB/Interface ratios
Make a bunch of for loops, changing one param at a time

Do 10 random generators per param change
radius size: go from 1% to 10%, 100 steps
% comp: 30% to 80%, 100 steps

last: standard deviation, 0.3 to 0.8, 100 steps
plot param vs %AP, %HTPB, and % interface on separate plots
'''

import numpy as np
import os
import gen_meso as gm
import line_comp as lc

#temporary file directory
folder = "./generated_images"
os.makedirs(folder, exist_ok=True)  # create folder if it doesn't exist

# Full path for saving the figure
temp_ignore = os.path.join(folder, "ignore.png") #we won't actually use this
temp_img = os.path.join(folder, "temp_meso.png")
img_trimmed = os.path.join(folder, "trimmed.png")



#set of points to test for each parameter
radius_set = np.linspace(0.02, 0.1, 5)
comp_set = np.linspace(0.40, 0.80, 2)
dev_set = np.linspace(0.3, 0.8, 50)

#set other parameters as constant while testing one
radius_const = np.median(radius_set) #0.055
comp_const = np.median(comp_set) #0.6
dev_const = np.median(dev_set) #0.45

n = len(radius_set)

def test_radius(radius_set):
    combined_avg = [] #this will be compositions, AP/HTPB/Interface
    '''Test a parameter while keeping the others constant'''
    #do five microstructure generations and analyses per point
    for i in range(n): #for each 
        mini_list = []
        for j in range(5):
            img_trimmed = img_trimmed = os.path.join(folder, f"trimmed_{i}.png")
            temp_ignore = os.path.join(folder, f"ignore_{i}.png")
            gm.gen_struct(temp_ignore, temp_img, 1, radius_set[i], comp_const, dev_const, 50000)
            lc.trim_edges(temp_img, img_trimmed)
            mini_list.append(lc.vert_avg(img_trimmed))
        combined_avg.append(np.mean(mini_list, axis=0))
    return combined_avg

def test_comp(comp_set):
    combined_avg = [] #this will be compositions, AP/HTPB/Interface
    '''Test a parameter while keeping the others constant'''
    #do five microstructure generations and analyses per point
    for i in range(n): #for each 
        mini_list = []
        for j in range(1):
            img_trimmed = img_trimmed = os.path.join(folder, f"trimmed_{i}.png")
            temp_ignore = os.path.join(folder, f"ignore_{i}.png")
            gm.gen_struct(temp_ignore, temp_img, 1, radius_const, comp_set[i], dev_const, 50000)
            lc.trim_edges(temp_img, img_trimmed)
            mini_list.append(lc.vert_avg(img_trimmed))
        combined_avg.append(np.mean(mini_list, axis=0))
    return combined_avg

def test_dev(dev_set):
    combined_avg = [] #this will be compositions, AP/HTPB/Interface
    '''Test a parameter while keeping the others constant'''
    #do five microstructure generations and analyses per point
    for i in range(n): #for each 
        mini_list = []
        for j in range(1):
            img_trimmed = img_trimmed = os.path.join(folder, f"trimmed_{i}.png")
            temp_ignore = os.path.join(folder, f"ignore_{i}.png")
            gm.gen_struct(temp_ignore, temp_img, 1, radius_const, comp_const, dev_set[i], 50000)
            lc.trim_edges(temp_img, img_trimmed)
            mini_list.append(lc.vert_avg(img_trimmed))
        combined_avg.append(np.mean(mini_list, axis=0))
    return combined_avg

if __name__ == "__main__":
    list = test_radius(radius_set)
    print(list)