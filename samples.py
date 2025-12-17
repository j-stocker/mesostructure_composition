'''Generate:
    - 50 unimodal distributions with approximately the same volume % of AP but varying particle sizes
    - 50 unimodal distributions with varying volume % of AP but the same particle sizes
    - 50 bimodal distributions with approximately the same volume % of AP but varying particle sizes
    - 50 bimodal distributions with varying volume % of AP but the same particle sizes'''
import os
import gen_meso as gm
import numpy as np
import glob
import line_comp as lc
import gen_plots as gp
from scipy import ndimage

from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
#clear folders to generate new images


def clear_folder(folder_path):
    folder_path = os.path.abspath(folder_path)
    base = os.path.abspath("./generated_images")

    if folder_path == base:
        raise RuntimeError("Refusing to clear generated_images root")

    for f in glob.glob(os.path.join(folder_path, "*")):
        if os.path.isfile(f):
            os.remove(f)


# Parameters
std_dev_uni = 0.2
AP_vol_50A = 0.5
avg_rad_50A = np.linspace(40e-6, 140e-6, 200)

avg_rad_50B = 70e-6  # fixed particle size
AP_vol_50B = np.linspace(0.35, 0.65, 200)  # varying AP

std_dev_bi = [0.2, 0.25] #standard dev of coarse and fine groups
mean_rad_bi_size = [np.linspace(80e-6, 110e-6, 200), 25e-6]  # varying size, coarse/fine
AP_fixed_bi = 0.55

mean_rad_bi_AP = [100e-6, 20e-6]  # fixed size, coarse/fine
AP_bi_var = np.linspace(0.40, 0.55, 3)
#AP_bi_var = np.linspace(0.35, 0.65, 5)  # varying AP
mix_bi = 1/21 #coarse:fine

img_size = 1
physical_size = 1e-2
max_attempts = 100000

interface_thickness = 2e-6 #2 um

folderA_png = "./generated_images/unimodal_varying_part_size/pngs"
folderA_xyzr = "./generated_images/unimodal_varying_part_size/xyzrs"

folderB_png = "./generated_images/unimodal_varying_AP_ratio/pngs"
folderB_xyzr = "./generated_images/unimodal_varying_AP_ratio/xyzrs"

folderC_png = "./generated_images/bimodal_varying_part_size/pngs"
folderC_xyzr = "./generated_images/bimodal_varying_part_size/xyzrs"


folderD_png = "./generated_images/bimodal_varying_AP_ratio/pngs"
folderD_xyzr = "./generated_images/bimodal_varying_AP_ratio/xyzrs"

folderE_png = "./generated_images/test/pngs"
folderE_xyzr = "./generated_images/test/xyzrs"

folder_ignore = "./generated_images/ignore"

resultsA = "./generated_images/unimodal_varying_part_size/results_uni_part_size.txt"
resultsB = "./generated_images/unimodal_varying_AP_ratio/results_uni_AP_ratio.txt"
resultsC = "./generated_images/bimodal_varying_part_size/results_bi_part_size.txt"
resultsD = "./generated_images/bimodal_varying_AP_ratio/results_bi_AP_ratio.txt"
resultsE = "./generated_images/test/results.txt"


def reset():
    #clear_folder(folderA_png)
    #clear_folder(folderA_xyzr)

    #clear_folder(folderB_png)
    #clear_folder(folderB_xyzr)

    #clear_folder(folderC_png)
    #clear_folder(folderC_xyzr)

    #clear_folder(folderD_png)
    #clear_folder(folderD_xyzr)
    
    clear_folder(folderE_png)
    clear_folder(folderE_xyzr)


    #os.makedirs(folderA_png, exist_ok=True)
    #os.makedirs(folderA_xyzr, exist_ok=True)

    #os.makedirs(folderB_png, exist_ok=True)
    #os.makedirs(folderB_xyzr, exist_ok=True)

    #os.makedirs(folderC_png, exist_ok=True)
    #os.makedirs(folderC_xyzr, exist_ok=True)

    #os.makedirs(folderD_png, exist_ok=True)
    #os.makedirs(folderD_xyzr, exist_ok=True)
    
    os.makedirs(folderE_png, exist_ok=True)
    os.makedirs(folderE_xyzr, exist_ok=True)

    os.makedirs(folder_ignore, exist_ok=True)


    for fp in [resultsA, resultsB]:
        os.makedirs(os.path.dirname(fp), exist_ok=True)



save_path_ignore = os.path.join(folder_ignore, "ignore.png")

def generateA():
    results = []
    # --- A: Unimodal, varying particle size ---
    for i, radius in enumerate(avg_rad_50A):
        
        temp_png = os.path.join(folderA_png, f"temp_{i}.png")
        temp_xyzr = os.path.join(folderA_xyzr, f"temp_{i}.xyzr")
        
        _, _, AP_achieved, _ = gm.gen_struct(
            temp_png, save_path_ignore, temp_xyzr,
            img_size, physical_size, radius, AP_vol_50A,
            std_dev_uni, max_attempts, mode=1, max_tries=20
        )

        AP_str = str(int(AP_achieved * 10000))  # 4-digit AP fraction without leading 0
        radius_um = int(radius * 1e7) #3 digit radius
        final_png = os.path.join(folderA_png, f"uni_R{radius_um}um_AP{AP_str}.png")
        final_xyzr = os.path.join(folderA_xyzr, f"uni_R{radius_um}um_AP{AP_str}.xyzr")
        ap, htpb, interface = lc.vert_avg_fast(save_path_ignore)
        results.append((radius_um, ap, htpb, interface))
        if os.path.exists(temp_png):
            os.replace(temp_png, final_png)
        if os.path.exists(temp_xyzr):
            os.replace(temp_xyzr, final_xyzr)
        else:
            print(f"Warning: {temp_xyzr} was not created!")
    with open(resultsA, "w") as f:
        f.write("radius in um, AP_vert, HTPB_vert, Interface_vert\n")  # optional header
        for item in results:
            f.write(f"{item[0]}, {item[1]}, {item[2]}, {item[3]}\n")



            
def generateB():
    results = []
    # --- B: Unimodal, varying AP fraction ---
    for i, AP_target in enumerate(AP_vol_50B):
        temp_png = os.path.join(folderB_png, f"temp_{i}.png")
        temp_xyzr = os.path.join(folderB_xyzr, f"temp_{i}.xyzr")

        _, _, AP_achieved, _ = gm.gen_struct(
            temp_png, save_path_ignore, temp_xyzr,
            img_size, physical_size, avg_rad_50B, AP_target,
            std_dev_uni, max_attempts, mode=1, max_tries=5
        )

        AP_str = str(int(AP_achieved * 10000))
        radius_um = int(avg_rad_50B * 1e6)
        final_png = os.path.join(folderB_png, f"uni_AP{AP_str}_R{radius_um}um.png")
        final_xyzr = os.path.join(folderB_xyzr, f"uni_AP{AP_str}_R{radius_um}um.xyzr")
        ap, htpb, interface = lc.vert_avg_fast(save_path_ignore)
        results.append((AP_achieved*100, ap, htpb, interface))
        if os.path.exists(temp_png):
            os.replace(temp_png, final_png)
        if os.path.exists(temp_xyzr):
            os.replace(temp_xyzr, final_xyzr)
        else:
            print(f"Warning: {temp_xyzr} was not created!")
    with open(resultsB, "w") as f:
        f.write("AP_achieved, AP_vert, HTPB_vert, Interface_vert\n")  # optional header
        for item in results:
            f.write(f"{item[0]}, {item[1]}, {item[2]}, {item[3]}\n")


def generateC():
    results = []
    
    # --- C: Bimodal, varying particle size ---
    for i, coarse_radius in enumerate(mean_rad_bi_size[0]):
        avg_radius = []
        avg_radii = []
        fine_radius = mean_rad_bi_size[1]  # pick corresponding fine particle
        radius_input = [float(coarse_radius), float(fine_radius)]  # pass as list for bimodal

        temp_png = os.path.join(folderC_png, f"temp_{i}.png")
        temp_xyzr = os.path.join(folderC_xyzr, f"temp_{i}.xyzr")
        
        _, _, AP_achieved, _ = gm.gen_struct(
            temp_png, save_path_ignore, temp_xyzr,
            img_size, physical_size, radius_input, AP_fixed_bi,
            std_dev_bi, max_attempts, mode=2, mix=mix_bi, max_tries=200
        )

        #pull average radius from xyzr
        with open(temp_xyzr, "r") as f:
            for line in f:# 4th column = radius
                parts = line.strip().split()
                avg_radii.append(float(parts[3]))
            image_avg_radius = np.mean(avg_radii)
        avg_radius.append(image_avg_radius*1e6) #put in um

        AP_str = str(int(AP_achieved * 10000))  # 4-digit AP fraction
        radius_um = int(coarse_radius * 1e6) #3 digit radius
        final_png = os.path.join(folderC_png, f"bi_R{radius_um}um_AP{AP_str}.png")
        final_xyzr = os.path.join(folderC_xyzr, f"bi_R{radius_um}um_AP{AP_str}.xyzr")
        ap, htpb, interface = lc.vert_avg_fast(save_path_ignore)
        
        results.append((AP_achieved, ap, htpb, interface, avg_radius[0]))
        if os.path.exists(temp_png):
            os.replace(temp_png, final_png)
        if os.path.exists(temp_xyzr):
            os.replace(temp_xyzr, final_xyzr)
        else:
            print(f"Warning: {temp_xyzr} was not created!")
    with open(resultsC, "w") as f:
        f.write("AP_achieved, AP_vert, HTPB_vert, Interface_vert, Avg Radius\n")  # optional header
        for item in results:
            f.write(f"{item[0]}, {item[1]}, {item[2]}, {item[3]}, {item[4]}\n")

def generateD():
    results = []
    # --- D: Bimodal, varying AP fraction ---
    r_target = 30e-6 #avg radius 70 micrometers  
    mix = mix_bi                   # number fraction of coarse

    coarse_radius = mean_rad_bi_AP[0]  # keep fixed
    fine_radius = (r_target - mix * coarse_radius) / (1 - mix)

    r_avg = mix_bi * coarse_radius + (1 - mix_bi) * fine_radius
    r_avg_um = int(r_avg * 1e6)


    for i, AP_target in enumerate(AP_bi_var):
        avg_radius = []
        avg_radii = []
        temp_png = os.path.join(folderD_png, f"temp_{i}.png")
        temp_xyzr = os.path.join(folderD_xyzr, f"temp_{i}.xyzr")
        radius_input = [coarse_radius, fine_radius]


        
        _, _, AP_achieved, _ = gm.gen_struct(
            temp_png, save_path_ignore, temp_xyzr,
            img_size, physical_size, radius_input, AP_target,
            std_dev_bi, max_attempts, mode=2, mix=mix_bi, max_tries=2
        )
        
        with open(temp_xyzr, "r") as f:
            avg_radii = [float(line.strip().split()[3]) for line in f]
        image_avg_radius = np.mean(avg_radii)
        avg_radius.append(image_avg_radius*1e6)


        AP_str = str(int(AP_achieved * 10000))  # 4-digit AP fraction
        final_png = os.path.join(folderD_png, f"bi_AP{AP_str}_R{image_avg_radius*1e6:.1f}um.png")
        final_xyzr = os.path.join(folderD_xyzr, f"bi_AP{AP_str}_R{image_avg_radius*1e6:.1f}um.xyzr")


        ap, htpb, interface = lc.vert_avg(save_path_ignore)
        results.append((AP_achieved, ap, htpb, interface, avg_radius[0]))
        if os.path.exists(temp_png):
            os.replace(temp_png, final_png)
        if os.path.exists(temp_xyzr):
            os.replace(temp_xyzr, final_xyzr)
        else:
            print(f"Warning: {temp_xyzr} was not created!")
    with open(resultsD, "w") as f:
        f.write("AP_achieved, AP_vert, HTPB_vert, Interface_vert, Avg Radius\n")  # optional header
        for item in results:
            f.write(f"{item[0]}, {item[1]}, {item[2]}, {item[3]}, {item[4]}\n")
import imageio.v2 as imageio
from imageio import imread
from scipy import ndimage
from skimage.morphology import skeletonize
from scipy.integrate import quad
'''
def mean_distance_to_interface(r_array, phi):
    def integrand(r):
        return np.exp(-phi * np.mean((r_array + r)**2) / np.mean(r_array**2))

    ED, _ = quad(integrand, 0, np.inf)
    return ED
'''


from scipy.integrate import quad
'''
def mean_distance_to_interface(r_array, phi):
    r_array = np.array(r_array)
    R_mean = np.mean(r_array)
    ratio = r_array / R_mean  # dimensionless

    # Define numerically stable integrand
    def integrand(r):
        # r is scaled by R_mean
        r_scaled = r / R_mean
        # Use max exponent trick to avoid underflow
        exponents = -phi * (r_scaled + ratio)**2
        max_exp = np.max(exponents)
        exp_sum = np.mean(np.exp(exponents - max_exp))
        return exp_sum * np.exp(max_exp)

    # Integrate over scaled distance (0 -> some reasonable multiple of mean radius)
    r_max = 10 * R_mean  # can increase if needed
    ED, _ = quad(integrand, 0, r_max, limit=100)
    return ED
'''
import numpy as np
from scipy.integrate import quad

'''
def mean_distance_to_interface(r_array, phi, r_max_factor=10):
    """
    Compute the mean distance to the nearest interface for a 2D Poisson distribution of disks.

    Parameters
    ----------
    r_array : array-like
        Array of particle radii [m]
    phi : float
        Area fraction (0 < phi < 1)
    r_max_factor : float
        Factor to multiply by mean radius for upper integration limit
    """
    r_array = np.array(r_array)
    R_mean = np.mean(r_array)
    r_max = r_max_factor * R_mean  # integration limit

    # Define the integrand in dimensionless form
    def integrand(r):
        r_scaled = r / R_mean
        return np.mean(np.exp(-phi * (r_scaled + r_array / R_mean)**2))

    # Integrate from 0 to r_max / R_mean
    ED_scaled, _ = quad(integrand, 0, r_max / R_mean)
    
    # Multiply by R_mean to return physical units
    ED = R_mean * ED_scaled
    return ED
'''

import numpy as np
from scipy.integrate import quad

def mean_distance_to_interface2D(r_array, phi):
    """
    Compute the mean distance to the nearest interface for a 2D Poisson distribution
    using the original full integral (non-approximated).

    Parameters
    ----------
    r_array : array-like
        Array of particle radii [m]
    phi : float
        Area fraction (0 < phi < 1)

    Returns
    -------
    ED : float
        Mean distance to the nearest interface [m]
    """
    r_array = np.array(r_array)
    N = len(r_array)

    # Original integral (non-approximated)
    def integrand(r):
        mean_squared_ratio = np.mean((r_array + r)**2) / np.mean(r_array**2)
        return np.exp(-phi * mean_squared_ratio)

    # Integrate from 0 to infinity will crash do 10*R_mean
    ED, _ = quad(integrand, 0, 10*np.mean(r_array), limit=100)
    return ED

def mean_distance_to_interface3D(r_array, phi):
    """
    Compute the mean distance to the nearest interface for a 2D Poisson distribution
    using the original full integral (non-approximated).

    Parameters
    ----------
    r_array : array-like
        Array of particle radii [m]
    phi : float
        Area fraction (0 < phi < 1)

    Returns
    -------
    ED : float
        Mean distance to the nearest interface [m]
    """
    r_array = np.array(r_array)
    N = len(r_array)

    # Original integral (non-approximated)
    def integrand(r):
        mean_squared_ratio = np.mean((r_array + r)**3) / np.mean(r_array**3)
        return np.exp(-phi * mean_squared_ratio)

    # Integrate from 0 to infinity will crash do 10*R_mean
    ED, _ = quad(integrand, 0, 10*np.mean(r_array), limit=100)
    return ED


def mean_distance_poisson_Hs(r_array, phi):
    """
    Compute the mean distance to the nearest disk interface for a 2D Poisson process
    using the classical H_s(r) formula:
    
        H_s(r) = 1 - exp[-lambda * pi * r * (2 E[R] + r)]
    
    only works for monodisperse 
    
    Parameters
    ----------
    r_array : array-like
        Array of disk radii [m]
    lambda_density : float
        Number density of disks (disks per unit area)
    
    Returns
    -------
    ED : float
        Mean distance to nearest interface [m]
    """
    lambda_density = phi / (np.pi * np.mean(r_array**2))
    r_array = np.array(r_array)
    R_mean = np.mean(r_array)
    
    # Hole probability function H_s(r)
    def H_s(r):
        return 1 - np.exp(-lambda_density * np.pi * r * (2 * R_mean + r))
    
    # Integrate H_s(r) from 0 to infinity to get mean distance
    ED, _ = quad(H_s, 0, 10*np.mean(r_array), limit=200)
    return ED




def tester_structures():
    '''Generating random radii/%AP, put into text file, see if it matches with equation'''
    # just going to test with unimodal to start
    radius_m = np.random.uniform(0.000025, 0.0005, 100)
    percent_ap = np.random.uniform(0.45, 0.65, 100)
    results = []
    MoE = []
    factor = []

    for i, AP_target in enumerate(percent_ap):

        R_m = radius_m[i]
        R_um = R_m * 1e6
        physical_size = 1e-2  # m
        temp_png = os.path.join(folderE_png, f"temp_{i}.png")
        temp_xyzr = os.path.join(folderE_xyzr, f"temp_{i}.xyzr")
        r_array = []
        
        
        
        _, _, AP_achieved, _ = gm.gen_struct(
            temp_png, save_path_ignore, temp_xyzr,
            1, physical_size, R_m, AP_target,
            0.8, max_attempts=100000, mode=1, max_tries=1, interface_width=1e-6, dpi_highres=500
        )
        print('Image', i+1, 'created.')

        img = imageio.imread(save_path_ignore)
        img_size = img.shape[0]  # pixels
        dx = physical_size / img_size  # m per pixel
        tolerance = 10
        interface_mask = (
            (np.abs(img[:, :, 0] - 255) <= tolerance) &
            (np.abs(img[:, :, 1] - 255) <= tolerance) &
            (np.abs(img[:, :, 2] - 255) <= tolerance)
        )

        # distance map (experimental)
        interface_mask = skeletonize(interface_mask)  # get to clean line
        distance_px = ndimage.distance_transform_edt(~interface_mask)
        avg_distance = distance_px[~interface_mask].mean() * dx

        


        AP_str = str(int(AP_achieved * 10000))
        radius_um = int(R_m * 1e6)
        final_png = os.path.join(folderE_png, f"uni_AP{AP_str}_R{radius_um}um.png")
        final_xyzr = os.path.join(folderE_xyzr, f"uni_AP{AP_str}_R{radius_um}um.xyzr")
        ap, htpb, interface = lc.vert_avg_fast(save_path_ignore)


        with open(temp_xyzr, "r") as f:
            for line in f:
                parts = line.strip().split()
                r_array.append(float(parts[3]))
        r_array = np.array(r_array)
        
        ED = mean_distance_to_interface2D(r_array, AP_achieved)
        ED3 = mean_distance_to_interface3D(r_array, AP_achieved)
        ED1 = mean_distance_poisson_Hs(r_array, AP_achieved)
        #factor_val = abs((avg_distance - ED) / avg_distance * 100)
        error2 = abs(avg_distance - ED)/avg_distance * 100
        error3 = abs(avg_distance - ED3)/avg_distance * 100
        error1 = abs(avg_distance - ED1)/avg_distance * 100
        # store results
        results.append((AP_achieved * 100, radius_um, avg_distance, ED, error2, ED3, error3))

        if os.path.exists(temp_png):
            os.replace(temp_png, final_png)
        if os.path.exists(temp_xyzr):
            os.replace(temp_xyzr, final_xyzr)
        else:
            print(f"Warning: {temp_xyzr} was not created!")

    # write results to file
    with open(resultsE, "w") as f:
        f.write("AP_achieved, Avg Radius (um), Avg Distance Experimental, Avg 2D Approx, 2D Error, Avg 3D Approx, 3D Error\n")
        for item in results:
            f.write(f"{item[0]:10.4f}, {item[1]:6d}, {item[2]:12.6e}, {item[3]:12.6e}, {item[4]:10.4f}, {item[5]:12.6e}, {item[6]:10.4f}\n")

    print(np.mean([x[4] for x in results]))

def read_resultsE():
    with open(resultsE, "r") as f:
        exp = []
        eq = []
        for line in f:
            if line.startswith("AP_achieved"):
                continue
            parts = line.strip().split(',')
            exp.append(float(parts[2]))
            eq.append(float(parts[3]))
    exp_array = np.array(exp)
    eq_array = np.array(eq)
    print(np.mean(exp_array) - np.mean(eq_array))



if __name__ == "__main__":
    #reset()
    #generateA()
    #generateB()
    #tester_structures()
    read_resultsE()
