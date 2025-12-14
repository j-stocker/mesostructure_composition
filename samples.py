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
    clear_folder(folderA_png)
    clear_folder(folderA_xyzr)

    clear_folder(folderB_png)
    clear_folder(folderB_xyzr)

    #clear_folder(folderC_png)
    #clear_folder(folderC_xyzr)

    #clear_folder(folderD_png)
    #clear_folder(folderD_xyzr)
    
    #clear_folder(folderE_png)
    #clear_folder(folderE_xyzr)


    os.makedirs(folderA_png, exist_ok=True)
    os.makedirs(folderA_xyzr, exist_ok=True)

    os.makedirs(folderB_png, exist_ok=True)
    os.makedirs(folderB_xyzr, exist_ok=True)

    #os.makedirs(folderC_png, exist_ok=True)
    #os.makedirs(folderC_xyzr, exist_ok=True)

    #os.makedirs(folderD_png, exist_ok=True)
    #os.makedirs(folderD_xyzr, exist_ok=True)
    
    #os.makedirs(folderE_png, exist_ok=True)
    #os.makedirs(folderE_xyzr, exist_ok=True)

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



def tester_structures():
    '''Generating random radii/%AP, put into text file, see if it matches with equation'''
    #just going to test with unimodal to start
    avg_rad_set = np.random.uniform(40e-6, 140e-6, 1)
    percent_ap = np.random.uniform(0.4, 0.6, 1)
    results = []
    MoE = []

    for i, AP_target in enumerate(percent_ap):
        temp_png = os.path.join(folderE_png, f"temp_{i}.png")
        temp_xyzr = os.path.join(folderE_xyzr, f"temp_{i}.xyzr")

        _, _, AP_achieved, _ = gm.gen_struct(
            temp_png, save_path_ignore, temp_xyzr,
            1, 5e-3, avg_rad_set[i], AP_target,
            std_dev_uni, max_attempts, mode=1, max_tries=1
        )

        AP_str = str(int(AP_achieved * 10000))
        radius_um = int(avg_rad_set[i] * 1e6)
        final_png = os.path.join(folderE_png, f"uni_AP{AP_str}_R{radius_um}um.png")
        final_xyzr = os.path.join(folderE_xyzr, f"uni_AP{AP_str}_R{radius_um}um.xyzr")
        ap, htpb, interface = lc.vert_avg_fast(save_path_ignore)
        #I equation
        # --- compute effective radius from xyzr ---
        radii = []
        with open(temp_xyzr, "r") as f:
            for line in f:
                r = float(line.strip().split()[3])  # meters
                radii.append(r)

        radii = np.array(radii)

        # effective radius that matches σ = 0.4 fit
        R_eff = np.mean(radii**2) / np.mean(radii)   # meters
        R_eff_um = R_eff * 1e6                        # microns

        I_theory = 6.238e-4*R_eff_um**2 - 0.1152*R_eff_um + 10.71*AP_achieved**2 - 2.851*AP_achieved + 6.422 
        #I_theory = 7.663e-3*radius_um**2 - 0.8833*radius_um + 25.56*AP_achieved**2 - 4.616*AP_achieved + 18.627
        #
        #I_theory = (2*I_theory1 + I_theory2)/3
        
        #equation's margin of error
        MoE.append(np.abs(I_theory - interface)/interface * 100)
        
        results.append((AP_achieved*100, R_eff_um, interface, I_theory, MoE[i]))
        if os.path.exists(temp_png):
            os.replace(temp_png, final_png)
        if os.path.exists(temp_xyzr):
            os.replace(temp_xyzr, final_xyzr)
        else:
            print(f"Warning: {temp_xyzr} was not created!")
        #put something with equation in here too...
    with open(resultsE, "w") as f:
        f.write("AP_achieved, Avg Radius (um), Interface Experimental, Interface from Eq., Eq. Margin of Error\n")  # optional header
        for item in results:
            f.write(f"{item[0]}, {item[1]}, {item[2]}, {item[3]}, {item[4]}\n")
    print(np.mean(MoE))
    
    



if __name__ == "__main__":
    reset()
    generateA()
    generateB()
