'''Generate:
    - 50 unimodal distributions with approximately the same volume % of AP but varying particle sizes
    - 50 unimodal distributions with varying volume % of AP but the same particle sizes
    - 50 bimodal distributions with approximately the same volume % of AP but varying particle sizes
    - 50 bimodal distributions with varying volume % of AP but the same particle sizes'''
import os
import gen_meso as gm
import numpy as np
import glob
#clear folders to generate new images


def clear_folder(folder_path):
    # Get all files in the folder
    files = glob.glob(os.path.join(folder_path, "*"))
    for f in files:
        try:
            os.remove(f)  # delete file
        except Exception as e:
            print(f"Could not delete {f}: {e}")




# Parameters
std_dev_uni = 0.2
AP_vol_50A = 0.5
avg_rad_50A = np.linspace(30e-6, 80e-6, 50)

avg_rad_50B = 50e-6  # fixed particle size
AP_vol_50B = np.linspace(0.4, 0.6, 50)  # varying AP

std_dev_bi = [0.2, 0.5]
mean_rad_bi_size = [np.linspace(80e-6, 130e-6, 50), [27e-6]*50]  # varying size, coarse/fine
AP_fixed_bi = 0.5

mean_rad_bi_AP = [125e-6, 15e-6]  # fixed size, coarse/fine
AP_bi_var = np.linspace(0.4, 0.6, 50)  # varying AP
mix_bi = 1/10 #1:10 coarse:fine

img_size = 1
physical_size = 5e-4
max_attempts = 5000

folderA_png = "./generated_images/unimodal_varying_part_size/pngs"
folderA_xyzr = "./generated_images/unimodal_varying_part_size/xyzrs"
folderB_png = "./generated_images/unimodal_varying_AP_ratio/pngs"
folderB_xyzr = "./generated_images/unimodal_varying_AP_ratio/xyzrs"
folderC_png = "./generated_images/bimodal_varying_part_size/pngs"
folderC_xyzr = "./generated_images/bimodal_varying_part_size/xyzrs"
folderD_png = "./generated_images/bimodal_varying_AP_ratio/pngs"
folderD_xyzr = "./generated_images/bimodal_varying_AP_ratio/xyzrs"
folder_ignore = "./generated_images/ignore"

clear_folder(folderA_png)
clear_folder(folderA_xyzr)
clear_folder(folderB_png)
clear_folder(folderB_xyzr)
clear_folder(folderC_png)
clear_folder(folderC_xyzr)
clear_folder(folderD_png)
clear_folder(folderD_xyzr)

os.makedirs(folderA_png, exist_ok=True)
os.makedirs(folderA_xyzr, exist_ok=True)
os.makedirs(folderB_png, exist_ok=True)
os.makedirs(folderB_xyzr, exist_ok=True)
os.makedirs(folderC_png, exist_ok=True)
os.makedirs(folderC_xyzr, exist_ok=True)
os.makedirs(folderD_png, exist_ok=True)
os.makedirs(folderD_xyzr, exist_ok=True)
os.makedirs(folder_ignore, exist_ok=True)

save_path_ignore = os.path.join(folder_ignore, "ignore.png")

def generate():
    
    # --- A: Unimodal, varying particle size ---
    for i, radius in enumerate(avg_rad_50A):
        temp_png = os.path.join(folderA_png, f"temp_{i}.png")
        temp_xyzr = os.path.join(folderA_xyzr, f"temp_{i}.xyzr")

        _, _, AP_achieved, _ = gm.gen_struct(
            temp_png, save_path_ignore, temp_xyzr,
            img_size, physical_size, radius, AP_vol_50A,
            std_dev_uni, max_attempts, mode=1
        )

        AP_str = str(int(AP_achieved * 10000))  # 4-digit AP fraction without leading 0
        radius_um = int(radius * 1e6)
        final_png = os.path.join(folderA_png, f"uni_AP{AP_str}_R{radius_um}um.png")
        final_xyzr = os.path.join(folderA_xyzr, f"uni_AP{AP_str}_R{radius_um}um.xyzr")

        if os.path.exists(temp_png):
            os.replace(temp_png, final_png)
        if os.path.exists(temp_xyzr):
            os.replace(temp_xyzr, final_xyzr)
        else:
            print(f"Warning: {temp_xyzr} was not created!")

    # --- B: Unimodal, varying AP fraction ---
    for i, AP_target in enumerate(AP_vol_50B):
        temp_png = os.path.join(folderB_png, f"temp_{i}.png")
        temp_xyzr = os.path.join(folderB_xyzr, f"temp_{i}.xyzr")

        _, _, AP_achieved, _ = gm.gen_struct(
            temp_png, save_path_ignore, temp_xyzr,
            img_size, physical_size, avg_rad_50B, AP_target,
            std_dev_uni, max_attempts, mode=1
        )

        AP_str = str(int(AP_achieved * 10000))
        radius_um = int(avg_rad_50B * 1e6)
        final_png = os.path.join(folderB_png, f"uni_AP{AP_str}_R{radius_um}um.png")
        final_xyzr = os.path.join(folderB_xyzr, f"uni_AP{AP_str}_R{radius_um}um.xyzr")

        if os.path.exists(temp_png):
            os.replace(temp_png, final_png)
        if os.path.exists(temp_xyzr):
            os.replace(temp_xyzr, final_xyzr)
        else:
            print(f"Warning: {temp_xyzr} was not created!")

    # --- C: Bimodal, varying particle size ---
    for i, coarse_radius in enumerate(mean_rad_bi_size[0]):
        fine_radius = mean_rad_bi_size[1][i]  # pick corresponding fine particle
        radius_input = [float(coarse_radius), float(fine_radius)]  # pass as list for bimodal

        temp_png = os.path.join(folderC_png, f"temp_{i}.png")
        temp_xyzr = os.path.join(folderC_xyzr, f"temp_{i}.xyzr")

        _, _, AP_achieved, _ = gm.gen_struct(
            temp_png, save_path_ignore, temp_xyzr,
            img_size, physical_size, radius_input, AP_fixed_bi,
            std_dev_bi, max_attempts, mode=2, mix=mix_bi
        )

        AP_str = str(int(AP_achieved * 10000))  # 4-digit AP fraction
        radius_um = int(coarse_radius * 1e6)
        final_png = os.path.join(folderC_png, f"bi_AP{AP_str}_R{radius_um}um.png")
        final_xyzr = os.path.join(folderC_xyzr, f"bi_AP{AP_str}_R{radius_um}um.xyzr")

        if os.path.exists(temp_png):
            os.replace(temp_png, final_png)
        if os.path.exists(temp_xyzr):
            os.replace(temp_xyzr, final_xyzr)
        else:
            print(f"Warning: {temp_xyzr} was not created!")


    # --- D: Bimodal, varying AP fraction ---
    coarse_radius = mean_rad_bi_AP[0]
    fine_radius = mean_rad_bi_AP[1]
    radius_input = [coarse_radius, fine_radius]

    for i, AP_target in enumerate(AP_bi_var):
        temp_png = os.path.join(folderD_png, f"temp_{i}.png")
        temp_xyzr = os.path.join(folderD_xyzr, f"temp_{i}.xyzr")

        _, _, AP_achieved, _ = gm.gen_struct(
            temp_png, save_path_ignore, temp_xyzr,
            img_size, physical_size, radius_input, AP_target,
            std_dev_bi, max_attempts, mode=2, mix=mix_bi
        )

        AP_str = str(int(AP_achieved * 10000))  # 4-digit AP fraction
        radius_um = int(coarse_radius * 1e6)
        final_png = os.path.join(folderD_png, f"bi_AP{AP_str}_R{radius_um}um.png")
        final_xyzr = os.path.join(folderD_xyzr, f"bi_AP{AP_str}_R{radius_um}um.xyzr")

        if os.path.exists(temp_png):
            os.replace(temp_png, final_png)
        if os.path.exists(temp_xyzr):
            os.replace(temp_xyzr, final_xyzr)
        else:
            print(f"Warning: {temp_xyzr} was not created!")


if __name__ == "__main__":
    generate()
    print('Success!')
