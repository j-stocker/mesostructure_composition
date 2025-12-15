#generating structures but multithreaded
#vibe coded...

import os
import gen_meso as gm
import numpy as np
import glob
import line_comp as lc
import gen_plots as gp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

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
avg_rad_50A = np.linspace(20e-6, 140e-6, 200)

avg_rad_50B = 70e-6  # fixed particle size
AP_vol_50B = np.linspace(0.33, 0.63, 200)  # varying AP

img_size = 1
physical_size = 1e-2
max_attempts = 100000

interface_thickness = 2e-6 #2 um

folderA_png = "./generated_images/unimodal_varying_part_size/pngs"
folderA_xyzr = "./generated_images/unimodal_varying_part_size/xyzrs"

folderB_png = "./generated_images/unimodal_varying_AP_ratio/pngs"
folderB_xyzr = "./generated_images/unimodal_varying_AP_ratio/xyzrs"


folder_ignore = "./generated_images/ignore"

resultsA = "./generated_images/unimodal_varying_part_size/results_uni_part_size.txt"
resultsB = "./generated_images/unimodal_varying_AP_ratio/results_uni_AP_ratio.txt"

def reset():
    clear_folder(folderA_png)
    clear_folder(folderA_xyzr)

    clear_folder(folderB_png)
    clear_folder(folderB_xyzr)



    os.makedirs(folderA_png, exist_ok=True)
    os.makedirs(folderA_xyzr, exist_ok=True)

    os.makedirs(folderB_png, exist_ok=True)
    os.makedirs(folderB_xyzr, exist_ok=True)

    os.makedirs(folder_ignore, exist_ok=True)


    for fp in [resultsA, resultsB]:
        os.makedirs(os.path.dirname(fp), exist_ok=True)



save_path_ignore = os.path.join(folder_ignore, "ignore.png")


def _task_generateA(i, radius):
    label = f"A | idx={i} | R={radius*1e6:.2f} µm"
    
    temp_png = os.path.join(folderA_png, f"temp_{i}.png")
    temp_xyzr = os.path.join(folderA_xyzr, f"temp_{i}.xyzr")

    _, _, AP_achieved, _ = gm.gen_struct(
        save_path_ignore, temp_png, temp_xyzr,
        img_size, physical_size, radius, AP_vol_50A,
        std_dev_uni, max_attempts, mode=1, max_tries=20
    )

    AP_str = str(int(AP_achieved * 10000))
    radius_um = int(radius * 1e7)

    final_png = os.path.join(folderA_png, f"uni_R{radius_um}um_AP{AP_str}.png")
    final_xyzr = os.path.join(folderA_xyzr, f"uni_R{radius_um}um_AP{AP_str}.xyzr")

    ap, htpb, interface = lc.vert_avg_fast(save_path_ignore)

    if os.path.exists(temp_png):
        os.replace(temp_png, final_png)
    if os.path.exists(temp_xyzr):
        os.replace(temp_xyzr, final_xyzr)

    return label, (radius_um, ap, htpb, interface)


def _task_generateB(i, AP_target):
    label = f"B | idx={i} | AP={AP_target:.3f}"
    temp_png = os.path.join(folderB_png, f"temp_{i}.png")
    temp_xyzr = os.path.join(folderB_xyzr, f"temp_{i}.xyzr")

    _, _, AP_achieved, _ = gm.gen_struct(
        save_path_ignore, temp_png, temp_xyzr,
        img_size, physical_size, avg_rad_50B, AP_target,
        std_dev_uni, max_attempts, mode=1, max_tries=5
    )

    AP_str = str(int(AP_achieved * 10000))
    radius_um = int(avg_rad_50B * 1e6)

    final_png = os.path.join(folderB_png, f"uni_AP{AP_str}_R{radius_um}um.png")
    final_xyzr = os.path.join(folderB_xyzr, f"uni_AP{AP_str}_R{radius_um}um.xyzr")

    ap, htpb, interface = lc.vert_avg_fast(save_path_ignore)

    if os.path.exists(temp_png):
        os.replace(temp_png, final_png)
    if os.path.exists(temp_xyzr):
        os.replace(temp_xyzr, final_xyzr)

    return label, (AP_achieved * 100, ap, htpb, interface)


from concurrent.futures import ProcessPoolExecutor, as_completed

def generateA():
    results = []

    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(_task_generateA, i, r): i
            for i, r in enumerate(avg_rad_50A)
        }

        with tqdm(total=len(futures), desc="Generating A images") as pbar:
            for future in as_completed(futures):
                label, result = future.result()
                tqdm.write(f"✔ Finished {label}")
                results.append(result)
                pbar.update(1)

    with open(resultsA, "w") as f:
        f.write("radius in um, AP_vert, HTPB_vert, Interface_vert\n")
        for r in sorted(results, key=lambda x: x[0]):
            f.write(f"{r[0]}, {r[1]}, {r[2]}, {r[3]}\n")
def generateB():
    results = []

    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(_task_generateB, i, ap): i
            for i, ap in enumerate(AP_vol_50B)
        }

        with tqdm(total=len(futures), desc="Generating B images") as pbar:
            for future in as_completed(futures):
                label, result = future.result()
                tqdm.write(f"✔ Finished {label}")
                results.append(result)
                pbar.update(1)

    with open(resultsB, "w") as f:
        f.write("AP_achieved, AP_vert, HTPB_vert, Interface_vert\n")
        for r in sorted(results, key=lambda x: x[0]):
            f.write(f"{r[0]}, {r[1]}, {r[2]}, {r[3]}\n")


if __name__ == "__main__":
    reset()
    generateA()
    generateB()


