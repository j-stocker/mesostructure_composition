# mesostructure_composition
Generate an AP/HTPB mesostructure assuming perfectly circular AP particles, given mean radius, standard deviation (normal distribution), and % composition. Analyze lines within the 2D image to draw relationships between those parameters and the probability of finding "interface".


in samples folders: 
for varying AP ratios:
the filenames are in the format of uni_AP4908_R50, which correlates to a unimodal distribution, AP ratio of 49.08%, and average particle radius of 50 um.
for varying radii:
the filenames are in the format of bi_R450um_AP5034, which correlates to a bimodal distribution, average coarse particle radius of 45.0 um, and AP ratio of 50.34%. For a unimodal distribution, R450 would correspond to simply an overall average radius of the same value.


This script works by taking in the average radius size (overall, or for coarse and fine particles separately), the % vol AP, and the type of distribution. It randomly generates a set of radii to match the standard deviation and radius size. It then sorts the radii and places them largest to smallest at randomly generated coordinates. To avoid overlap, the image domain is sorted into cells slightly larger than the size of the average radius. Before accepting the placement, each of the cells surrounding that of the generated coordinate are checked for overlap. If there is overlap, that location is skipped. Because circles may be placed outside the boundary of the image domain, the total AP % area is computed by counting the pixels within the domain as a portion of the whole image. This process continues until the target area is reached (within a given tolerance) or the system gives up. For each set of parameters, several attempts will be made and the arrangement closest to the target area is selected.

Currently comparing Poisson geometry predictions for packed disks to skeletonized images to find average distance from a random point to the nearest interface.
