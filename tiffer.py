from __future__ import division
from future.builtins import input
from skimage import io as skio
from modules.vpipes import mirror_finder, pgm_to_tiff, zipper, tiff_maker
import numpy as np
import glob, os, re

#*********************************************************************************************#

pgm_list = []
while pgm_list == []: ##Keep repeating until pgm files are found
    iris_path = input("\nPlease type in the path to the folder that contains the IRIS data:\n")
    os.chdir(iris_path)
    pgm_list = sorted(glob.glob('*.pgm'))

tiff_list = tiff_maker(pgm_list, tiff_compression = 1, archive = False)

print(tiff_list)

# chip_name = pgm_list[0].split(".")[0]
#
# pgm_list, mirror = mirror_finder(pgm_list)
# pgm_set = set([".".join(file.split(".")[:3]) for file in pgm_list])
#
# spot_counter = max([int(pgmfile.split(".")[1]) for pgmfile in pgm_list])##Important
# zslice_count = max([int(pgmfile.split(".")[3]) for pgmfile in pgm_list])
#
# print("There are {} PGM images, in stacks of {}.\n".format(len(pgm_list), zslice_count))
#
# pass_counter = int(max([pgm.split(".")[2] for pgm in pgm_list]))##Important
#
# spot_to_scan = 1
#
# while spot_to_scan <= spot_counter:
#
#     pps_list = sorted([file for file in pgm_set if int(file.split(".")[1]) == spot_to_scan])
#     passes_per_spot = len(pps_list)
#
#     if (passes_per_spot != pass_counter):
#         print("Missing pgm files... \n")
#
#     for scan in range(0,passes_per_spot,1):
#
#         stack_list = [file for file in pgm_list if file.startswith(pps_list[scan])]
#
#         fluor_files = [file for file in stack_list if file.split(".")[-2] in 'ABC']
#         if fluor_files:
#             stack_list = [file for file in stack_list if file not in fluor_files]
#             print("Fluorescent channel(s) detected: {}\n".format(fluor_files))
#
#         scan_collection = skio.imread_collection(stack_list)
#         pgm_name = ".".join(stack_list[0].split(".")[:3])
#
#         pic3D = np.array([pic for pic in scan_collection], dtype='uint16')
#         # if mirror_correction == True:
#         #     pic3D = (pic3D / mirror).astype('float32')
#
#         pgm_to_tiff(pic3D, pgm_name, stack_list, tiff_compression = 1, archive_pgm=False)
#
#     spot_to_scan += 1
# if mirror_correction == True:
#     mirror_file = str(glob.glob('*000.pgm')).strip("'[]'")
#     zipper('mirror', mirror_file, compression='bz2', iris_path=os.getcwd())
#     os.remove(mirror_file)
