#!/usr/bin/env python3
from __future__ import division
import glob
import os
import sys
import warnings
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as smapi

import cv2

from future.builtins import input
from datetime import datetime
from skimage import io as skio
from skimage.external.tifffile import TiffFile
from skimage.morphology import disk
from skimage.filters import sobel_h, sobel_v, rank, threshold_triangle, threshold_li, try_all_threshold
from skimage.feature import shape_index, canny, register_translation
from skimage.exposure import rescale_intensity
from skimage.measure import perimeter
from scipy import stats
from scipy.ndimage import gaussian_filter, find_objects
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation
from scipy.spatial import cKDTree
from math import isnan

from cv2 import normalize, NORM_MINMAX

from modules import vpipes, vimage, vquant, vgraph, vfilo, vmulti
# from modules.filographs import filohisto
from images import logo
#
# from modules.brief import measure_shift_brief, measure_shift_orb


pd.set_option('display.width', 1000)
pd.options.display.max_rows=999
pd.options.display.max_columns=15
logo.print_logo()
version = '3.1.1'
"""
Version 3.1.1 = Various optimizations to make use of already collected data
Version 3.1.0 = Added code to measure the overlay of fluorescent and visible particles. Turned up
                sensitivity of particle finding slightly
Version 3.0.2 = Switched to OpenCV algo for HoughCircle to find Ab spot. Analysis time is now nearly
                halved relative to Version 2.X.X
Version 3.0.1 = Using OpenCV for matching descriptors in prescan/postscan images with ORB algo.
Version 3.0.0 = Started using OpenCV algorithms to speed up processing time (written in C++).
                First replaced scikit-image equalize_adapthist with cv2.createCLAHE
Version 2 Notes:
    Version 2.14.3 = Switched back to using shape index to find fluorescent particles
    Version 2.14.2 = Using triangle thresholding algorithm to calculate fluorescent pixels
    Version 2.14.1 = Calculating median spot background using only pixels that are not a part of particles
                     rather than all pixels in the counting area
    Version 2.14.0 = Now measuring fluorscent partilces rather than mean fluorescence
    Version 2.13.1 = Begun measuring mean fluorescence of spots for channels if those images are present
    Version 2.13.0 = Re-wrote filament measuring algo to get correct values for curved objects
    Version 2.12.1 = Started removing particles that dont correlate well to percent_contrastXintensity line
    Version 2.12.0 = Added simple algorithm for removing invalid particles based on defocus curve
    Version 2.11.3 = Bugfix for CLAHE algorithm that was adding horizontal line artifact
    Version 2.11.2 = Added function to prevent counting of halo artifacts of bright particles
    Version 2.11.1 = Removed corr function for now. Fixed bug in percent contrast calculation.
    Version 2.11.0 = Added a correlation function to better select pixels that are nanoparticles.
                      This increases analysis time. Only works with large-stack data (>21 images)
    Version 2.10.8 = Modified thresholds for shape finding; narrowed the window of the maxmin projection
    Version 2.10.7 = Removed for loops when doing data extraction from regionprops.
    Version 2.10.6 = changed how focus is caluclated using the Tenengrad algorithm
    Version 2.10.5 = Increased CLAHE clip limit from .002 to .008 for Exoviewer images to improve quality
    Version 2.10.4 = Added TIFF compatibility
    Version 2.10.3 = New template for Exoviewer markers
    Version 2.10.2 = Lowered CV cutoff for in-liquid experiments to 0.01
    Version 2.10.1 = Fixed histgrams so that values are normalized to counting area
    Version 2.10 = Added filament length measurements. Still in beta.
    Version 2.9 = Exoviewer & fluorescence functionality
    Version 2.8 = Using mirror normalized image rather than original image for calculations
    Version 2.6 = Using mask to remove old particles instead of relying on subtractions
"""
print("VERSION {}\n".format(version))
print(os.path.dirname(__file__))

def get_dupe_index(removal_index, l1, i1):

    seen = set()
    # adds all elements it doesn't know yet to seen and all other to seen_twice
    dupes = set(x for x in l1 if x in seen or seen.add(x))
        #return indices of all occurences of duplicates
    dupes = list(dupes)

    if dupes == []:
        return []
    else:
        for dupe_val in dupes:
            # print(dupe_val)
            dupe_indices = [i for i, val in enumerate(l1) if val == dupe_val]
            # print(dupe_indices)
            dupe_ious = [i1[ix] for ix in dupe_indices]
            # print(dupe_ious)
            dupe_to_remove = dupe_indices[dupe_ious.index(min(dupe_ious))]
            # print(dupe_to_remove)
            removal_index.append(dupe_to_remove)
            # print(removal_index)
        return removal_index

def remove_dupes_by_iou(overlap_ix_list):
    v1,f1,i1 = [],[],[]
    removal_index = []
    for n, tup in enumerate(overlap_ix_list):
        v1.append(tup[0])
        f1.append(tup[1])
        i1.append(tup[2])

    removal_index = get_dupe_index(removal_index, f1,i1)

    removal_index = get_dupe_index(removal_index, v1,i1)

    removal_list = [overlap_ix_list[ix] for ix in removal_index]

    return [val for val in overlap_ix_list if val not in removal_list]
#*********************************************************************************************#
#
#    CODE BEGINS HERE
#
#*********************************************************************************************#
##Quick-change Boolean Parameters
parser = argparse.ArgumentParser()

parser.add_argument("--foo", help="foo Help")

args = parser.parse_args()

print(sys.argv)

show_particles = True ##show particle info on output images
show_filos = False
# remove_overlapping_objs = True
Ab_spot_mode = True ##change to False if chips have not been spotted with antibody microarray
amab = 'N/A'#'AF568-'+'D'r'$\alpha$''G'
cmab = 'N/A'#'AF647-'+'D'r'$\alpha$''R'

#*********************************************************************************************#
IRISmarker_liq = skio.imread('images/IRISmarker_new.tif')
IRISmarker_exo = skio.imread('images/IRISmarker_v4_topstack.tif')

pgm_list, tiff_list = [],[]
marker_dict = {}
while (pgm_list == []) and (tiff_list == []): ##Keep repeating until pgm files are found
    iris_path = input("\nPlease type in the path to the folder that contains the IRIS data:\n")
    if iris_path == 'test':
        iris_path = '/Volumes/KatahdinHD/ResilioSync/DATA/IRIS/FIGS4PAPER/expts/tCHIP007_EBOVmay@1E6'
    else:
        iris_path = iris_path.strip('"')##Point to the correct directory
    os.chdir(iris_path)
    pgm_list = sorted(glob.glob('*.pgm'))
    tiff_list = sorted(glob.glob('*.tif'))

pgm_list, mirror = vpipes.mirror_finder(pgm_list)

fluor_files = [file for file in pgm_list if file.split(".")[-2] in 'ABC']

if tiff_list == []:
    tiff_toggle = False
    image_list = pgm_list
    convert_tiff = input("Convert PGM stacks to TIFFs (y/n)?")
    while convert_tiff.lower() not in ['yes', 'y', 'no', 'n']:
        convert_tiff = input("Convert PGM stacks to TIFFs (y/n)?")
    if convert_tiff in ['yes', 'y']:
        convert_tiff = True
    else:
        convert_tiff = False

elif pgm_list == []:
    tiff_toggle = True
    covert_tiff = False
    image_list = tiff_list

elif pgm_list == fluor_files:
    print("All remaining PGMs are fluorescent data. Not converting")
    tiff_toggle = True
    convert_tiff = False
    image_list = tiff_list + fluor_files

else:
    print("Mixture of PGM and TIFF files\n")#. Please convert all PGM files before continuing")
    image_list = sorted(set(tiff_list + pgm_list) - set(fluor_files))

    convert_tiff = input("Convert PGM stacks to TIFFs (y/n)?")
    while convert_tiff.lower() not in ['yes', 'y', 'no', 'n']:
        convert_tiff = input("Convert PGM stacks to TIFFs (y/n)?")
    if convert_tiff in ['yes', 'y']:
        convert_tiff = True
    else:
        convert_tiff = False
    # sys.exit()


txt_list = sorted(glob.glob('*.txt'))
xml_list = sorted(glob.glob('*/*.xml'))
if not xml_list: xml_list = sorted(glob.glob('../*/*.xml'))

chip_name = image_list[0].split(".")[0]
if chip_name == 'baseCHIP404':
    Ab_spot_mode = False
image_set = set(".".join(file.split(".")[:3]) for file in image_list)

txtcheck = [file.split(".") for file in txt_list]
iris_txt = [".".join(file) for file in txtcheck if (len(file) >= 3) and (file[2].isalpha())]

params_dict = {}
xml_file = [file for file in xml_list if chip_name in file]
if xml_file:
    chipFile = vpipes.xml_parser(xml_file[0])
    meta_dict = chipFile[0]
    if meta_dict['chipversion'] == '4':
        print("Exoviewer images\n")
        if Ab_spot_mode == True:
            cliplim = 0.004
            perc_range = (3,97)
            cv_cutoff = 0.5
        else:
            cliplim = 0.008
            perc_range = (3,97)
            cv_cutoff = 0.75

        # kernel_size = (270,404)
        params_dict['cam_micron_per_pix'] = 3.45 * 2
        params_dict['mag'] = 44
        exo_toggle = True
        # r2_cutoff = 0.85
        IRISmarker = IRISmarker_exo
        # canny_sig = 20
        # NA = 0.75
        timeseries_mode = 'scan'
        filohist_range = (0,150)

    elif meta_dict['chipversion'] == '5':
        print("In-liquid chip\n")
        cliplim = 2
        # kernel_size = (270,404)
        perc_range = (3,97)
        params_dict['cam_micron_per_pix'] = 5.86
        params_dict['mag'] = 40
        exo_toggle = False
        cv_cutoff = 0.01
        IRISmarker = IRISmarker_liq
        # canny_sig = 2
        # NA = 0.75
        filohist_range = (0,10)
    else:
        raise ValueError("Unknown IRIS chip version\n")

    # marker_h, marker_w =  IRISmarker.shape

    mAb_dict = vpipes.chipFile_reader(chipFile, remove_jargon = True)
    spot_counter = len([key for key in mAb_dict])##Important

else:
    print("\nNon-standard chip\n")
    raise ValueError('Cannot find IRIS chip file')
    params_dict['cam_micron_per_pix'] = 3.45
    params_dict['mag'] = 44
    exo_toggle = True
    cv_cutoff = 0.1
    IRISmarker = IRISmarker_exo
    cliplim = 0.004
    timeseries_mode = 'scan'
    spot_counter = max([int(pgmfile.split(".")[1]) for imgfile in image_list])##Important

    mAb_dict = {i:i for i in range(1,spot_counter+1)}
    # mAb_dict_rev = {1: list(range(1,spot_counter+1))}

if Ab_spot_mode == False:
    print("This chip has no antibody spots. There are {} locations\n".format(spot_counter))
else:
    print("There are {} antibody spots\n".format(spot_counter))

cam_micron_per_pix = params_dict['cam_micron_per_pix']
mag = params_dict['mag']
pix_per_um = mag / cam_micron_per_pix
spacing = 1 / pix_per_um
conv_factor = (cam_micron_per_pix / mag)**2

sample_name = vpipes.sample_namer(iris_path)

virago_dir = '{}/v3-analysis'.format(iris_path)
vcount_dir = '{}/vcounts'.format(virago_dir)
img_dir = '{}/processed_images'.format(virago_dir)
histo_dir = '{}/histograms'.format(virago_dir)
overlay_dir = '{}/overlays'.format(virago_dir)
filo_dir = '{}/filo'.format(virago_dir)
fluor_dir = '{}/fluor'.format(virago_dir)

if not os.path.exists(virago_dir):
    os.makedirs(virago_dir)
    made_dir = True
if not os.path.exists(img_dir): os.makedirs(img_dir)
if not os.path.exists(histo_dir): os.makedirs(histo_dir)
if not os.path.exists(fluor_dir): os.makedirs(fluor_dir)
if not os.path.exists(filo_dir): os.makedirs(filo_dir)
if not os.path.exists(overlay_dir): os.makedirs(overlay_dir)

if not os.path.exists(vcount_dir):
    os.makedirs(vcount_dir)
else:
    os.chdir(vcount_dir)
    vdata_list = sorted(glob.glob(chip_name +'*.vdata.txt'))
    made_dir = False

#*********************************************************************************************#
# Text file Parser
#*********************************************************************************************#
os.chdir(iris_path)

spot_list = [int(file[1]) for file in txtcheck if (len(file) > 2) and (file[2].isalpha())]

pass_counter = int(max([imgfile.split(".")[2] for imgfile in image_list]))##Important
if pass_counter == 2:
    timeseries_mode = 'scan'
    print("There are {} passes per image\n".format(pass_counter))
elif pass_counter > 3:
    timeseries_mode = 'time'
else:
    timeseries_mode = 'scan'


scanned_spots = set(np.arange(1,spot_counter+1,1))
missing_spots = tuple(scanned_spots.difference(spot_list))
for val in missing_spots:
    iris_txt.insert(val-1,val)

spot_df, expt_date = vpipes.iris_txt_reader(iris_txt, mAb_dict, pass_counter)



#*********************************************************************************************#
# Image Scanning
spot_to_scan = 1
#*********************************************************************************************#
if (image_set != set()):

    if made_dir == True: image_toggle = 'yes'
    else: image_toggle = ''

    toggle_list = [str(i) for i in spot_list]
    toggle_list.extend(['yes', 'y', 'no', 'n'])
    while image_toggle not in (toggle_list):
        image_toggle = input("\nImage files detected. Do you want scan them for particles? (y/n)\n"
                           + "WARNING: This will take a long time!\n")
else:
    image_toggle = 'no'

if image_toggle.lower() not in ('no', 'n'):
    if image_toggle.isdigit():
        spot_to_scan = int(image_toggle)
    startTime = datetime.now()

    circle_dict, marker_dict, overlay_dict, shift_dict = {},{},{},{}

    while spot_to_scan <= spot_counter:

        vdata_dict = vpipes.get_vdata_dict(exo_toggle, version)

        pps_list = sorted([file for file in image_list if int(file.split(".")[1]) == spot_to_scan])
        fluor_files = [file for file in pps_list if file.split(".")[-2] in 'ABC']
        if fluor_files:
            pps_list = tuple(file for file in pps_list if file not in fluor_files)
            print("\nFluorescent channel(s) detected: {}\n".format(fluor_files))


        passes_per_spot = len(pps_list)
        spot_ID = pps_list[0][:-8]
        scans_counted = {int(file.split(".")[2]) for file in pps_list}

        if (passes_per_spot != pass_counter):
            print("Missing pgm files... fixing...\n")

            scan_set = set(range(1,pass_counter+1))

            missing_csvs = scan_set.difference(scans_counted)

            for scan in missing_csvs:
                bad_scan = '{0}.{1}'.format(*(spot_ID, str(scan).zfill(3)))
                vdata_dict.update({'img_name':bad_scan})

                with open('{}/{}.vdata.txt'.format(vcount_dir,bad_scan),'w') as f:
                    for k,v in vdata_dict.items():
                        f.write('{}: {}\n'.format(k,v))
                print("Writing blank data files for {}".format(bad_scan))
                keep_data = []

        total_shape_df = pd.DataFrame()










        ##Main Loop for image processing begins here.
        for scan in range(0,passes_per_spot,1):
            first_scan = min(scans_counted)
            img_stack = tuple(file for file in image_list if file.startswith(pps_list[scan]))
            # img_stack = tuple(file for file in image_list if int(file.split('.')[1]) == spot_to_scan)
            #
            # fluor_files = [file for file in img_stack if file.split(".")[-2] in 'ABC']
            # if fluor_files:
            #     img_stack = tuple(file for file in img_stack if file not in fluor_files)
            #     print("\nFluorescent channel(s) detected: {}\n".format(fluor_files))

            first_img = pps_list[scan]
            print(first_img)
            if first_img.split('.')[-1] == 'tif':
                 tiff_toggle = True
            else:
                tiff_toggle = False
            validity = True

            spot_num, pass_num = map(int,first_img.split(".")[1:3])

            spot_pass_str = '{}.{}'.format(str(spot_num).zfill(3), str(pass_num).zfill(3))
            img_name = '.'.join([chip_name, spot_pass_str])

            spot_type = mAb_dict[spot_num][0].split("(")[-1].strip(")")

            vdata_file = vpipes.find_file(img_name + '.vdata.txt',vcount_dir)

            if type(vdata_file) != type(None):

                old_vdata_dict = {}
                with open(vdata_file) as f:
                    for line in f:
                        (key, val) = line.split(':')
                        old_vdata_dict[key] = val.strip(' \n')

                if (old_vdata_dict['validity'] == 'True')&(old_vdata_dict['version'] == version):
                    print("Previous data detected. Loading from {}\n".format(vdata_file.split('/')[-1]))

                    shift_dict[spot_pass_str] = tuple(map(float, old_vdata_dict['valid_shift'][1:-1].split(',')))

                    marker_list = old_vdata_dict['marker_locs'][2:-2].split('), (')
                    marker_dict[spot_pass_str] = list(map(vpipes.coord_parser, marker_list))

                    if pass_num == first_scan:
                        circle_dict[spot_num] = tuple(map(float, old_vdata_dict['spot_coords'][1:-1].split(',')))

                else:
                    print("Previous data detected but not loaded; will be overwritten.\n")

            if tiff_toggle == True:

                # images = vmulti.multip_image_reader(img_stack)

                for file in img_stack:
                    with TiffFile(file) as tif:
                        pages = range(tif.__len__())
                        pic3D = tif.asarray(key=pages)


            else:
                scan_collection = skio.imread_collection(img_stack)
                pic3D = np.array([pic for pic in scan_collection], dtype='uint16')
                if convert_tiff == True:
                    vpipes.pgm_to_tiff(pic3D, img_name, img_stack,
                                       tiff_compression=1, archive_pgm=True)

            print("Image Loaded\n")
            # if pic3D.shape[0] == 21:
            #     pic3D = pic3D[1:-2,:,:]


            # pic3D_orig = pic3D.copy()

            zslice_count, nrows, ncols = pic3D.shape
            total_pixels = nrows*ncols

            if total_pixels == 6981120:
                high_res = True
                minRad =700
                maxRad =1301
                cam_micron_per_pix = 3.45
                pix_per_um = mag / cam_micron_per_pix
                spacing = 1 / pix_per_um
                conv_factor = (cam_micron_per_pix / mag)**2
            else:
                minRad =350
                maxRad =601


            if mirror.size == total_pixels:
                pic3D = pic3D / mirror
                print("Applying mirror to image stack...\n")


            pic3D_norm = np.uint8(normalize(pic3D, None, 0, 255, NORM_MINMAX))

            pic3D_clahe = vimage.cv2_clahe_3D(pic3D_norm, kernel_size=(1,1), cliplim=4)

            # def multip_contrast_adjust(img_list, cpu_count = cpu_count()):
            #     start = default_timer()
            #     p = Pool(cpu_count)
            #     all_data = p.map(contrast_adjustment, img_list)
            #     print("Finished reading data in {} s using {} process(es)".format(default_timer()-start,p._processes))
            #
            #     return all_data
            #
            # images = multip_contrast_adjust(images)

            pic3D_rescale = vimage.rescale_3D(pic3D_clahe, perc_range=perc_range)

            print("Contrast adjusted\n")
            #Many operations are on the Z-stack compressed image. Several methods, but Standard Deviation works well.
            # maxmin_proj_rescale = np.max(pic3D_rescale, axis = 0) - np.min(pic3D_rescale, axis = 0)

            sd_proj_rescale = np.std(pic3D_rescale, axis=0)
            #Convert to 8-bit for OpenCV comptibility
            sd_proj_rescale = np.uint8(cv2.normalize(sd_proj_rescale, None, 0, 255, cv2.NORM_MINMAX))

            if pass_num == 1: marker = IRISmarker
            else: marker = found_markers

            if spot_pass_str not in marker_dict:
                marker_locs = vimage.marker_finder(pic3D_rescale[0], marker=marker,  thresh=0.6)
                marker_dict[spot_pass_str] = marker_locs
            else:
                marker_locs = marker_dict[spot_pass_str]



            pos_plane_list = vquant.measure_focal_plane(pic3D_norm, marker_locs,
                                                        exo_toggle, marker_shape=IRISmarker.shape
            )

            if pos_plane_list != []:
                pos_plane = max(pos_plane_list)
            else:
                pos_plane = zslice_count // 3

            pic_rescale_pos = pic3D_rescale[pos_plane]

            overlay_dict[spot_pass_str] = sd_proj_rescale

            print("Using image {} from stack\n".format(str(pos_plane + 1).zfill(3)))

            if pass_counter <= 15:
                overlay_mode = 'series'
            else:
                overlay_mode = 'baseline'

            if pass_num == first_scan:
                print("First Valid Scan\n")
                valid_shift = (0,0)
                overlay_toggle = False

            else:
                prescan_img, postscan_img = vimage._dict_matcher(overlay_dict, spot_num, pass_num, mode=overlay_mode)
                overlay_toggle = True
                if spot_pass_str in shift_dict:
                    valid_shift = shift_dict[spot_pass_str]


                else:

                    median_shift = vimage.measure_shift_ORB(prescan_img, postscan_img, ham_thresh=10, show=False)
                    for coord in median_shift:
                        if abs(coord) < 75:

                            valid_shift = median_shift
                        else: ##In case ORB fails to give a good value
                            # overlay_toggle = False
                            print("Using alternative shift measurement...\n")
                            mean_shift, overlay_toggle = vimage.measure_shift(marker_dict,pass_num,
                                                                                spot_num,mode=overlay_mode
                            )
                            valid_shift = mean_shift

                print("Valid Shift: {}\n".format(valid_shift))

                img_overlay = vimage.overlayer(prescan_img, postscan_img, valid_shift)
                shape_mask = vimage.shape_mask_shift(shape_mask, valid_shift)
                img_overlay_difference = np.int16(img_overlay[:,:,1]) - np.int16(img_overlay[:,:,0])
                median_overlay = np.median(img_overlay_difference)
                sd_overlay = np.std(img_overlay_difference)
                print(median_overlay, sd_overlay)



                # overlay_name = "{}_overlay_{}".format(img_name, overlay_mode)
                # vimage.gen_img_deets(img_overlay_difference, name=overlay_name, savedir=overlay_dir)

            if (overlay_toggle == False) & (pass_num != first_scan):
                validity = False
                print("No compatible markers, cannot compute shift")
            #
            # else:
            #     print("Cannot overlay images\n")
            if spot_num in circle_dict:
                spot_coords = circle_dict[spot_num]
                shift_x = spot_coords[0] + valid_shift[1]
                shift_y = spot_coords[1] + valid_shift[0]
                spot_coords = (shift_x, shift_y, spot_coords[2])
                # circle_dict[spot_num] = spot_coords

            else:
                # shift_x = spot_coords[0] + valid_shift[1]
                # shift_y = spot_coords[1] + valid_shift[0]

                # pic_canny = canny(sd_proj_rescale, sigma = canny_sig)

                # spot_coords = vimage.spot_finder(pic_canny,rad_range=(425,651),Ab_spot=Ab_spot_mode)
                circles = None
                cannyMax = 200
                cannyMin = 100
                while type(circles) == type(None):
                    circles = cv2.HoughCircles(sd_proj_rescale,cv2.HOUGH_GRADIENT,1,minDist=500,
                                               param1=cannyMax, param2=cannyMin,
                                               minRadius=minRad, maxRadius=maxRad
                    )
                    cannyMax-=50
                    cannyMin-=25
                spot_coords = tuple(map(lambda x: round(x,0), circles[0][0]))
                print("Spot center coordinates (row, column, radius): {}\n".format(spot_coords))

            circle_dict[spot_num] = spot_coords


            row, col = np.ogrid[:nrows,:ncols]
            width = col - spot_coords[0]
            height = row - spot_coords[1]
            rad = spot_coords[2] - 25
            disk_mask = (width**2 + height**2 > rad**2)

            marker_mask, found_markers = vimage.marker_masker(pic3D_rescale[0], marker_locs, marker)

            full_mask = disk_mask + marker_mask



            # vimage.image_details(pic3D_norm[pos_plane], pic3D_clahe[pos_plane], pic3D_rescale[pos_plane].copy(), pic_canny)

            # maxmin_proj_rescale_masked = np.ma.array(maxmin_proj_rescale, mask=full_mask).filled(fill_value=np.nan)

            # maxmin_median = np.nanmedian(maxmin_proj_rescale_masked)

#*********************************************************************************************#
            # if pass_num > first_scan:


                # if overlay_toggle == True:
                    # img_overlay = np.ma.array(img_overlay, mask=full_mask).filled(fill_value=np.nan)


            # if Ab_spot_mode == False:
            #     pic_to_show = sd_proj_rescale
            # elif exo_toggle == True:
            #     pic_to_show = sd_proj_rescale
            # else:

            if pass_num > 1:
                pic_to_show = img_overlay_difference
            else:
                pic_to_show = sd_proj_rescale

#*********************************************************************************************#
            with warnings.catch_warnings():
                ##RuntimeWarning ignored: invalid values are expected
                warnings.simplefilter("ignore")
                warnings.warn(RuntimeWarning)

                shapedex = shape_index(pic_rescale_pos)
                shapedex = np.ma.array(shapedex,mask = full_mask).filled(fill_value = np.nan)
                if pass_num > first_scan:
                    shapedex = np.ma.array(shapedex,mask = shape_mask).filled(fill_value = -1)

                shapedex_gauss = gaussian_filter(shapedex, sigma=1)

            pix_area = np.count_nonzero(np.invert(np.isnan(shapedex)))
            area_sqmm = round((pix_area * conv_factor) * 1e-6, 6)

            ##Pixel topology classifications
            background = 0
            ridge = 0.5
            sphere = 1

            bg_rows,bg_cols = zip(*vquant.classify_shape(shapedex, sd_proj_rescale, background,
                                                         delta=0.25, intensity=0)
            )
            sd_proj_bg = sd_proj_rescale[bg_rows,bg_cols]

            sd_proj_bg_median = np.median(sd_proj_bg)##Important
            sd_proj_bg_stdev = np.std(sd_proj_bg)
            print("Median intensity of spot background={}, SD={}".format(round(sd_proj_bg_median,4),
                                                                         round(sd_proj_bg_stdev,4))
            )


            if Ab_spot_mode == True:
                if exo_toggle == True:

                    ridge_thresh   = sd_proj_bg_median*3.5
                    # ridge_thresh  = sd_proj_bg_median+(sd_proj_bg_stdev*7)
                    sphere_thresh  = sd_proj_bg_median*2.5
                    # sphere_thresh  = sd_proj_bg_median+(sd_proj_bg_stdev*6)
                    ridge_thresh_s = sd_proj_bg_median*3.5
                    # ridge_thresh_s = sd_proj_bg_median+(sd_proj_bg_stdev*6)
                else:
                    ridge_thresh   = sd_proj_bg_median+sd_proj_bg_stdev*2
                    sphere_thresh  = sd_proj_bg_median+sd_proj_bg_stdev*2
                    ridge_thresh_s = sd_proj_bg_median+sd_proj_bg_stdev*3
            else:
                ridge_thresh   = sd_proj_bg_median+sd_proj_bg_stdev*2.75
                sphere_thresh  = sd_proj_bg_median+sd_proj_bg_stdev*2.75
                ridge_thresh_s = sd_proj_bg_median+sd_proj_bg_stdev*2.75

            ridge_list = vquant.classify_shape(shapedex, sd_proj_rescale, ridge,
                                               delta=0.25, intensity=ridge_thresh
            )

            sphere_list = vquant.classify_shape(shapedex, sd_proj_rescale, sphere,
                                                delta=0.2, intensity=sphere_thresh
            )

            ridge_list_s = vquant.classify_shape(shapedex_gauss, sd_proj_rescale, ridge,
                                                 delta=0.3, intensity=ridge_thresh_s
            )

            pix_list = ridge_list + sphere_list
            ridge_list_s = list(set(pix_list) - set(ridge_list_s))
            pix_list = pix_list + ridge_list_s

            pic_binary = np.zeros_like(sd_proj_rescale, dtype=int)

            if not pix_list == []:
                rows,cols = zip(*pix_list)

                pic_binary[rows,cols] = 1

                pic_binary = binary_fill_holes(pic_binary)

#*********************************************************************************************#
            vdata_dict.update({'img_name': img_name,
                               'spot_type': spot_type,
                               'area_sqmm': area_sqmm,
                               'valid_shift': valid_shift,
                               'overlay_mode': overlay_mode,
                               'pos_plane': pos_plane,
                               'spot_coords': spot_coords,
                               'marker_locs': marker_locs,
                               'classifier_median':round(sd_proj_bg_median,4)
            })

#*********************************************************************************************#
            ##EXTRACT DATA FROM THE BINARY IMAGE
            prop_list =['label','coords','area','centroid','moments_central','bbox',
                        'filled_image','major_axis_length','minor_axis_length']

            shape_df = vquant.binary_data_extraction(pic_binary, pic3D[pos_plane], prop_list, pix_range=(3,500))
            print('S')
            if not shape_df.empty:

                particle_mask = vquant.particle_masker(pic_binary, shape_df, pass_num, first_scan)

                if pass_num == first_scan:
                    shape_mask = binary_dilation(particle_mask, iterations=3)
                else:
                    shape_mask = np.add(shape_mask, binary_dilation(particle_mask, iterations=2))

                shape_df['pass_number'] = [pass_num]*len(shape_df.index)

                shape_df['coords'] = shape_df.coords.apply(lambda a: [tuple(x) for x in a])


                shape_df['bbox'] = shape_df.bbox.map(vquant.bbox_verts)

                print('R')
            else:
                print("----No valid particle shapes----\n")
                vdata_dict.update({'total_valid_particles': 0, 'validity':False})
                print(shape_df)
                prev_scan_validity = False

                with open('{}/{}.vdata.txt'.format(vcount_dir,img_name),'w') as f:
                    for k,v in vdata_dict.items():
                        f.write('{}: {}\n'.format(k,v))

                vgraph.gen_particle_image(pic_to_show,shape_df,spot_coords,
                                          pix_per_um=pix_per_um,
                                          show_particles=False,
                                          cv_cutoff=cv_cutoff,
                                          r2_cutoff=0,
                                          scalebar=15, markers=marker_locs,
                                          exo_toggle=exo_toggle
                )
                plt.savefig('{}/{}.{}.png'.format(img_dir, img_name, spot_type), dpi = 96)
                plt.clf(); plt.close('all')
                print("#******************PNG generated for {}************************#".format(img_name))

                continue
#*********************************************************************************************#

            filo_pts_tot, round_pts_tot  = [],[]
            z_intensity_list, max_z_slice_list, max_z_stacks, shape_validity = [],[],[],[]
            greatest_max_list, sd_above_med_difference = [],[]
            intensity_increase_list = []
            print('Measuring particle intensities...\n')
            for coord_array in shape_df.coords:

                coord_set = set(coord_array)
                filo_pts = len(coord_set.intersection(ridge_list))

                filo_pts = filo_pts + (len(coord_set.intersection(ridge_list_s)) * 0.15)

                round_pts = len(coord_set.intersection(sphere_list))

                filo_pts_tot.append(filo_pts)
                round_pts_tot.append(round_pts)

                if pic3D.ndim > 2:

                    all_z_stacks = np.array([pic3D[:,coords[0],coords[1]] for coords in coord_array])
                    greatest_max = np.max(all_z_stacks)
                    max_z_stack = all_z_stacks[np.where(all_z_stacks == np.max(all_z_stacks))[0][0]].tolist()
                    if max_z_stack[0] >= max_z_stack[-1]:
                        shape_validity.append(True)
                    else:
                        shape_validity.append(False)
                    maxmax_z = max(max_z_stack)
                    max_z_slice = max_z_stack.index(maxmax_z)
                    z_intensity = (maxmax_z - min(max_z_stack))*100
                    # mean_z_stack = list(np.round(np.mean(all_z_stacks, axis=0),4))
                    std_z_stack = list(np.round(np.std(all_z_stacks, axis=0),4))

                    # defocus, max_z, r2 = vquant.measure_defocus(max_z_stack,std_z_stack, measure_corr=False)

                    # defocus_list.append(defocus)
                    max_z_slice_list.append(max_z_slice)
                    max_z_stacks.append(max_z_stack)
                    z_intensity_list.append(z_intensity)
                    greatest_max_list.append(greatest_max)


                if (pass_num > 1) & (overlay_toggle == True):

                    intensity_increase = max([img_overlay_difference[coords[0],coords[1]] for coords in coord_array])
                    intensity_increase_list.append(intensity_increase)
                    # sd_above_med_difference.append(round((intensity_increase - median_overlay)/sd_overlay,3))
                else:
                    intensity_increase_list = [np.nan] * len(shape_df)
                    # sd_above_med_difference = [np.nan] * len(shape_df)

            print('N')


            # shape_df['defocus'] = defocus_list
            shape_df['max_z_slice'] = max_z_slice_list
            shape_df['max_z_stack'] = max_z_stacks
            shape_df['z_intensity'] = z_intensity_list

            shape_df['greatest_max'] = greatest_max_list
            shape_df['validity'] = shape_validity

            shape_df['filo_points'] = filo_pts_tot
            shape_df['round_points'] = round_pts_tot

            shape_df['intensity_increase'] = intensity_increase_list


            bbox_pixels = [vquant.get_bbox_pixels(bbox, pic3D[z])
                          for i, z, bbox in shape_df[['max_z_slice','bbox']].itertuples()
            ]

            median_bg_list, shape_df['cv_bg'] = zip(*map(lambda x: (np.median(x),
                                                                    np.std(x)/np.mean(x)),
                                                                    bbox_pixels)
            )

            shape_df['perc_contrast'] = ((shape_df['greatest_max'] - median_bg_list)*100
                                                    / median_bg_list
            )

            shape_df.loc[shape_df.perc_contrast <= 0,'validity'] = False
            shape_df.loc[shape_df.cv_bg > cv_cutoff,'validity'] = False
            shape_df.loc[shape_df.intensity_increase < 40,'validity'] = False

            if len(shape_df) > 1:
                regression = smapi.OLS(shape_df.z_intensity, shape_df.perc_contrast).fit()
                outlier_df = regression.outlier_test()
                shape_df.loc[outlier_df['bonf(p)'] < 0.5, 'validity'] = False
            print('A')
            shape_df = vquant.remove_overlapping_objs(shape_df, radius=10)
            print('B')
            # else:
            #     shape_df.loc[shape_df.index.isin(overlap_ix),'validity' ] = False
            # if exo_toggle == False:
            #     shape_df.loc[(shape_df.max_z == 0) | (shape_df.max_z == (zslice_count-1)),'validity'] = False
#---------------------------------------------------------------------------------------------#
            ##Filament Measurements


            shape_df['circularity'] = list(map(lambda A,P: round((4*np.pi*A)/(perimeter(P)**2),4),
                                                    shape_df.area, shape_df.filled_image))

            shape_df['ellipticity'] = round(shape_df.major_axis_length/shape_df.minor_axis_length,4)#max val = 1

            shape_df['eccentricity'] = shape_df.moments_central.map(vquant.eccentricity)


            # for nu in shape_df.nu_ji:
            #     print(abs(nu[1,1]))
            shape_df = shape_df[(shape_df['filo_points'] + shape_df['round_points']) >= 1]

            shape_df['filo_score'] = ((shape_df['filo_points'] / shape_df['area'])
                                     -(shape_df['round_points'] / shape_df['area'])
            )
            shape_df['roundness_score'] = ((shape_df['round_points'] / shape_df['area']))



#---------------------------------------------------------------------------------------------#
            filolen_df = pd.DataFrame([vfilo.measure_fiber_length(coords, spacing=spacing)
                                        for coords in shape_df.coords],
                                       columns=['fiber_length','vertices'], index=shape_df.index)
            shape_df = pd.concat([shape_df, filolen_df],axis=1)
            shape_df['curl'] = (shape_df['major_axis_length'] * spacing) / shape_df['fiber_length']


            total_particles = len(shape_df)
            shape_df['channel'] = ['V'] * total_particles

            valid_shape_df = shape_df[(shape_df['cv_bg'] < cv_cutoff) & (shape_df['validity'] == True)]
            total_valid_particles = len(valid_shape_df)
#---------------------------------------------------------------------------------------------#
            ### Fluorescent File Processer

            if not fluor_files:
                for fluor_file in fluor_files[:1]:
                    fluor_norm, channel = vfluor.normalize_img(fluor_file)
                    f1,f2 = np.percentile(fluor_norm, (2,98))
                    print(f1,f2)

                    fluor_binary = np.zeros_like(fluor_img, dtype=int)
                    background_fl = []

                    if f2 <= f1:
                        print('Fluorescence too weak to measure')
                        break
                    # else:
                    fluor_rescale = rescale_intensity(fluor_norm, in_range=(f1,f2))
                    with warnings.catch_warnings():##RuntimeWarning ignored: invalid values are expected
                        warnings.simplefilter("ignore")
                        warnings.warn(RuntimeWarning)
                        shapedex_fl = np.ma.array(shape_index(fluor_rescale), mask=full_mask).filled(fill_value=-1)

                    background_fl = vquant.classify_shape(gaussian_filter(shapedex_fl, sigma=1),
                                                                     fluor_rescale, background,
                                                                     delta=0.15, intensity=0)


#*********************************************************************************************#



            # jp = sns.jointplot(x=valid_shape_df.z_intensity, y=valid_shape_df.perc_contrast, kind='reg',
            #                   xlim=(0,0.2), ylim=(0, 20), size=8,
            #                   joint_kws=dict(marker='+'),
            #                   marginal_kws=dict(bins=200)
            #                   #'y={0:0.1f}x + {1:0.1f}'.format(mslope, yint)}
            #
            # )


            # print('\nPercentContrast = {0:0.1f}Intensity + {1:0.1f}\n'.format(mslope, yint))
            # print("R2 = {0:0.4f}".format(r2))
            # plt.show()
            if (show_filos == True):
                vgraph.filo_image_gen(shape_df, sd_proj_rescale, shapedex, pic_binary,
                                      ridge_list, sphere_list, list(zip(bg_rows,bg_cols)),
                                      cv_cutoff=cv_cutoff, r2_cutoff = 0, show=True
                )

#---------------------------------------------------------------------------------------------#

            kparticle_density = round(total_valid_particles / area_sqmm * 0.001, 2)

            if pass_num != first_scan:
                print("Particle density in {}: {} kp/sq.mm\n".format(img_name, kparticle_density))
            else:
                print("Background density in {}: {} kp/sq.mm\n".format(img_name, kparticle_density))


            # shape_df.reset_index(drop=True, inplace=True)

            total_shape_df = pd.concat([total_shape_df, shape_df], axis=0, sort=False)
            # total_shape_df.reset_index(drop=True, inplace=True)

            keep_data = ['label','area','centroid','pass_number','max_z_slice',
                        'eccentricity','ellipticity',
                        'curl', 'circularity',
                         'validity','z_intensity','perc_contrast','cv_bg','sd_above_med_difference',
                         'fiber_length','filo_score','roundness_score','channel','fl_intensity'
                         ]
            # if (passes_per_spot > 1) & (pass_num == passes_per_spot) & (total_valid_particles > 0):
            #     vfilo.filohisto(total_shape_df, filo_cutoff = 0.25, irreg_cutoff = -0.25, range=filohist_range)
            #     plt.savefig('{}/{}.filohistogram.png'.format(filo_dir, img_name))
            #     print("Filament histogram generated")

            vdata_dict.update({'total_valid_particles': total_valid_particles, 'validity':validity})

            with open('{}/{}.vdata.txt'.format(vcount_dir,img_name),'w') as f:
                for k,v in vdata_dict.items():
                    f.write('{}: {}\n'.format(k,v))


#---------------------------------------------------------------------------------------------#
        ####Processed Image Renderer
            vgraph.gen_particle_image(pic_to_show,total_shape_df,spot_coords,
                                      pix_per_um=pix_per_um,
                                      cv_cutoff=cv_cutoff,
                                      r2_cutoff=0,
                                      show_particles=show_particles,
                                      scalebar=15, markers=marker_locs,
                                      exo_toggle=exo_toggle
            )


            plt.savefig('{}/{}.{}.png'.format(img_dir, img_name, spot_type), dpi = 96)
            plt.clf(); plt.close('all')
            print("#******************PNG generated for {}************************#".format(img_name))
            if not (shape_df.empty) | np.all(shape_df.validity == False):
                vgraph.defocus_profile_graph(valid_shape_df, pass_num, zslice_count,
                                               vcount_dir, exo_toggle, img_name
                )


#---------------------------------------------------------------------------------------------#
        total_shape_df.to_csv('{}/{}.particle_data.csv'.format(vcount_dir, spot_ID),
                              columns = keep_data
        )
        analysis_time = str(datetime.now() - startTime)

        spot_to_scan += 1

        print("Time to scan images: {}".format(analysis_time))
        # raise error

#*********************************************************************************************#
    info_dict = {'chip_name'      : chip_name,
                 'sample_info'    : sample_name,
                 'spot_number'    : spot_counter,
                 'total_passes'   : pass_counter,
                 'experiment_date': expt_date,
                 'analysis_time'  : analysis_time,
                 'zslice_count'   : zslice_count,
                 'version'        : version
    }
    with open('{}/{}.expt_info_{}.txt'.format(virago_dir,chip_name,version),'w') as info_file:
        for k,v in info_dict.items():
            info_file.write('{}: {}\n'.format(k,v))
#*********************************************************************************************#
os.chdir(virago_dir)
info_list = sorted(glob.glob('*_info_*'))
if info_list == []:
    print("No valid data to interpret. Exiting...")
    sys.exit()
else:
    version_list = ['.'.join(file.split('_')[-1].split('.')[:3]) for file in info_list]
    version = max(version_list, key=vpipes.version_finder)
    print("\nData from version {}\n".format(version))

    info_file = [file for file in info_list if version in file][0]

    with open(info_file) as info_f:

        info_list= [line.split(':') for line in info_f]
        info_dict = {line[0]:line[1].strip(' \n') for line in info_list}




os.chdir(vcount_dir)
particle_data_list = sorted(glob.glob(chip_name +'*.particle_data.csv'))
vdata_list = sorted(glob.glob(chip_name +'*.vdata.txt'))

if len(vdata_list) >= (len(iris_txt) * pass_counter):
    metric_str = str(input("\nEnter the minimum and maximum percent intensity values,"
                                + " separated by a dash.\n"))
    while "-" not in metric_str:
        metric_str = str(input("\nPlease enter two values separated by a dash.\n"))
    else:
        min_cont, max_cont = map(float, metric_str.split("-"))

    metric_window = [min_cont, max_cont]

    vdata_df = vquant.vdata_reader(vdata_list)

    spot_df = pd.concat([spot_df, vdata_df[['area_sqmm',
                                            'validity',
                                            'fluor_particles_A',
                                            'fluor_particles_C']]], axis=1)


    bad_spots = set(int(img_name.split('.')[1])
                    for img_name in list(vdata_df.img_name[vdata_df.validity == False])
    )
    particle_counts, filo_counts,irreg_counts = [],[],[]

    if int(info_dict['zslice_count']) >= 18:
        histo_metric ='z_intensity'
    else:
        histo_metric ='perc_contrast'
    print("Using {} to make histograms".format(histo_metric))

    metric_df = pd.DataFrame()
    new_particle_count, cum_particle_count = [],[]
    for i, csvfile in enumerate(particle_data_list):
        if i+1 not in bad_spots:
            particle_df = pd.read_csv(csvfile, error_bad_lines=False,  header=0,
                                      usecols=['perc_contrast','z_intensity','validity','pass_number'])
            cumulative_particles = 0
            for j in range(1,pass_counter+1):

                area_str = spot_df.area_sqmm[(spot_df.spot_number == i+1)
                                            & (spot_df.scan_number == j)].values[0]
                area_squm = int(float(area_str)*1e6)

                csv_id = '{}.{}.{}'.format(csvfile.split(".")[1], str(j).zfill(3), area_squm)

                metric_series = particle_df[histo_metric][ (particle_df.pass_number == j)
                                                         & (particle_df.validity == True)
                ].reset_index(drop=True).rename(csv_id)

                metric_df = pd.concat([metric_df, metric_series], axis=1)

                kept_particles = metric_series[(metric_series > min_cont)
                                             & (metric_series <= max_cont)
                ]

                particles_per_pass = len(kept_particles)
                new_particle_count.append(particles_per_pass)
                cumulative_particles += particles_per_pass
                cum_particle_count.append(cumulative_particles)

                print(  'File scanned: {}; '.format(csvfile)
                      + 'Scan {}, '.format(j)
                      + 'Particles accumulated: {}'.format(cumulative_particles)
                )
        else:
            print("No data for {}\n".format(csvfile))
            new_particle_count = new_particle_count + ([0]*pass_counter)
            cum_particle_count = cum_particle_count + ([0]*pass_counter)

    spot_df['new_particles'] = new_particle_count
    spot_df['cumulative_particles'] = cum_particle_count
    spot_df['kparticle_density'] = np.round(cum_particle_count
                                            / spot_df.area_sqmm.astype(float) * 0.001, 3
    )

    spot_df.loc[spot_df.kparticle_density == 0, 'validity'] = False

    spot_df['normalized_density'] = vquant.density_normalizer(spot_df, spot_counter)

    os.chdir(iris_path)

elif len(vdata_list) != (len(iris_txt) * pass_counter):
    print("Missing VIRAGO analysis files! Exiting...\n")
    sys.exit()

print(spot_df[['spot_number','scan_number','spot_type','validity','kparticle_density','normalized_density']])

spot_df, metric_df = vquant.spot_remover(spot_df, metric_df, vcount_dir, iris_path,
                                           quarantine_img=True
)


#*********************************************************************************************#
##HISTOGRAM CODE

raw_histogram_df = vgraph.histogrammer(metric_df, spot_counter, metric_window, bin_size=0.2)
raw_histogram_df.to_csv('{}/{}_raw_histogram_data.v{}.csv'.format(histo_dir, chip_name, version))

sum_histogram_df = vgraph.sum_histogram(raw_histogram_df, spot_counter)
sum_histogram_df.to_csv('{}/{}.sum_histogram_data.v{}.csv'.format(histo_dir, chip_name, version))

avg_histogram_df = vgraph.average_histogram(sum_histogram_df, spot_df, pass_counter)
avg_histogram_df.to_csv('{}/{}_avg_histogram_data.v{}.csv'.format(histo_dir, chip_name, version))

vgraph.generate_histogram(avg_histogram_df, pass_counter, chip_name, metric_str, histo_metric, histo_dir)

#*********************************************************************************************#
spot_df['normalized_density'] = vquant.density_normalizer(spot_df, spot_counter)
spot_df_name='{}/{}_spot_data.{}contrast.v{}.csv'.format(virago_dir, chip_name,
                                                         metric_str, version
)
spot_df.to_csv(spot_df_name)
print('File generated: {}'.format(spot_df_name))

averaged_df = vgraph.average_spot_data(spot_df, pass_counter)

if pass_counter > 2:
    vgraph.generate_timeseries(spot_df, averaged_df, metric_window, mAb_dict,
                                chip_name, sample_name, version,
                                scan_or_time = timeseries_mode, baseline = True,
                                savedir = virago_dir
    )
elif pass_counter <= 2:
    vgraph.iris_barplot_gen(spot_df, pass_counter, metric_window=metric_window, chip_name=chip_name,
                            version=version, savedir=virago_dir, plot_3sigma=True,
    )
if sys.platform != 'win32':
    vgraph.chipArray_graph(spot_df,
                           chip_name=chip_name, sample_name=sample_name, metric_str=metric_str,
                           exo_toggle=exo_toggle, savedir=virago_dir, version=version
    )



def fluor_bargraph(spot_df, pass_counter, chip_name=chip_name, version=version,
                    savedir=fluor_dir, plot_3sigma=True, Amab ='', Cmab='',
                    neg_ctrl_str='8G5|MOUSE IGG|muIgG|GFP'):

    """
    Generates a barplot for the dataset.
    Most useful for before and after scans (pass_counter == 2)
    """

    sns.set_style('darkgrid')

    Achan = '594(A)'
    Cchan = '695(C)'
    axis_min = 2

    pre_df = spot_df[(spot_df.scan_number == 1)  & (spot_df.validity == True)]
    post_df = spot_df[(spot_df.scan_number == pass_counter) & (spot_df.validity == True)]
    dflen = len(pre_df)

    chan_series = pd.Series(['{}: {}'.format(Achan,Amab)]*dflen + ['{}: {}'.format(Cchan,Cmab)]*dflen, name='channel')
    spot_type_series = pre_df.spot_type.apply(lambda x: x.split('_')[0])
    pre_area_series = pre_df.area_sqmm.astype('float')*1000
    spot_type_series_x2 = pd.concat([spot_type_series, spot_type_series]).reset_index(drop=True)
    channel_pre_df  = pd.concat([pre_df['fluor_particles_A'].astype('float')/pre_area_series,
                                 pre_df['fluor_particles_C'].astype('float')/pre_area_series],
                                 ignore_index=True).rename('fluor_val_pre')

    fluorbar_df = pd.concat([spot_type_series_x2, chan_series, channel_pre_df], axis=1)
    ax1_max = max(fluorbar_df['fluor_val_pre'])
    if ax1_max < axis_min:
        ax1_max = axis_min
    elif isnan(ax1_max):
        ax1_max = axis_min
    else:
        ax1_max = round(ax1_max,1) + 1

    post_area_series =post_df.area_sqmm.astype('float')*1000
    channel_post_df  = pd.concat([post_df['fluor_particles_A'].astype('float')/ post_area_series,
                                  post_df['fluor_particles_C'].astype('float')/ post_area_series],
                                  ignore_index=True).rename('fluor_val_post')


    fluorbar_df = pd.concat([fluorbar_df, channel_post_df], axis=1)

    ax2_max = max(fluorbar_df['fluor_val_post'])
    if ax2_max < axis_min:
        ax2_max = axis_min
    elif isnan(ax2_max):
        ax2_max = axis_min
    else:
        ax2_max = round(ax2_max,1) + 1

    fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(8, 6), sharey=True,
                                 gridspec_kw = {'width_ratios':[ax1_max, ax2_max]}
    )
    colors = ('#4daf4a','#de2d26')
    # labels = [Patch(color=colors[c], label=val) for c, val in enumerate(fluorbar_df.channel.unique())]

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.suptitle("Fluorescence of {}".format(chip_name), y=1, fontsize=20)
    plt.xlabel("Fluorescent Particle Density (kparticles/mm" + r'$^2$'+')', fontsize=14)

    ax1 = sns.barplot(x='fluor_val_pre',y='spot_type',hue='channel', data=fluorbar_df,
                     palette=colors, alpha = 0.5, errwidth=2, ci='sd', ax=ax1)
    ax1.set_xlim([ax1_max,0])
    ax1.set_title("Prescan", fontsize=12)
    ax1.set_ylabel('')
    ax1.set_xlabel('')

    ax1.legend('')

    ax2 = sns.barplot(x='fluor_val_post',y='spot_type',hue='channel', data=fluorbar_df,
                     palette=colors, errwidth=2, ci='sd', ax=ax2)
    if plot_3sigma == True:
        neg_control_df = fluorbar_df[fluorbar_df.spot_type.str.contains(neg_ctrl_str)]
        A_neg_vals = neg_control_df.fluor_val_post[neg_control_df.channel.str.contains('A')]
        C_neg_vals = neg_control_df.fluor_val_post[neg_control_df.channel.str.contains('C')]
        three_sigma_A = (np.std(A_neg_vals) * 3) + np.mean(A_neg_vals)
        three_sigma_C = (np.std(C_neg_vals) * 3) + np.mean(C_neg_vals)
        ax2.axvline(x=three_sigma_A,ls='--',lw=1,color='g', label='3'+r'$\sigma$'+' Signal Threshold A')
        ax2.axvline(x=three_sigma_C,ls=':',lw=1,color='r', label='3'+r'$\sigma$'+' Signal Threshold C')
        # line_legend = ax2.get_legend_handles_labels()
        # labels = labels+line_legend[0]
    labels = ax2.get_legend_handles_labels()[0]
    ax2.legend(handles=labels, fontsize=10, loc ='best')
    ax2.set_xlim([0,ax2_max])
    ax2.yaxis.set_tick_params(labelsize=12, rotation = 45)
    ax2.set_title("Postscan", fontsize=12)
    ax2.set_ylabel('')
    ax2.set_xlabel('')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    plot_name="{}_fluorescence_barplot.v{}.png".format(chip_name, version)
    plt.savefig('{}/{}'.format(fluor_dir, plot_name), bbox_inches='tight', pad_inches=0.1, dpi=300)
    print('File generated: {}'.format(plot_name))
    plt.close('all')

if fluor_files:
    fluor_fig = fluor_bargraph(spot_df, pass_counter, chip_name, version, virago_dir,
                               Amab=amab, Cmab=cmab
    )
