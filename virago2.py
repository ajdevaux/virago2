#!/usr/bin/env python3
from __future__ import division
from future.builtins import input
from datetime import datetime
from skimage import io as skio
from skimage.filters import laplace, sobel_h, sobel_v
from skimage.feature import shape_index
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from modules import vpipes, vimage, vquant, vgraph
# from modules import filoquant as filo
from images import logo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os, warnings, sys
# import random

pd.set_option('display.width', 1000)
pd.options.display.max_rows = 999
logo.print_logo()
version = '2.7.6'
print("VERSION {}".format(version))
#*********************************************************************************************#
#
#    CODE BEGINS HERE
#
#*********************************************************************************************#

IRISmarker_liq = skio.imread('images/IRISmarker_new.tif')
IRISmarker_exo = skio.imread('images/IRISmarker_maxmin_v4.tif')
finish_anal = 'no'
pgm_list, zip_list = [],[]
marker_dict = {}
while (pgm_list == []) and (zip_list == []): ##Keep repeating until pgm files are found
    iris_path = input("\nPlease type in the path to the folder that contains the IRIS data:\n")
    if iris_path == 'test':
        iris_path = '/Volumes/KatahdinHD/ResilioSync/DATA/IRIS/FIGS4PAPER/expts/tCHIP007_EBOVmay@1E6'
    else:
        iris_path = iris_path.strip('"')##Point to the correct directory
    os.chdir(iris_path)
    pgm_list = sorted(glob.glob('*.pgm'))
    zip_list = sorted(glob.glob('*.bz2'))
if pgm_list:
    archive_mode = False
else:
    archive_mode = True
    print("\nArchive extraction mode\n")

txt_list = sorted(glob.glob('*.txt'))
xml_list = sorted(glob.glob('*/*.xml'))
if not xml_list: xml_list = sorted(glob.glob('../*/*.xml'))
chip_name = pgm_list[0].split(".")[0]

pgm_list, mirror = vpipes.mirror_finder(pgm_list)
pgm_set = set([".".join(file.split(".")[:3]) for file in pgm_list])
pgm_set2 = pgm_set
zslice_count = max([int(pgmfile.split(".")[3]) for pgmfile in pgm_list])
txtcheck = [file.split(".") for file in txt_list]
iris_txt = [".".join(file) for file in txtcheck if (len(file) >= 3) and (file[2].isalpha())]

xml_file = [file for file in xml_list if chip_name in file]
chip_file = vpipes.chip_file_reader(xml_file[0])

mAb_dict, mAb_dict_rev = vpipes.dejargonifier(chip_file)
spot_tuple = tuple(mAb_dict_rev.keys())

sample_name = vpipes.sample_namer(iris_path)

virago_dir = '{}/v2results/{}'.format('/'.join(iris_path.split('/')[:-1]), chip_name)
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
# if not os.path.exists(overlay_dir): os.makedirs(overlay_dir)
# if not os.path.exists(fluor_dir): os.makedirs(fluor_dir)
# if not os.path.exists(filo_dir): os.makedirs(filo_dir)
if not os.path.exists(vcount_dir):
    os.makedirs(vcount_dir)

else:
    os.chdir(vcount_dir)
    vdata_list = sorted(glob.glob(chip_name +'*.vdata.txt'))

    if len(vdata_list) < len(pgm_set):
        finish_anal = input("Data partially analyzed. Finish (y) or restart (n)? (y/[n])")
        if finish_anal.lower() in ('yes', 'y'):
            vdata_names = ['.'.join(file.split('.')[:3]) for file in vdata_list]
            pgm_set = pgm_set.difference(vdata_names)

            vdata_dict = vquant.vdata_reader(vdata_list,['marker_coords_RC'])

            for i, filename in enumerate(vdata_list):
                splitname = filename.split('.')
                spot_num = int(splitname[1])
                pass_num = int(splitname[2])
                marker_dict['{}.{}'.format(spot_num, pass_num)] = vdata_dict['marker_coords_RC'][i]
            print(marker_dict)
#*********************************************************************************************#
# Text file Parser
#*********************************************************************************************#
os.chdir(iris_path)
spot_counter = len([key for key in mAb_dict])##Important
spot_df = pd.DataFrame([])
spot_list = [int(file[1]) for file in txtcheck if (len(file) > 2) and (file[2].isalpha())]

pass_counter = int(max([pgm.split(".")[2] for pgm in pgm_list]))##Important
if pass_counter > 3: timeseries_mode = 'time'
else: timeseries_mode = 'scan'

scanned_spots = set(np.arange(1,spot_counter+1,1))
missing_spots = tuple(scanned_spots.difference(spot_list))
for val in missing_spots:
    iris_txt.insert(val-1,val)

for ix, txtfile in enumerate(iris_txt):
    spot_data_solo = pd.DataFrame({'spot_number': [ix+1] * pass_counter,
                                   'scan_number': range(1,pass_counter + 1),
                                   'spot_type': [mAb_dict[ix+1]]*pass_counter
                                   })
    if not type(txtfile) is str:
        print("Missing text file for spot {}".format(txtfile))
        spot_data_solo['scan_time'] = [0] * pass_counter

    else:
        txtdata = pd.read_table(txtfile, sep = ':', error_bad_lines = False,
                            header = None, index_col = 0, usecols = [0, 1])
        expt_date = txtdata.loc['experiment_start'][1].split(" ")[0]

        pass_labels = [row for row in txtdata.index if row.startswith('pass_time')]
        times_s = txtdata.loc[pass_labels].values.flatten().astype(np.float)
        pass_diff = pass_counter - len(pass_labels)
        if pass_diff > 0:
            times_s = np.append(times_s, [0] * pass_diff)
        spot_data_solo['scan_time'] = np.round(times_s / 60,2)
        print('File scanned:  {}'.format(txtfile))

    spot_df = spot_df.append(spot_data_solo, ignore_index = True)

#*********************************************************************************************#
# PGM Scanning
spot_to_scan = 1

#*********************************************************************************************#
if finish_anal in ('yes', 'y'):
    pgm_toggle = 'yes'
elif (pgm_set != set()):
    pgm_toggle = input("\nImage files detected. Do you want scan them for particles? ([y]/n)\n"
                        + "WARNING: This will take a long time!\n")
else:
    pgm_toggle = 'no'

if pgm_toggle.lower() not in ('no', 'n'):
    if pgm_toggle.isdigit(): spot_to_scan = int(pgm_toggle)
    startTime = datetime.now()
    circle_dict, shift_dict, rotation_dict,marker_dict,overlay_dict,bin_mask_dict = {},{},{},{},{},{}

    while spot_to_scan <= spot_counter:

        pps_list = sorted([file for file in pgm_set
                                    if int(file.split(".")[1]) == spot_to_scan])
        passes_per_spot = len(pps_list)
        spot_ID = '{}.{}'.format(chip_name, vpipes.three_digs(spot_to_scan))

        if (passes_per_spot != pass_counter) and (finish_anal.lower() not in ('yes', 'y')):
            print("Missing pgm files... fixing...")

            scans_counted = [int(file.split(".")[-1]) for file in pps_list]
            scan_set = set(range(1,pass_counter+1))

            missing_csvs = scan_set.difference(scans_counted)

            for scan in missing_csvs:
                vpipes.bad_data_writer(chip_name, spot_to_scan, scan, marker_dict, vcount_dir)

        whole_spot_df = pd.DataFrame()
        cum_mean_shift = (0,0)
        for scan in range(0,passes_per_spot,1):
            scan_list = [file for file in pgm_list if file.startswith(pps_list[scan])]
            dpi = 96
            validity = True

            fluor_files = [file for file in scan_list if file.split(".")[-2] in 'ABC']
            if fluor_files:
                scan_list = [file for file in scan_list if file not in fluor_files]
                print("Fluorescent channel(s) detected: {}\n".format(fluor_files))

            scan_collection = skio.imread_collection(scan_list)
            pgm_name = scan_list[0].split(".")
            if 'bad' in pgm_name:
                print('\nBad image\n')
                for scan in range(1,passes_per_spot+1):
                    vpipes.bad_data_writer(spot_to_scan, scan, vcount_dir)
                break

            spot_num = int(pgm_name[1])
            pass_num = int(pgm_name[2])
            spot_pass_str = '{}.{}'.format(spot_num, pass_num)
            img_name = '.'.join(pgm_name[:3])
            spot_type = mAb_dict[spot_num]


            pic3D = np.array([pic for pic in scan_collection], dtype='uint16')
            pic3D_orig = pic3D.copy()

            zslice_count, nrows, ncols = pic3D.shape

            cam_micron_per_pix, mag, exo_toggle = vpipes.determine_IRIS(nrows, ncols)

            if exo_toggle == True:
                cv_cutoff = 5
                IRISmarker = IRISmarker_exo
                timeseries_mode = 'scan'
            else:
                cv_cutoff = 0.02
                IRISmarker = IRISmarker_liq

            if mirror.size == pic3D[0].size:
                pic3D = pic3D / mirror
                print("Applying mirror to images...\n")


            pic3D_norm = pic3D / (np.median(pic3D) * 2)

            pic3D_norm[pic3D_norm > 1] = 1

            pic3D_clahe = vimage.clahe_3D(pic3D_norm, cliplim = 0.004)##UserWarning silenced

            pic3D_rescale = vimage.rescale_3D(pic3D_clahe, perc_range = (3,97))

            print("Contrast adjusted\n")



            if zslice_count > 1: focal_plane = int(np.floor(zslice_count/2)) + 1
            else: focal_plane = 0

            # def find_focus(pic3D):
            #     z, nrows, ncols = pic3D.shape
            #     # pic3D_center = pic3D[:,(nrows//2-100):(nrows//2+100),(ncols//2-100):(ncols//2+100)]
            #     teng_vals = [np.mean(sobel_h(pic)**2 + sobel_v(pic)**2) for pic in pic3D]
            #     teng_vals_norm = [val/sum(teng_vals) for val in teng_vals]
            #     laplace_vals = [laplace(pic,3).var() for pic in pic3D]
            #     laplace_vals_norm = [val/sum(laplace_vals) for val in laplace_vals]
            #     teng_diff = list(np.diff(teng_vals))
            #     laplace_diff = list(np.diff(laplace_vals))
            #     # teng_sign = []
            #     # for val in teng_diff:
            #     #     if val < 0:
            #     #         teng_sign.append('Neg')
            #     #     elif val > 0:
            #     #         teng_sign.append('Pos')
            #     #     else:
            #     #         teng_sign.append(None)
            #     print(teng_sign)
            #     print(teng_diff.index(min(teng_diff))+1)
            #     print(laplace_diff.index(min(laplace_diff))+1)
            #     plt.plot(teng_vals_norm)
            #     plt.plot(laplace_vals_norm)
            #     plt.show()
            #     plt.clf()

            # find_focus(pic3D_rescale)

            pic_rescale_focus = pic3D_rescale[focal_plane]
            print("Best focused image in stack: {}\n".format(vpipes.three_digs(focal_plane + 1)))


            pic_maxmin = np.max(pic3D_rescale, axis = 0) - np.min(pic3D_rescale, axis = 0)

            marker_locs= vimage.marker_finder(image = pic3D_rescale[focal_plane + 1],
                                                marker = IRISmarker,
                                                thresh = 0.85,
                                                gen_mask = False,
            )
            marker_dict[spot_pass_str] = marker_locs

            # img_rotation = vimage.measure_rotation(marker_dict, spot_pass_str)
            # rotation_dict[spot_pass_str] = img_rotation

            if pass_counter <= 10: overlay_mode = 'series'
            else: overlay_mode = 'baseline'

            mean_shift, overlay_toggle = vimage.measure_shift(marker_dict, pass_num,
                                                              spot_num, mode = overlay_mode
            )
            shift_dict[spot_pass_str] = mean_shift

            # overlay_dict[spot_pass_str] = pic_maxmin

            if (overlay_toggle == True) & (finish_anal not in ('yes', 'y')):
                print("Valid Shift\n")
                # img_overlay = vimage.overlayer(overlay_dict, overlay_toggle, spot_num, pass_num,
                #                                 mean_shift, mode = overlay_mode)
                # if img_overlay is not None:
                #     overlay_name = "{}_overlay_{}".format(img_name, overlay_mode)
                #     vimage.gen_img(img_overlay,
                #                    name = overlay_name,
                #                    savedir = overlay_dir,
                #                    show = False)
            elif (overlay_toggle == False) & (pass_num != 1):
                validity = False
            elif pass_num == 1:
                print("First scan\n")
            else:
                print("Cannot overlay images\n")

            if spot_num in circle_dict:
                spot_coords = circle_dict[spot_num]
                shift_x = spot_coords[0] + mean_shift[1]
                shift_y = spot_coords[1] + mean_shift[0]
                spot_coords = (shift_x, shift_y, spot_coords[2])
                circle_dict[spot_num] = spot_coords
            else:
                spot_coords, pic_canny = vimage.spot_finder(pic_rescale_focus,
                                                            canny_sig = 2.75,
                                                            rad_range=(450,651),
                                                            center_mode = False
                )
                circle_dict[spot_num] = spot_coords

            row, col = np.ogrid[:nrows,:ncols]
            width = col - spot_coords[0]
            height = row - spot_coords[1]
            rad = spot_coords[2] - 25
            disk_mask = (width**2 + height**2 > rad**2)
            full_mask = disk_mask# + marker_mask




            pic_maxmin_masked = np.ma.array(pic_maxmin,
                                            mask = full_mask).filled(fill_value = np.nan)

            # pic3D_rescale_ma = vimage.masker_3D(pic3D_rescale, full_mask, filled = True, fill_val = np.nan)








            ###    COULD BE INTERESTING:
                # f = np.fft.fft2(plane)
                # fshift = np.fft.fftshift(f)
                # mag_spec = 20*np.log(np.abs(fshift))
                #
                # rows, cols = plane.shape
                # crow,ccol = rows//2 , cols//2
                # fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
                # f_ishift = np.fft.ifftshift(fshift)
                # img_back = np.fft.ifft2(f_ishift)
                # img_back = np.abs(img_back)
                # vimage.gen_img(img_back)
                # print(np.max(img_back))

























#*********************************************************************************************#
            if pass_num > 1:

                vshift = int(np.ceil(mean_shift[0]))
                hshift = int(np.ceil(mean_shift[1]))


                if vshift > 0:
                    particle_mask = np.delete(particle_mask, np.s_[-abs(vshift):], axis = 0)
                    particle_mask = np.insert(particle_mask, np.s_[0:abs(vshift)], False, axis = 0)
                elif vshift < 0:
                    particle_mask = np.delete(particle_mask, np.s_[0:abs(vshift)], axis = 0)
                    particle_mask = np.insert(particle_mask, np.s_[-abs(vshift):], False, axis = 0)
                if hshift > 0:
                    particle_mask = np.delete(particle_mask, np.s_[-abs(hshift):], axis = 1)
                    particle_mask = np.insert(particle_mask, np.s_[0:abs(hshift)], False, axis = 1)
                elif hshift < 0:
                    particle_mask = np.delete(particle_mask, np.s_[0:abs(hshift)], axis = 1)
                    particle_mask = np.insert(particle_mask, np.s_[-abs(hshift):], False, axis = 1)
#*********************************************************************************************#





















            with warnings.catch_warnings():
                ##RuntimeWarning ignored: invalid values are expected
                warnings.simplefilter("ignore")
                warnings.warn(RuntimeWarning)
                shapedex = shape_index(pic_rescale_focus)
                shapedex = np.ma.array(shapedex,mask = full_mask).filled(fill_value = np.nan)
                if pass_num > 1:
                    shapedex= np.ma.array(shapedex,mask = particle_mask).filled(fill_value = -1)

            # vimage.gen_img(shapedex)
            shapedex_gauss = ndi.gaussian_filter(shapedex, sigma=1)

            pix_area = np.count_nonzero(np.invert(np.isnan(shapedex)))

            pix_per_um = mag/cam_micron_per_pix
            conv_factor = (cam_micron_per_pix / mag)**2
            area_sqmm = round((pix_area * conv_factor) * 1e-6, 6)


            ridge = 0.5
            sphere = 1
            # trough = -0.5

            maxmin_median = np.nanmedian(pic_maxmin_masked)
            # print(maxmin_median)

            ridge_list = vquant.classify_shape(shapedex, pic_maxmin, ridge,
                                               delta=0.15, intensity=(maxmin_median*4)
            )

            sphere_list = vquant.classify_shape(shapedex, pic_maxmin, sphere,
                                                delta=0.2, intensity=(maxmin_median*2)
            )
            ridge_list_s = vquant.classify_shape(shapedex_gauss, pic_maxmin, 0.5,
                                                 delta=0.2, intensity=(maxmin_median*3)
            )

            ridge_and_sphere_list = ridge_list + sphere_list
            ridge_list_s = [coord for coord in ridge_list_s if coord not in ridge_and_sphere_list]

            pic_binary = np.zeros_like(pic_maxmin, dtype=int)
            for coord in ridge_and_sphere_list + ridge_list_s:
                if pic_binary[coord] == 0: pic_binary[coord] = 1

            pic_binary = ndi.morphology.binary_fill_holes(pic_binary)
            # pic_binary = ndi.morphology.binary_dilation(pic_binary).astype(int)


            # vimage.gen_img(particle_mask)
                # sys.exit()








#*********************************************************************************************#
            pic_binary_label = label(pic_binary, connectivity = 2)
            binary_props = regionprops(pic_binary_label, pic3D_orig[focal_plane], cache = True)

            label_list, centroid_list, area_list, coords_list=[],[],[],[]
            bbox_list, perim_list,elong_list = [],[],[]
            shape_df = pd.DataFrame()
            for region in binary_props:
                if (region.area > 3) & (region.area <= 150):
                    with warnings.catch_warnings():
                        ##UserWarning ignored
                        warnings.simplefilter("ignore")
                        warnings.warn(UserWarning)
                        label_list.append(region.label)
                        coords_list.append([tuple(coords) for coords in region.coords])
                        centroid_list.append(region.centroid)
                        bbox_list.append((region.bbox[0:2], region.bbox[2:]))
                        area_list.append(region.area)
                        perim_list.append(region.perimeter)
                        try: elong_list.append(region.major_axis_length / region.minor_axis_length)
                        except ZeroDivisionError: elong_list.append(np.inf)



            if pass_num == 1:
                particle_mask = ndi.morphology.binary_dilation(pic_binary, iterations=2)
            else:
                particle_mask = np.add(particle_mask,
                                       ndi.morphology.binary_dilation(pic_binary, iterations=2)
                )


#*********************************************************************************************#
            with warnings.catch_warnings():
                ##RuntimeWarning ignored: invalid values are expected
                warnings.simplefilter("ignore")
                warnings.warn(RuntimeWarning)
                perim_area_ratio = vquant.shape_factor_reciprocal(area_list, perim_list)

            shape_df['pass_number'] = [pass_num]*len(label_list)
            shape_df['perim_area_ratio'] = perim_area_ratio

            shape_df['label_bin'] = label_list
            shape_df['coords'] = coords_list
            shape_df['centroid'] = centroid_list
            shape_df['area'] = area_list
            shape_df['elongation'] = elong_list

            median_bg_list, bbox_vert_list = [],[]
            for bbox in bbox_list:
                top_left =  (bbox[0][0]-2, bbox[0][1]-2)
                top_rt   =  (bbox[0][0]-2, bbox[1][1]+2)
                bot_rt   =  (bbox[1][0]+2, bbox[1][1]+2)
                bot_left =  (bbox[1][0]+2, bbox[0][1]-2)
                bbox_verts = np.array([top_left,top_rt,bot_rt,bot_left])
                bbox_vert_list.append(bbox_verts)

            shape_df['bbox_verts'] = bbox_vert_list

            shape_df.reset_index(drop = True, inplace = True)

            filo_pts_tot,round_pts_tot,med_intensity,max_z,greatest_max = [],[],[],[],[]

            for coord_array in shape_df.coords:

                filo_pts = len(set(coord_array).intersection(ridge_list))
                filo_pts_s = len(set(coord_array).intersection(ridge_list_s))
                filo_pts = filo_pts + (filo_pts_s * 0.15)

                round_pts = len(set(coord_array).intersection(sphere_list))

                filo_pts_tot.append(filo_pts)
                round_pts_tot.append(round_pts)

                area_intensity,pix_max_list,z_list = [],[],[]
                for coords in coord_array:
                    y, x = coords[0], coords[1]
                    pixel_stack = pic3D_orig[:,y,x]

                    pix_max, pix_min = np.max(pixel_stack), np.min(pixel_stack)
                    pix_max_list.append(pix_max)

                    z = (np.where(pixel_stack==pix_max)[0][0])+1
                    z_list.append(z)

                    pix_intensity = pix_max - pix_min
                    area_intensity.append(pix_intensity)

                greatest_max.append(np.max(pix_max_list))
                # med_intensity.append(np.median(area_intensity))
                max_z.append(z_list[pix_max_list.index(max(pix_max_list))])

            shape_df['greatest_max'] = greatest_max
            shape_df['max_z'] = max_z
            # shape_df['perc_intensity'] = [val / 655.35 for val in med_intensity]
            shape_df['filo_points'] = filo_pts_tot
            shape_df['round_points'] = round_pts_tot

            median_bg_list, cv_bg_list = [],[]

            for i, bbox in enumerate(shape_df['bbox_verts']):
                plane = shape_df['max_z'].loc[i]-1

                bbox[:,0][np.where(bbox[:,0] >= nrows)] = nrows - 1
                bbox[:,1][np.where(bbox[:,1] >= ncols)] = ncols - 1

                top_edge = pic3D_orig[plane][bbox[0][0],bbox[0][1]:bbox[1][1]+1]
                bottom_edge = pic3D_orig[plane][bbox[1][0]-1,bbox[0][1]:bbox[1][1]+1]
                rt_edge = pic3D_orig[plane][bbox[0][0]:bbox[1][0]+1,bbox[1][1]]
                left_edge = pic3D_orig[plane][bbox[0][0]:bbox[1][0]+1,bbox[0][1]]
                all_edges = np.hstack([top_edge, bottom_edge, rt_edge, left_edge])

                median_bg = np.median(all_edges)
                median_bg_list.append(median_bg)

                cv_bg = np.std(all_edges)/np.mean(all_edges)
                cv_bg_list.append(cv_bg)

            shape_df['median_bg'] = median_bg_list
            shape_df['cv_bg'] = cv_bg_list


            shape_df = shape_df[(shape_df['filo_points'] + shape_df['round_points']) >= 1]

            shape_df['perc_contrast'] = ((shape_df['greatest_max'] - shape_df['median_bg'])*100
                                                    / shape_df['median_bg']
            )
            shape_df = shape_df[(shape_df['filo_points'] + shape_df['round_points']) >= 1]

            shape_df['filo_score'] = ((shape_df['filo_points']/shape_df['area'])
                                     - (shape_df['round_points']/shape_df['area'])
            )
#*********************************************************************************************#
            valid_shape_df = shape_df[shape_df['cv_bg'] < cv_cutoff]

            # particle_count = len(shape_df)

#---------------------------------------------------------------------------------------------#


            # filo_ct = len(valid_shape_df[shape_df.filo_score > 0.2])
            total_particles = len(valid_shape_df.perc_contrast)
            # try: perc_fil = round((filo_ct / (filo_ct + total_particles))*100,2)
            # except ZeroDivisionError: perc_fil=0

            print("Total valid particles counted in {}: {}\n".format(img_name, total_particles))
            print("""
                     #**********************************************
                     ***********************************************#
            """)
            # print("Filaments counted: {}".format(filo_ct))
            # print("Percent filaments: {}\n".format(perc_fil))

            shape_df.reset_index(drop=True, inplace=True)



            # print(whole_spot_df['centroid'][0])
            # if scan_num == 1:
            # whole_spot_df['centroid_shift'] = whole_spot_df.centroid
            shape_df['centroid_shift'] = shape_df['centroid']



            # shape_df.centroid.apply(lambda x: tuple(map(sum,zip(x,cum_mean_shift)
            # )))

            shift_list = [shift
                          for key, shift in shift_dict.items()
                          if spot_num == int(key.split('.')[0])
            ]




            # shape_df.to_csv('{}/{}.vcount.csv'.format(vcount_dir, img_name))

            whole_spot_df = pd.concat([whole_spot_df, shape_df], axis = 0)
            whole_spot_df.reset_index(drop=True, inplace=True)


            vdata_dict= {'image_name'      : img_name,
                         'spot_type'       : spot_type,
                         'area_sqmm'       : area_sqmm,
                         'image_shift_RC'  : mean_shift,
                         'overlay_mode'    : overlay_mode,
                         'total_particles' : total_particles,
                         'focal_plane'     : focal_plane,
                         'exo_toggle'      : exo_toggle,
                         'spot_coords_xyr' : spot_coords,
                         'marker_coords_RC': marker_locs,
                         'valid'           : validity,
                         'VIRAGO_version'  : version

            }
            with open('{}/{}.vdata.txt'.format(vcount_dir,img_name),'w') as f:
                for k,v in vdata_dict.items():
                    f.write('{}: {}\n'.format(k,v))


            # vpipes.write_vdata(vcount_dir, img_name, vdata_vals)

#---------------------------------------------------------------------------------------------#
        ####Processed Image Renderer
            pic_to_show = pic_rescale_focus

            vgraph.gen_particle_image(pic_to_show,whole_spot_df,spot_coords,
                                      pix_per_um=pix_per_um, cv_cutoff=cv_cutoff,
                                      show_particles = False, scalebar = 15,
                                      markers = ''
            )
            plt.savefig('{}/{}.png'.format(img_dir, img_name), dpi = 96)
            plt.clf(); plt.close('all')
#---------------------------------------------------------------------------------------------#
            # particle_df.drop(rounding_cols, axis = 1, inplace = True)
        whole_spot_df.to_csv('{}/{}.v2combined.csv'.format(vcount_dir, spot_ID))
        analysis_time = str(datetime.now() - startTime)


        spot_to_scan += 1
        print("Time to scan PGMs: {}".format(analysis_time))
#*********************************************************************************************#
    info_dict = {'chip_name'      : chip_name,
                 'sample_info'    : sample_name,
                 'spot_number'    : spot_counter,
                 'total_passes'   : pass_counter,
                 'experiment_date': expt_date,
                 'analysis_time'  : analysis_time,
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
for file in info_list:
    version = '.'.join(file.split('_')[-1].split('.')[:3])
print("\nData from version {}\n".format(version))

os.chdir(vcount_dir)
v2combo_list = sorted(glob.glob(chip_name +'*.v2combined.csv'))
vdata_list = sorted(glob.glob(chip_name +'*.vdata.txt'))

if len(vdata_list) >= (len(iris_txt) * pass_counter):
    cont_window_str = str(input("\nEnter the minimum and maximum percent intensity values,"\
                                "separated by a dash.\n"))
    while "-" not in cont_window_str:
        cont_window_str = str(input("\nPlease enter two values separated by a dash.\n"))
    else:
        cont_window = cont_window_str.split("-")

    vdata_dict = vquant.vdata_reader(vdata_list, ['area_sqmm','valid', 'exo_toggle'])
    # spot_df['filo_counts'] = vdata_dict['filo_counts']
    spot_df['area'] = vdata_dict['area_sqmm']
    spot_df['valid'] = vdata_dict['valid']


    particle_counts, filo_counts,irreg_counts = [],[],[]
    particle_dict = {}

    min_cont = float(cont_window[0])
    max_cont = float(cont_window[1])


    new_particle_count, cum_particle_count = [],[]
    for i, csvfile in enumerate(v2combo_list):
        combo_df = pd.read_csv(csvfile, error_bad_lines = False,
                               header = 0, index_col = 0
        )
        if not combo_df.empty:
            combo_df = combo_df[  (combo_df.perc_contrast > min_cont)
                                & (combo_df.perc_contrast <= max_cont)
                                & (combo_df.cv_bg < 0.02)
            ]
            cumulative_particles = 0
            for j in range(1,pass_counter+1):
                scan_df = combo_df[combo_df.pass_number == j]

                csv_id = '{}.{}'.format(csvfile.split(".")[1], vpipes.three_digs(j))
                particle_dict[csv_id] = list(scan_df.perc_contrast)
                particles_per_pass = len(particle_dict[csv_id])

                new_particle_count.append(particles_per_pass)

                cumulative_particles += particles_per_pass
                cum_particle_count.append(cumulative_particles)

                print(  'File scanned: {}; '.format(csvfile)
                      + 'Scan {}, '.format(j)
                      + 'Particles accumulated: {}'.format(cumulative_particles)
                )
        else:
            print("Missing data for {}\n".format(csvfile))
            new_particle_count = new_particle_count + ([0]*pass_counter)
            cum_particle_count = cum_particle_count + ([0]*pass_counter)


    spot_df['new_particles'] = new_particle_count
    spot_df['cumulative_particles'] = cum_particle_count
    spot_df['kparticle_density'] = np.round(cum_particle_count
                                            / spot_df.area.astype(float) * 0.001, 3
    )

    spot_df.loc[spot_df.kparticle_density == 0, 'valid'] = False

    spot_df['normalized_density'] = vquant.density_normalizer(spot_df, spot_counter)

    os.chdir(iris_path)

elif len(vdata_list) != (len(iris_txt) * pass_counter):
    pgms_remaining = total_pgms - len(vcount_csv_list)


spot_df, particle_dict = vquant.spot_remover(spot_df, particle_dict)

vhf_colormap = ('#e41a1c',
                '#377eb8',
                '#4daf4a',
                '#984ea3',
                '#ff7f00',
                '#ffff33',
                '#a65628',
                '#f781bf',
                'gray',
                'black'
)


#*********************************************************************************************#
##HISTOGRAM CODE
if min_cont == 0:
    raw_histogram_df = vgraph.histogrammer(particle_dict, spot_counter, cont_window, baselined = False)
    raw_histogram_df.to_csv('{}/{}_raw_histogram_data.v{}.csv'.format(histo_dir, chip_name, version))

    sum_histogram_df = vgraph.sum_histogram(raw_histogram_df, spot_counter, pass_counter)
    sum_histogram_df.to_csv('{}/{}.sum_histogram_data.v{}.csv'.format(histo_dir, chip_name, version))

    avg_histogram_df = vgraph.average_histogram(sum_histogram_df, mAb_dict_rev, pass_counter)
    avg_histogram_df.to_csv('{}/{}_avg_histogram_data.v{}.csv'.format(histo_dir, chip_name, version))
#*********************************************************************************************#
    """Generates a histogram figure for each pass in the IRIS experiment from a
    DataFrame representing the average data for every spot type"""

    bin_series = avg_histogram_df.pop('bins')
    smooth_histo_df = avg_histogram_df.filter(regex='overline')

    histo_min = int(np.min(np.min(avg_histogram_df)))
    histo_max = int(np.max(np.max(avg_histogram_df)))
    y_minmod = histo_min % 5
    y_maxmod = histo_max % 5
    if y_minmod == 0: y_min = histo_min - 5
    else: y_min = histo_min - y_minmod
    if y_maxmod == 0: y_max = histo_max + 5
    else: y_max = histo_max + y_maxmod
    if abs(y_max - y_min) < 10: y_grid = 1
    else: y_grid = int(np.ceil(abs((y_max - y_min) // 10)/5) * 5)
    if pass_counter < 10: passes_to_show = 1
    else: passes_to_show = pass_counter // 10
    line_settings = dict(lw=2 ,alpha=0.75)
    for i in range(2, pass_counter+1, passes_to_show):
        sns.set(style='ticks')
        c = 0
        for col in smooth_histo_df:
            spot_type = col.split("_")[0]
            pass_num = int(col.split("_")[2][0])

            if pass_num == i:
                plt.plot(bin_series,
                         smooth_histo_df[col],
                         color = vhf_colormap[c],
                         label = spot_type,
                         **line_settings
                )
                c += 1

        plt.title(chip_name+" Pass "+str(i)+" Average Histograms")
        plt.axhline(y=0, ls='dotted', c='black', alpha=0.75)
        plt.legend(loc = 'best', fontsize = 14)
        plt.ylabel("Rolling Average Particle Count", size = 14)

        plt.yticks(range(y_min,y_max+5,y_grid), size = 12)
        plt.xlabel("Percent Contrast", size = 14)
        if (len(bin_series) >= 100) & (len(bin_series) < 200): x_grid = 10
        elif len(bin_series) >= 200: x_grid = 20
        else: x_grid = 5
        plt.xticks(bin_series[::x_grid], size = 12, rotation = 30)

        cont_str = '{0}-{1}'.format(*cont_window)
        figname = ('{}_combohisto_pass_{}_contrast_{}.png'.format(chip_name,i,cont_str))
        plt.savefig('{}/{}'.format(histo_dir,figname), bbox_inches = 'tight', dpi = 150)
        print("File generated: {}".format(figname))
        plt.clf()


normalized_density = vquant.density_normalizer(spot_df, spot_counter)

spot_df['normalized_density'] = normalized_density


averaged_df = vgraph.average_spot_data(spot_df, spot_tuple, pass_counter)

if pass_counter > 2:
    vgraph.generate_timeseries(spot_df, averaged_df, mAb_dict, spot_tuple,
                                chip_name, sample_name, vhf_colormap, cont_window, version,
                                scan_or_time = timeseries_mode, baseline = True,
                                savedir = virago_dir
    )
elif pass_counter <= 2:
    vgraph.generate_barplot(spot_df, pass_counter, cont_window, version,
                            chip_name, sample_name, savedir=virago_dir
    )
spot_df_name='{}/{}_spot_data.{}contrast.v{}.csv'.format(virago_dir,chip_name,cont_window_str,version)
spot_df.to_csv(spot_df_name)
print('File generated: {}'.format(spot_df_name))
