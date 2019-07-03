import numpy as np
import pandas as pd
import warnings
import os

from matplotlib.pyplot import savefig, clf, close
from cv2 import normalize, NORM_MINMAX, HoughCircles, HOUGH_GRADIENT
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation
from skimage.feature import shape_index
from skimage.measure import perimeter
from skimage.io import imread, imread_collection
from sys import stdin


from modules import vpipes, vimage, vquant, vfilo, vgraph


#Feed in pps_list, which is the list of all images of a single spot
def main_loop(pps_list, mirror, params_dict):

    virago_dir = '{}/v3-analysis'.format(os.getcwd())
    vcount_dir = '{}/vcounts'.format(virago_dir)
    img_dir = '{}/processed_images'.format(virago_dir)
    histo_dir = '{}/histograms'.format(virago_dir)
    overlay_dir = '{}/overlays'.format(virago_dir)
    filo_dir = '{}/filo'.format(virago_dir)
    fluor_dir = '{}/fluor'.format(virago_dir)

    if not os.path.exists(virago_dir):
        os.makedirs(virago_dir)
        if not os.path.exists(img_dir): os.makedirs(img_dir)
        if not os.path.exists(histo_dir): os.makedirs(histo_dir)
        if not os.path.exists(fluor_dir): os.makedirs(fluor_dir)
        if not os.path.exists(filo_dir): os.makedirs(filo_dir)
        if not os.path.exists(overlay_dir): os.makedirs(overlay_dir)
        if not os.path.exists(vcount_dir): os.makedirs(vcount_dir)

    cam_micron_per_pix = params_dict['cam_micron_per_pix']
    mag = params_dict['mag']
    pix_per_um = mag / cam_micron_per_pix
    spacing = 1 / pix_per_um
    conv_factor = (cam_micron_per_pix / mag)**2

    exo_toggle = params_dict['exo_toggle']
    cv_cutoff = params_dict['cv_cutoff']

    perc_range = params_dict['perc_range']

    # IRISmarker = imread('/usr3/bustaff/ajdevaux/virago/images/IRISmarker_new.tif')
    IRISmarker = params_dict['IRISmarker']
    # IRISmarker_exo = skio.imread('images/IRISmarker_v4_topstack.tif')


    # pps_list, mirror = vpipes.mirror_finder(pps_list)

    passes_per_spot = len(pps_list)
    spot_ID = pps_list[0][:-8]
    scans_counted = [int(file.split(".")[2]) for file in pps_list]
    first_scan = min(scans_counted)

    circle_dict, marker_dict, overlay_dict, shift_dict = {},{},{},{}

    vdata_dict = vpipes.get_vdata_dict(exo_toggle, version="3.x")

    # missing_data = set(range(1,pass_counter+1)).difference(scans_counted)
    #
    # if missing_data != set():
    #     print("Missing image files... fixing...\n")
    #     for scan in missing_data:
    #         bad_scan = '{0}.{1}'.format(*(spot_ID, str(scan).zfill(3)))
    #         vdata_dict.update({'img_name':bad_scan})
    #
    #         with open('{}/{}.vdata.txt'.format(vcount_dir,bad_scan),'w') as f:
    #             for k,v in vdata_dict.items():
    #                 f.write('{}: {}\n'.format(k,v))
    #         print("Writing blank data files for {}".format(bad_scan))

    total_shape_df = pd.DataFrame()

    for scan in range(0,passes_per_spot):##Main Loop for image processing begins here.
        img_stack = tuple(file for file in pps_list if file.startswith(pps_list[scan]))
        fluor_files = [file for file in img_stack if file.split(".")[-2] in 'ABC']
        if fluor_files:
            img_stack = tuple(file for file in img_stack if file not in fluor_files)
            print("\nFluorescent channel(s) detected: {}\n".format(fluor_files))

        topstack_img = img_stack[0]
        name_split = topstack_img.split('.')
        img_name = '.'.join(name_split[:-1])
        spot_num, pass_num = map(int,name_split[1:3])

        if name_split[-1] == 'tif':
            tiff_toggle = True
        else:
            tiff_toggle = False

        pic3D = vpipes.load_image(img_stack, tiff_toggle)

        print("{} Loaded\n".format(img_name))

        validity = True

    #     if convert_tiff == True:
    #         vpipes.pgm_to_tiff(pic3D, img_name, img_stack,
    #                            tiff_compression=1, archive_pgm=True)
    #
        zslice_count, nrows, ncols = pic3D.shape
        total_pixels = nrows*ncols

        if mirror.size == total_pixels:
            pic3D = pic3D / mirror
            print("Applying mirror to image stack...\n")

        pic3D_norm = np.uint8(normalize(pic3D, None, 0, 255, NORM_MINMAX))

        pic3D_clahe = vimage.cv2_clahe_3D(pic3D_norm, kernel_size=(1,1), cliplim=4)

        pic3D_rescale = vimage.rescale_3D(pic3D_clahe, perc_range=perc_range)

        print("Contrast adjusted\n")
        #Many operations are on the Z-stack compressed image.
        #Several methods to choose, but Standard Deviation works well.
        # maxmin_proj_rescale = np.max(pic3D_rescale, axis = 0) - np.min(pic3D_rescale, axis = 0)
        sd_proj_rescale = np.std(pic3D_rescale, axis=0)
        #Convert to 8-bit for OpenCV comptibility
        sd_proj_rescale = np.uint8(normalize(sd_proj_rescale, None, 0, 255, NORM_MINMAX))

        if pass_num == 1:
            marker = IRISmarker
        else:
            marker = found_markers

        if img_name not in marker_dict:
            marker_locs = vimage.marker_finder(pic3D_rescale[0], marker=marker,  thresh=0.6)
            marker_dict[img_name] = marker_locs
        else:
            marker_locs = marker_dict[img_name]

        pos_plane_list = vquant.measure_focal_plane(pic3D_norm, marker_locs,
                                                    exo_toggle, marker_shape=IRISmarker.shape
        )

        if pos_plane_list != []:
            pos_plane = max(pos_plane_list)
        else:
            pos_plane = zslice_count // 3

        pic_rescale_pos = pic3D_rescale[pos_plane]

        overlay_dict['.'.join(img_name.split('.')[1:])] = sd_proj_rescale

        print("Using image {} from stack\n".format(str(pos_plane + 1).zfill(3)))

        # if pass_counter <= 15:
        #     overlay_mode = 'series'
        # else:
        overlay_mode = 'baseline'

        if pass_num == first_scan:
            print("First Valid Scan\n")
            valid_shift = (0,0)
            overlay_toggle = False

        else:
            prescan_img, postscan_img = vimage._dict_matcher(overlay_dict, spot_num, pass_num, mode=overlay_mode)
            overlay_toggle = True
            # if img_name in shift_dict:
            #     valid_shift = shift_dict[img_name]
            # else:
            ORB_shift = vimage.measure_shift_ORB(prescan_img, postscan_img, ham_thresh=10, show=False)
                # for coord in ORB_shift:
                #     if abs(coord) < 75:
                #
            valid_shift = ORB_shift
                #     else: ##In case ORB fails to give a good value
                #         # overlay_toggle = False
                #         print("Using alternative shift measurement...\n")
                #         mean_shift, overlay_toggle = vimage.measure_shift(marker_dict,pass_num,
                #                                                             spot_num,mode=overlay_mode
                #         )
                #         valid_shift = mean_shift

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
        if spot_num in circle_dict: #Get the location of the Antibody spot
            spot_coords = circle_dict[spot_num]
            shift_x = spot_coords[0] + valid_shift[1]
            shift_y = spot_coords[1] + valid_shift[0]
            spot_coords = (shift_x, shift_y, spot_coords[2])

        else: #Find the Antibody spot if it has not already been determined
            circles = None
            cannyMax = 200
            cannyMin = 100
            while type(circles) == type(None):
                circles = HoughCircles(sd_proj_rescale, HOUGH_GRADIENT,1,minDist=500,
                                           param1=cannyMax, param2=cannyMin,
                                           minRadius=300, maxRadius=600
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

        if pass_num == 1:
            pic_to_show = sd_proj_rescale
        else:
            pic_to_show = img_overlay_difference

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

        # if Ab_spot_mode == True:
            # if exo_toggle == True:
        ridge_thresh   = sd_proj_bg_median*3.5
        sphere_thresh  = sd_proj_bg_median*2.5
        ridge_thresh_s = sd_proj_bg_median*3.5
            # else:
            #     ridge_thresh   = sd_proj_bg_median+sd_proj_bg_stdev*2
            #     sphere_thresh  = sd_proj_bg_median+sd_proj_bg_stdev*2
            #     ridge_thresh_s = sd_proj_bg_median+sd_proj_bg_stdev*3
        # else:
        #     ridge_thresh   = sd_proj_bg_median+sd_proj_bg_stdev*2.75
        #     sphere_thresh  = sd_proj_bg_median+sd_proj_bg_stdev*2.75
        #     ridge_thresh_s = sd_proj_bg_median+sd_proj_bg_stdev*2.75

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
                           # 'spot_type': spot_type,
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

            with open('{}/{}.vdata.txt'.format(vcount_dir,img_name),'w') as f:
                for k,v in vdata_dict.items():
                    f.write('{}: {}\n'.format(k,v))

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

            # if pic3D.ndim > 2:

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

            std_z_stack = list(np.round(np.std(all_z_stacks, axis=0),4))

            max_z_slice_list.append(max_z_slice)
            max_z_stacks.append(max_z_stack)
            z_intensity_list.append(z_intensity)
            greatest_max_list.append(greatest_max)

            if (pass_num > first_scan) & (overlay_toggle == True):
                intensity_increase = max([img_overlay_difference[coords[0],coords[1]] for coords in coord_array])
                intensity_increase_list.append(intensity_increase)
            else:
                intensity_increase_list = [np.nan] * len(shape_df)

        print('N')

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

        # if len(shape_df) > 1:
        #     regression = smapi.OLS(shape_df.z_intensity, shape_df.perc_contrast).fit()
        #     outlier_df = regression.outlier_test()
        #     shape_df.loc[outlier_df['bonf(p)'] < 0.5, 'validity'] = False

        shape_df = vquant.remove_overlapping_objs(shape_df, radius=10)
    #---------------------------------------------------------------------------------------------#
        ##Filament Measurements
        shape_df['circularity'] = list(map(lambda A,P: round((4*np.pi*A)/(perimeter(P)**2),4),
                                                shape_df.area, shape_df.filled_image))

        shape_df['ellipticity'] = round(shape_df.major_axis_length/shape_df.minor_axis_length,4)#max val = 1

        shape_df['eccentricity'] = shape_df.moments_central.map(vquant.eccentricity)

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
                    'eccentricity','ellipticity','curl','circularity',
                     'validity','z_intensity','perc_contrast','cv_bg','sd_above_med_difference',
                     'fiber_length','filo_score','roundness_score','channel','fl_intensity'
                     ]

        vdata_dict.update({'total_valid_particles': total_valid_particles, 'validity':validity})

        with open('{}/{}.vdata.txt'.format(vcount_dir,img_name),'w') as f:
            for k,v in vdata_dict.items():
                f.write('{}: {}\n'.format(k,v))

    #---------------------------------------------------------------------------------------------#
        vgraph.gen_particle_image(pic_to_show,shape_df,spot_coords,
                                  pix_per_um=pix_per_um,
                                  show_particles=True,
                                  cv_cutoff=cv_cutoff,
                                  r2_cutoff=0,
                                  scalebar=15, markers=marker_locs,
                                  exo_toggle=exo_toggle
        )
        savefig('{}/{}.png'.format(img_dir, img_name), dpi = 96)
        clf(); close('all')
        print("#******************PNG generated for {}************************#\n\n".format(img_name))
    #---------------------------------------------------------------------------------------------#
    total_shape_df.to_csv('{}/{}.particle_data.csv'.format(vcount_dir, spot_ID),
                          columns = keep_data
    )
#*********************************************************************************************#
if __name__ == "__main__":

    pps_list = [line.rstrip() for line in stdin]

    xml_list = sorted(glob.glob('*/*.xml'))
    if not xml_list: xml_list = sorted(glob.glob('../*/*.xml'))

    chip_name = pps_list[0].split(".")[0]
    if chip_name == 'baseCHIP404':
        Ab_spot_mode = False
    image_set = set(".".join(file.split(".")[:3]) for file in image_list)

    txtcheck = [file.split(".") for file in txt_list]
    iris_txt = [".".join(file) for file in txtcheck if (len(file) >= 3) and (file[2].isalpha())]

    xml_file = [file for file in xml_list if chip_name in file]
    if xml_file:
        chipFile = vpipes.xml_parser(xml_file[0])
        mAb_dict = vpipes.chipFile_reader(chipFile, remove_jargon = True)
        spot_counter = len([key for key in mAb_dict])##Important
        params_dict = vpipes.get_chip_params(chipFile, Ab_spot_mode)

        main_loop(pps_list, mirror, params_dict)
