simport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, csgraph
from skimage.feature import shape_index
from skimage.draw import circle
from skimage.measure import label, regionprops, points_in_poly
from skimage.filters import threshold_local,threshold_otsu, rank, threshold_sauvola
from skimage.morphology import medial_axis
from skimage import io as skio
import os, glob, warnings
from skimage.morphology import disk
from modules import vpipes, vimage, vquant, vgraph


os.chdir('/Volumes/KatahdinHD/ResilioSync/DATA/pydata/virago/images/')
IRISmarker = skio.imread('IRISmarker_maxmin_v5.tif')
os.chdir('/Volumes/KatahdinHD/ResilioSync/DATA/IRIS/FIGS4PAPER/filo_images')
os.chdir('t4')

pgm_list = sorted(glob.glob('*.pgm'))
chip_name = pgm_list[0].split(".")[0]

pgm_list, mirror = vpipes.mirror_finder(pgm_list)

scan_set = set(['.'.join(fn.split('.')[:3]) for fn in pgm_list])
for img_name in scan_set:
    if not os.path.exists(img_name):os.makedirs(img_name)
    if not os.path.exists('{}/filo'.format(img_name)): os.makedirs('{}/filo'.format(img_name))
    scan_list = [file for file in pgm_list if file.startswith(img_name)]

    pic3D = np.array([pic for pic in skio.imread_collection(scan_list)], dtype='uint16')
    zslice_count, nrows, ncols = pic3D.shape
    pic3D_orig = pic3D.copy()

    cam_micron_per_pix, mag, exo_toggle = vpipes.determine_IRIS(nrows, ncols)
    pix_per_um = mag / cam_micron_per_pix

    pic3D = pic3D / mirror
    print("Applying mirror to images...\n")

    pic3D_norm = pic3D / (np.median(pic3D) * 2)

    pic3D_norm[pic3D_norm > 1] = 1

    pic3D_clahe = vimage.clahe_3D(pic3D_norm, cliplim = 0.004)##UserWarning silenced

    pic3D_rescale = vimage.rescale_3D(pic3D_clahe, perc_range = (3,97))
    print("Contrast adjusted\n")

    if zslice_count > 1: focal_plane = int(np.floor(zslice_count/2))
    else: focal_plane = 0

    pic_maxmin = np.max(pic3D_rescale, axis = 0) - np.min(pic3D_rescale, axis = 0)

    marker_locs = vimage.marker_finder(image = pic_maxmin,
                                        marker = IRISmarker,
                                        thresh = 0.88,
                                        gen_mask = False
    )

    xyr, pic_canny = vimage.spot_finder(pic_maxmin, canny_sig = 4, rad_range=(450,651))

    row, col = np.ogrid[:nrows,:ncols]
    width = col - xyr[0]
    height = row - xyr[1]
    rad = xyr[2] - 50
    disk_mask = (width**2 + height**2 > rad**2)
    full_mask = disk_mask# + marker_mask

#*********************************************************************************************#
    pic_maxmin_masked = np.ma.array(pic_maxmin, mask = full_mask).filled(fill_value = np.nan)

    with warnings.catch_warnings():
        ##RuntimeWarning ignored: invalid values are expected
        warnings.simplefilter("ignore")
        warnings.warn(RuntimeWarning)
        shapedex = shape_index(pic3D_rescale[5])
        shapedex = np.ma.array(shapedex,mask = full_mask).filled(fill_value = np.nan)
        # if pass_num > 1:
        #     shapedex= np.ma.array(shapedex,mask = particle_mask).filled(fill_value = -1)

    # vimage.gen_img(shapedex)
    shapedex_gauss = ndi.gaussian_filter(shapedex, sigma=1)

    pix_area = np.count_nonzero(np.invert(np.isnan(shapedex)))
    pix_per_um = mag/cam_micron_per_pix
    conv_factor = (cam_micron_per_pix / mag)**2
    area_sqmm = round((pix_area * conv_factor) * 1e-6, 6)



    # def classify_shape(shapedex, pic2D, shape, delta, intensity = 0.55, operator = 'greater'):
    #     with warnings.catch_warnings():
    #         ##RuntimeWarning ignored: invalid values are expected
    #         warnings.simplefilter("ignore")
    #         warnings.warn(RuntimeWarning)
    #         if operator == 'greater':
    #             shape_y, shape_x = np.where((np.abs(shapedex - shape) <= delta)
    #                                         & (pic2D >= intensity)
    #             )
    #         else:
    #             shape_y, shape_x = np.where((np.abs(shapedex - shape) <= delta)
    #                                         & (pic2D <= intensity)
    #             )
    #     return list(zip(shape_y, shape_x))
    ridge = 0.5
    sphere = 1
    # trough = -0.5

    maxmin_median = np.nanmedian(pic_maxmin)
    print(maxmin_median)

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

    pic_binary = ndi.morphology.binary_fill_holes(pic_binary).astype(int)
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

#*********************************************************************************************#
    with warnings.catch_warnings():
        ##RuntimeWarning ignored: invalid values are expected
        warnings.simplefilter("ignore")
        warnings.warn(RuntimeWarning)
        perim_area_ratio = vquant.shape_factor_reciprocal(area_list, perim_list)

    # shape_df['pass_number'] = [pass_num]*len(label_list)
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
    shape_df.reset_index(drop=True, inplace=True)
#*********************************************************************************************#
    filo_df = shape_df[(shape_df.filo_score >= 0.25) & (shape_df.area > 9)]
    filo_lengths, vertices = [],[]
    for ix in filo_df.index:
        coords = filo_df.coords[ix]
        box = filo_df.bbox_verts[ix]
        # skel = np.argwhere(medial_axis(pic_binary[box[0][0]:box[2][0],box[0][1]:box[2][1]]) == True)

        sparse_matrix = csr_matrix(squareform(pdist(coords, metric='cityblock')))
        distances = csgraph.shortest_path(sparse_matrix,
                                          method = 'FW',
                                          return_predecessors=False
        )
        ls_path = np.max(distances)
        farpoints = np.where(distances == ls_path)

        filo_lengths.append(float(round(ls_path / pix_per_um, 3)))
        vertices.append([coords[farpoints[0][0]],coords[farpoints[0][len(farpoints[0]) // 2]]])
        filo_pic = pic_maxmin[box[0][0]:box[2][0],box[0][1]:box[2][1]]
        # vimage.gen_img(filo_pic,name = filo_df.label_bin[ix], savedir = '{}/filo'.format(img_name), show=False)


    filo_df['filo_lengths'] = filo_lengths
    filo_df['vertices'] = vertices

    shape_df = pd.concat([shape_df, filo_df[['filo_lengths','vertices']]],axis=1)
    # vimage.gen_img(pic_skels)
    shape_df.to_csv('{}/shape_df.csv'.format(img_name))
#*********************************************************************************************#


    #
    #
    # with warnings.catch_warnings():
    #     ##RuntimeWarning ignored: invalid values are expected
    #     warnings.simplefilter("ignore")
    #     warnings.warn(RuntimeWarning)
    #     perim_area_ratio = shape_factor_reciprocal(area_list, perim_list)
    #
    # shape_df['perim_area_ratio'] = perim_area_ratio
    #
    # # shape_df.reset_index(drop = True, inplace = True)
    #
    # filo_pts_tot, round_pts_tot, med_intensity, max_z, greatest_max = [],[],[],[],[]
    # for coord_array in shape_df.coords:
    #
    #     filo_pts = len(set(coord_array).intersection(ridge_list))
    #     # filo_pts_s = len(set(coord_array).intersection(ridge_list_s))
    #     # filo_pts = filo_pts + (filo_pts_s * 0.15)
    #
    #     round_pts = len(set(coord_array).intersection(sphere_list))
    #
    #     filo_pts_tot.append(filo_pts)
    #     round_pts_tot.append(round_pts)
    #
    #     area_intensity,pix_max_list,z_list = [],[],[]
    #     for coords in coord_array:
    #         y, x = coords[0], coords[1]
    #         pixel_stack = pic3D_orig[:,y,x]
    #
    #         pix_max, pix_min = np.max(pixel_stack), np.min(pixel_stack)
    #         pix_max_list.append(pix_max)
    #
    #         z = (np.where(pixel_stack==pix_max)[0][0])+1
    #         z_list.append(z)
    #
    #         pix_intensity = pix_max - pix_min
    #         area_intensity.append(pix_intensity)
    #
    #     greatest_max.append(np.max(pix_max_list))
    #     med_intensity.append(np.median(area_intensity))
    #     max_z.append(z_list[pix_max_list.index(max(pix_max_list))])
    # shape_df['greatest_max'] = greatest_max
    # shape_df['max_z'] = max_z
    # shape_df['perc_intensity'] = [val / 655.35 for val in med_intensity]
    # shape_df['filo_points'] = filo_pts_tot
    # shape_df['round_points'] = round_pts_tot
    #
    # median_bg_list, cv_bg_list = [],[]
    # for i, bbox in enumerate(shape_df['bbox_verts']):
    #
    #     plane = shape_df['max_z'].loc[i]-1
    #     top_edge = pic3D_orig[plane][bbox[0][0],bbox[0][1]:bbox[1][1]+1]
    #     bottom_edge = pic3D_orig[plane][bbox[1][0]-1,bbox[0][1]:bbox[1][1]+1]
    #     rt_edge = pic3D_orig[plane][bbox[0][0]:bbox[1][0]+1,bbox[1][1]]
    #     left_edge = pic3D_orig[plane][bbox[0][0]:bbox[1][0]+1,bbox[0][1]]
    #     all_edges = np.hstack([top_edge, bottom_edge, rt_edge, left_edge])
    #
    #     median_bg = np.median(all_edges)
    #     median_bg_list.append(median_bg)
    #
    #     cv_bg = np.std(all_edges)/np.mean(all_edges)
    #     cv_bg_list.append(cv_bg)
    #
    # shape_df['median_bg'] = median_bg_list
    # shape_df['cv_bg'] = cv_bg_list
    #
    #
    # shape_df = shape_df[(shape_df['filo_points'] + shape_df['round_points']) >= 1]
    #
    # shape_df['perc_contrast'] = ((shape_df['greatest_max'] - shape_df['median_bg'])*100
    #                                         / shape_df['median_bg']
    # )
    # shape_df = shape_df[(shape_df['filo_points'] + shape_df['round_points']) > 1]
    #
    # shape_df['filo_points']-(shape_df['round_points']*1.5)
    cv_cutoff = 0.02
    valid_shape_df = shape_df[shape_df['cv_bg'] < cv_cutoff]
    #
    # shape_df['filo_score'] = ((shape_df['filo_points']/shape_df['area'])
    #                          -(shape_df['round_points']/shape_df['area'])
    # )
    # shape_df.reset_index(drop=True, inplace=True)
    #
    #
    # filo_df = shape_df[(shape_df.filo_score >= 0.25) & (shape_df.area > 9)]
    # filo_lengths, vertices = [],[]
    # for ix in filo_df.index:
    #     coords = filo_df.coords[ix]
    #     box = filo_df.bbox_verts[ix]
    #     # skel = np.argwhere(medial_axis(pic_binary[box[0][0]:box[2][0],box[0][1]:box[2][1]]) == True)
    #
    #     sparse_matrix = csr_matrix(squareform(pdist(coords, metric='cityblock')))
    #     distances = csgraph.shortest_path(sparse_matrix,
    #                                       method = 'FW',
    #                                       return_predecessors=False
    #     )
    #     ls_path = np.max(distances)
    #     farpoints = np.where(distances == ls_path)
    #
    #     filo_lengths.append(float(round(ls_path / pix_per_um, 3)))
    #     vertices.append([coords[farpoints[0][0]],coords[farpoints[0][len(farpoints[0]) // 2]]])
    #     filo_pic = pic_maxmin[box[0][0]:box[2][0],box[0][1]:box[2][1]]
    #     # vimage.gen_img(filo_pic,name = filo_df.label_bin[ix], savedir = '{}/filo'.format(img_name), show=False)
    #
    #
    # filo_df['filo_lengths'] = filo_lengths
    # filo_df['vertices'] = vertices
    #
    # shape_df = pd.concat([shape_df, filo_df[['filo_lengths','vertices']]],axis=1)
    # # vimage.gen_img(pic_skels)
    # shape_df.to_csv('{}/shape_df.csv'.format(img_name))


    vgraph.gen_particle_image(pic_maxmin,valid_shape_df,spot_coords,
                              pix_per_um=pix_per_um, cv_cutoff=cv_cutoff,
                              show_particles = False, scalebar = 15,
    )
    plt.savefig('{}/{}.png'.format(img_dir, img_name), dpi = 96)
    plt.clf(); plt.close('all')

    print("Analysis Complete")
#*********************************************************************************************#
    fig = plt.figure(figsize=(24, 8))
    ax1 = fig.add_subplot(1, 3, 1)

    ax1.imshow(pic_maxmin, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('Maxmin Compressed Image', fontsize=18)
    patch_settings = dict(fill=False, linewidth=1, alpha = 0.75)
    line_settings = dict(lw=1,color='purple',alpha=0.6)
    scatter_settings = dict(s=18, linewidths=0)


    for val in shape_df.index.values:
        filo_score = shape_df.filo_score.loc[val]
        perim_area_ratio = shape_df.perim_area_ratio.loc[val]
        elong = shape_df.elongation.loc[val]
        verts = shape_df.vertices[val]
        box = shape_df.bbox_verts.loc[val]

        if (filo_score > 0.2):
            color = 'r'
        elif (filo_score <= 0.1):
            color = 'c'
        # elif (filo_score > 0.2) & (perim_area_ratio <= 1):
        #     color = 'white'
        else: color = 'y'

        low_left_xy   =   (box[3][1], box[3][0])
        # up_left_xy  =   (box[0][1], box[0][0])
        # low_rt_xy   =   (box[2][1], box[2][0])
        # up_rt_xy    =   (box[1][1], box[1][0])
        if shape_df.cv_bg.loc[val] >= cv_cutoff:
            line1 = lines.Line2D([box[3][1],box[1][1]],[box[3][0],box[1][0]], **line_settings)
            line2 = lines.Line2D([box[0][1],box[2][1]],[box[0][0],box[2][0]], **line_settings)
            ax1.add_line(line1)
            ax1.add_line(line2)
        elif not np.isnan(verts).any():
            ax1.scatter(verts[0][1],verts[0][0], color = 'red', marker = '.')
            ax1.scatter(verts[1][1],verts[1][0], color = 'magenta', marker = '+')
        else:
            # ec = 'green'
            # h = box[0][0] - box[2][0]
            # w = box[1][1] - box[0][1]
            # filobox = plt.Rectangle(low_left_xy, w, h, ec = ec, **patch_settings)

            centroid = shape_df.centroid.loc[val]
            round_circ = plt.Circle((centroid[1], centroid[0]),
                                     shape_df.perc_contrast.loc[val]*3,
                                     color=color, **patch_settings
            )
            # ax1.add_patch(filobox)
            ax1.add_patch(round_circ)

    ax2 = fig.add_subplot(1, 3, 3, sharex=ax1, sharey=ax1)
    ax2.imshow(pic_binary, plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Scored Binary Image', fontsize = 18)

    ridge_y_s, ridge_x_s = zip(*ridge_list_s)
    ax2.scatter(ridge_x_s,ridge_y_s, color='orange', **scatter_settings, marker = '^')

    ridge_y,ridge_x = zip(*ridge_list)
    ax2.scatter(ridge_x,ridge_y, color='m', **scatter_settings)

    sphere_y, sphere_x = zip(*sphere_list)
    ax2.scatter(sphere_x, sphere_y, color='cyan', **scatter_settings)

    # trough_y,trough_x = zip(*trough_list)
    # ax2.scatter(trough_x, trough_y, color='blue', **scatter_settings,marker = '^',alpha=0.5)

    ax3 = fig.add_subplot(1, 3, 2, sharex=ax1, sharey=ax1)

    ax3.imshow(shapedex, cmap=plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('Shape Index', fontsize=18)

    fig.tight_layout()
    plt.show()
    plt.clf()
