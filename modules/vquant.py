from __future__ import division
from future.builtins import input
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import pdist, squareform

from scipy.sparse import csr_matrix, csgraph
from skimage import img_as_float
from skimage.feature import peak_local_max
import pandas as pd
import numpy as np
import itertools as itt
import math, warnings, re, os
from modules import vpipes
#*********************************************************************************************#
#
#           SUBROUTINES
#
#*********************************************************************************************#
def psf_sine(x,a,b,c):
    return a * np.sin(b * x + c)
#*********************************************************************************************#
def classify_shape(shapedex, pic2D, shape, delta, intensity, operator = 'greater'):
    with warnings.catch_warnings():
        ##RuntimeWarning ignored: invalid values are expected
        warnings.simplefilter("ignore")
        warnings.warn(RuntimeWarning)
        if operator == 'greater':
            shape_y, shape_x = np.where((np.abs(shapedex - shape) <= delta)
                                        & (pic2D >= intensity)
            )
        else:
            shape_y, shape_x = np.where((np.abs(shapedex - shape) <= delta)
                                        & (pic2D <= intensity)
            )
    return list(zip(shape_y, shape_x))
#*********************************************************************************************#
def density_normalizer(spot_df, spot_counter):
    """Particle count normalizer so pass 1 = 0 particle density"""
    normalized_density = []
    for x in range(1, spot_counter + 1):
        kp_df = spot_df.kparticle_density[(spot_df.spot_number == x)].reset_index(drop=True)
        j = 0
        while np.isnan(kp_df[j]):
            j += 1
            if j == len(kp_df) - 1:
                norm_val = [np.nan] * j
                break
            print("Invalid data for spot {}, scan {}; normalizing to scan {}".format(x,j,j+1))

        norm_val = [kp_df[i] - kp_df[j] for i in range(0,len(kp_df))]

        normalized_density.append(norm_val)

    return [item for sublist in normalized_density for item in sublist]
#*********************************************************************************************#
def vdata_reader(vdata_list):

    vdata_df = pd.DataFrame()
    for vfile in vdata_list:
        with open(vfile) as vdf:

            props, vals= zip(*[line.split(':') for line in vdf])

            vdata_df = vdata_df.append([vals], ignore_index=True)

    vdata_df = vdata_df.applymap(lambda x: x.strip(' \n'))
    vdata_df.columns = props
    vdata_df['validity'] = vdata_df['validity'].apply(lambda x: eval(x))

    return vdata_df
#
#
#
#              = [line.split(':')[0] for line in vdf]
#                 value = value.strip(' \n')
#                 vdata_dict.update({prop: value})
#                 vdata_df[prop] = value
#
# )
#
#
#
#
#                 if (prop == 'marker_coords_RC') | (prop == 'marker_locs'):
#
#                     coords_list = list(map(int,re.findall('\d+', value)))
#
#                     vdata_dict.update({prop: [tuple(coords_list[i:i+2])
#                                               for i in range(0, len(coords_list), 2)
#                                               if not coords_list == []
#                                               ]
#                     })
#
#                 elif (prop == 'valid') | (prop == 'validity'):
#                     if not value == 'True': value = False
#                     vdata_dict.update({prop: bool(value)})
#                 elif prop in prop_list:
#                     vdata_dict.update({prop: value})
#                 else:
# #                     pass
#
#     return vdata_dict
#*********************************************************************************************#
# def shape_factor_reciprocal(area_list, perim_list):
#     """
#     (Perimeter^2) / 4 * PI * Area).
#     This gives the reciprocal value of Shape Factor for those that are used to using it.
#     A circle will have a value slightly greater than or equal to 1.
#     Other shapes will increase in value.
#     """
#     circ_ratio = 4 * np.pi
#     return [(P**2)/(circ_ratio * A) for A,P in zip(area_list,perim_list)]
#     return roundness
#*********************************************************************************************#
def measure_filo_length(coords, pix_per_um):
    sparse_matrix = csr_matrix(squareform(pdist(coords,metric='euclidean')))
    distances     = csgraph.shortest_path(sparse_matrix,method = 'FW',return_predecessors=False)
    ls_path       = np.max(distances)
    farpoints     = np.where(distances == ls_path)
    filo_len      = float(round(ls_path / pix_per_um, 3))
    vertices      = [coords[farpoints[0][0]],coords[farpoints[0][len(farpoints[0]) // 2]]]

    return filo_len, vertices
#*********************************************************************************************#
def measure_defocus(z_stack, std_z_stack, measure_corr=True,
                    a0=0.1, b0=0.1, c0=1, show = False):
    """
    Measures through the z stack and collects intensity data
    """

    x_data = np.arange(0,len(z_stack))

    maxa, mina = max(z_stack), min(z_stack)
    max_z = z_stack.index(maxa)
    intensity = round(maxa - mina,5)

    basedex = (max_z + z_stack.index(mina)) // 2
    y_data = list(map(lambda x: x - z_stack[basedex], z_stack))

    if measure_corr == False:
        r2 = 0
    else:
        try:
            fit_params, params_cov = curve_fit(psf_sine, x_data, y_data,
                                                sigma=std_z_stack,
                                                p0=[a0, b0, c0],
                                              # bounds=([0.05,0.3,1],[0.4,0.5,1.2]),
                                                method='lm')
        except RuntimeError:
            return intensity, max_z, 0


        y_fit = psf_sine(x_data,fit_params[0],fit_params[1],fit_params[2])

        ss_res = np.sum((y_data - y_fit)**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r2 = round(1 - (ss_res / ss_tot),5)

        if show == True:
            plt.plot(y_fit)
            plt.errorbar(x=x_data,y=y_data,yerr=std_z_stack, linestyle='--')
            plt.text(x=0,y=0, s='R^2 = {0}; a,b,c = {1}'.format(r2,np.round(fit_params,4)))

            plt.xlim(-5,25)
            plt.title("Defocus Curve")
            plt.show()


    return intensity, max_z, r2
#*********************************************************************************************#
def get_bbox_pixels(bbox, max_z_img):
    """
    Collects the bounding box edge pixels for measurements.
    """
    nrows,ncols = max_z_img.shape

    bbox[:,0][np.where(bbox[:,0] >= nrows)] = nrows - 1
    bbox[:,1][np.where(bbox[:,1] >= ncols)] = ncols - 1

    top_row = bbox[0][0]
    bot_row = bbox[2][0]

    lft_col = bbox[0][1]
    rgt_col = bbox[2][1]

    # particle_image = max_z_img[top_row:bot_row+1,lft_col:rgt_col+1]

    top_bot = max_z_img[[top_row, bot_row], lft_col:rgt_col+1]

    lft_rgt = max_z_img[top_row+1:bot_row, [lft_col, rgt_col]]

    return np.concatenate((top_bot.ravel(), lft_rgt.ravel()))
#*********************************************************************************************#
def _overlap_tol(pc_i, pc_j):
    if pc_i == pc_j:
        return 1.0
    else:
        return 1/abs(pc_i - pc_j)
#*********************************************************************************************#
def _intersection_of_smaller(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : ndarray of bounding box vertices in row, column (r,c) format
          bb1[0] is top left coordinates
          bb1[1] is top right coordinates
          bb1[2] is bottom right coordinates
          bb1[3] is bottom left coordinates

    bb2 : ndarray same format as bb1

    Returns
    -------
    The amount the smaller bounding box intersects with the larger bounding box (IoS)
    as a float between 0 and 1
    """
    bb1_r1, bb1_c1 = bb1[0]
    bb1_r2, bb1_c2 = bb1[2]
    assert (bb1_c1 < bb1_c2) & (bb1_r1 < bb1_r2)

    bb2_r1, bb2_c1 = bb2[0]
    bb2_r2, bb2_c2 = bb2[2]
    assert (bb2_c1 < bb2_c2) & (bb2_r1 < bb2_r2)

    # determine the coordinates of the intersection rectangle
    col_left = max(bb1_c1, bb2_c1)
    row_top = max(bb1_r1, bb2_r1)
    col_right = min(bb1_c2, bb2_c2)
    row_bottom = min(bb1_r2, bb2_r2)

    if (col_right < col_left) | (row_bottom < row_top):
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (col_right - col_left) * (row_bottom - row_top)

    # compute the area of both AABBs
    bb1_area = (bb1_c2 - bb1_c1) * (bb1_r2 - bb1_r1)
    bb2_area = (bb2_c2 - bb2_c1) * (bb2_r2 - bb2_r1)

    #determine which bounding box is smaller
    #then divide by the smaller bounding box to get the amount
    #the smaller bounding box is enveloped by the larger one
    if bb1_area > bb2_area:
        ios = intersection_area / bb2_area
    else:
        ios = intersection_area / bb1_area


    assert (ios >= 0.0) & (ios <= 1.0)

    return ios
#*********************************************************************************************#
def mark_overlaps(neighbor_tree_dist, shape_df):
    pc_series = shape_df.perc_contrast
    bbox_series = shape_df.bbox_verts

    overlap_ix_list = []

    for i,j in neighbor_tree_dist:
        pc_i, pc_j = pc_series[[i,j]]
        bb_i, bb_j = bbox_series[[i,j]]

        overlap = _overlap_tol(pc_i, pc_j)
        ios = _intersection_of_smaller(bb_i, bb_j)

        if ios >= overlap:
            # print(iou, overlap)
            if pc_i < pc_j:
                overlap_ix_list.append(i)
            else:
                overlap_ix_list.append(j)

    return overlap_ix_list
#*********************************************************************************************#
def spot_remover(spot_df, particle_dict, vdata_list, vcount_dir, iris_path, quarantine_img = False):
    excise_toggle = input("Would you like to remove any spots from the analysis? (y/[n])\t")
    assert isinstance(excise_toggle, str)
    if excise_toggle.lower() in ('y','yes'):
        excise_spots = input("Which spots? (Separate all spot numbers by a comma)\t")
        excise_spots = excise_spots.split(",")
        for ex_spot in excise_spots:
            spot_df.loc[spot_df.spot_number == int(ex_spot), 'validity'] = False

            for key in particle_dict.keys():
                spot_num_dict = int(key.split(".")[0])
                if spot_num_dict == int(ex_spot):
                    particle_dict[key] = 0

            os.chdir(vcount_dir)
            for vfile in vdata_list:
                spot_num_vfile = int(vfile.split('.')[1])
                if spot_num_vfile == int(ex_spot):
                    print(vfile)
                    old_vfile = vfile+'~'
                    os.rename(vfile, old_vfile)

                    with open(old_vfile, 'r+') as vf_old, open(vfile, 'w') as vf_new:
                        lines = vf_old.readlines()
                        # print(lines)
                        for line in lines:
                            if line.split(':')[0] == 'validity':
                                # print(line)
                                newline = 'validity: False\n'
                                vf_new.write(newline)

                            else:
                                vf_new.write(line)
                    os.remove(old_vfile)

            if quarantine_img == True:
                os.chdir(iris_path)
                if not os.path.exists('bad_imgs'): os.makedirs('bad_imgs')
                bad_pgms = sorted(glob.glob('*.'+vpipes.three_digs(ex_spot)+'.*.*.pgm'))
                for pgm in bad_pgms:
                    os.rename(pgm, "{}/bad_imgs/{}".format(iris_path, pgm))


    return spot_df, particle_dict
#*********************************************************************************************#
