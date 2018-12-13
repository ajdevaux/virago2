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
def vdata_reader(vdata_list, prop_list):
    vdata_dict = {}
    for prop in prop_list:
        d_list = []
        for file in vdata_list:
            full_text = {}
            with open(file) as f:
                for line in f:

                    full_text[line.split(":")[0]] = line.split(":")[1].strip(' \n')
                data = full_text[prop]

            if prop == 'marker_coords_RC':
                coords_list = list(map(int,re.findall('\d+', data)))
                if not coords_list == []:
                    data = [tuple(coords_list[i:i+2]) for i in range(0, len(coords_list), 2)]
            elif prop == 'valid':
                if data == 'True': data = True
                else: data = False

            d_list.append(data)

        vdata_dict[prop] = d_list

    return vdata_dict
#*********************************************************************************************#
def shape_factor_reciprocal(area_list, perim_list):
    """
    (Perimeter^2) / 4 * PI * Area).
    This gives the reciprocal value of Shape Factor for those that are used to using it.
    A circle will have a value slightly greater than or equal to 1.
    Other shapes will increase in value.
    """
    circ_ratio = 4 * np.pi
    return [(P**2)/(circ_ratio * A) for A,P in zip(area_list,perim_list)]
    return roundness
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
def measure_max_intensity_stack(pic3D, coord_array):
    pix_max_list, z_list = [],[]
    for coords in coord_array:
        pixel_stack = pic3D[:, coords[0], coords[1]]

        pix_max = np.max(pixel_stack)
        # pix_min = np.min(pixel_stack)
        pix_max_list.append(pix_max)

        z_list.append((np.where(pixel_stack == pix_max)[0][0]) + 1)

    return pix_max_list, z_list
#*********************************************************************************************#
def spot_remover(spot_df, particle_dict, vdata_list, vcount_dir, iris_path, quarantine_img = False):
    excise_toggle = input("Would you like to remove any spots from the analysis? (y/[n])\t")
    assert isinstance(excise_toggle, str)
    if excise_toggle.lower() in ('y','yes'):
        excise_spots = input("Which spots? (Separate all spot numbers by a comma)\t")
        excise_spots = excise_spots.split(",")
        for ex_spot in excise_spots:
            spot_df.loc[spot_df.spot_number == int(ex_spot), 'valid'] = False

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
                            if line.split(':')[0] == 'valid':
                                # print(line)
                                newline = 'valid: False\n'
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
