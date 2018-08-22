from __future__ import division
from future.builtins import input
from scipy.ndimage.filters import gaussian_filter
from skimage import img_as_float
from skimage.feature import peak_local_max
import pandas as pd
import numpy as np
import itertools as itt
import math, warnings, re

#*********************************************************************************************#
#
#           SUBROUTINES
#
#*********************************************************************************************#
def blob_detect_3D(image_stack, min_sig, max_sig, ratio = 1.6, thresh = 0.5, image_list = ""):
    """This is the primary function for detecting "blobs" in the stack of IRIS images.
    Uses the Difference of Gaussians algorithm"""

    def _blob_overlap(blob1, blob2):
        """Finds the overlapping area fraction between two blobs.
        Returns a float representing fraction of overlapped area.
        Parameters
        ----------
        blob1 : sequence
            A sequence of ``(y,x,sigma)``, where ``x,y`` are coordinates of blob
            and sigma is the standard deviation of the Gaussian kernel which
            detected the blob.
        blob2 : sequence
            A sequence of ``(y,x,sigma)``, where ``x,y`` are coordinates of blob
            and sigma is the standard deviation of the Gaussian kernel which
            detected the blob.
        Returns
        -------
        f : float
            Fraction of overlapped area.
        """
        root2 = math.sqrt(2)

        # extent of the blob is given by sqrt(2)*scale
        r1 = blob1[2] * root2
        r2 = blob2[2] * root2

        d = math.hypot(blob1[0] - blob2[0], blob1[1] - blob2[1])

        if d > r1 + r2:
            return 0

        # one blob is inside the other, the smaller blob must die
        if d <= abs(r1 - r2):
            return 1

        ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
        ratio1 = np.clip(ratio1, -1, 1)
        acos1 = np.arccos(ratio1)

        ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
        ratio2 = np.clip(ratio2, -1, 1)
        acos2 = np.arccos(ratio2)

        a = -d + r2 + r1
        b = d - r2 + r1
        c = d + r2 - r1
        d = d + r2 + r1
        area = r1 ** 2 * acos1 + r2 ** 2 * acos2 - 0.5 * math.sqrt(abs(a * b * c * d))

        return area / (math.pi * (min(r1, r2) ** 2))

    def _prune_blobs(blobs_array, overlap):
        """Eliminated blobs with area overlap.
        Parameters
        ----------
        blobs_array : ndarray
            A 2d array with each row representing 3 values, ``(y,x,sigma)``
            where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
            standard deviation of the Gaussian kernel which detected the blob.
        overlap : float
            A value between 0 and 1. If the fraction of area overlapping for 2
            blobs is greater than `overlap` the smaller blob is eliminated.
        Returns
        -------
        A : ndarray
            `array` with overlapping blobs removed.
        """
        # iterating again might eliminate more blobs, but one iteration suffices
        # for most cases
        for blob1, blob2 in itt.combinations(blobs_array, 2):
            if _blob_overlap(blob1, blob2) > overlap:
                if blob1[2] > blob2[2]:
                    blob2[2] = -1
                else:
                    blob1[2] = -1

        # return blobs_array[blobs_array[:, 2] > 0]
        return np.array([b for b in blobs_array if b[2] > 0])


    def blob_dog(image, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=2.0,
                     overlap=.5):


        image = img_as_float(image)

        # k such that min_sigma*(sigma_ratio**k) > max_sigma
        k = int(math.log(float(max_sigma) / min_sigma, sigma_ratio)) + 1
        # a geometric progression of standard deviations for gaussian kernels
        sigma_list = np.array([min_sigma * (sigma_ratio ** i) for i in range(k + 1)])
        print(sigma_list)
        gaussian_images = [gaussian_filter(image, s) for s in sigma_list]

        # computing difference between two successive Gaussian blurred images
        # multiplying with standard deviation provides scale invariance
        dog_images = [(gaussian_images[i] - gaussian_images[i + 1])
                      * sigma_list[i] for i in range(k)
        ]

        image_cube = np.stack(dog_images, axis=-1)

        # local_maxima = get_local_maxima(image_cube, threshold)
        local_maxima = peak_local_max(image_cube, threshold_abs=threshold,
                                      footprint=np.ones((3,) * (image.ndim + 1)),
                                      threshold_rel=0.0,
                                      exclude_border=False
        )
        # Catch no peaks
        if local_maxima.size == 0:
            return np.empty((0, 3))
        # Convert local_maxima to float64
        lm = local_maxima.astype(np.float64)
        # Convert the last index to its corresponding scale value
        lm[:, -1] = sigma_list[local_maxima[:, -1]]
        return _prune_blobs(lm, overlap)


    total_blobs = np.empty(shape = (0,4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn(RuntimeWarning)
        for plane, image in enumerate(image_stack):
            blobs = blob_dog(image, min_sigma = min_sig, max_sigma = max_sig,
                                     sigma_ratio = ratio, threshold = thresh, overlap = 0.5)

            if len(blobs) == 0:
                print("No blobs here")
                blobs = np.zeros(shape = (1,4))
            else:
                z_arr = np.full((len(blobs),1), int(plane+1))
                blobs = np.append(blobs,z_arr, axis = 1)

            total_blobs = np.append(total_blobs, blobs, axis = 0)
            total_blobs = total_blobs.astype(int, copy = False)
            print("Image scanned: {}".format(image_list[plane]))

    return total_blobs
#*********************************************************************************************#
def particle_quant_3D(image_stack, d_blobs, cv_thresh = 0.1):
    """This measures the percent contrast for every detected blob in the stack
    and filters out blobs that are on edges by setting a cutoff for standard deviation of the mean
     for measured background intensity. Blobs are now considered "particles" """
    perc_contrast, cv_background, norm_intensity, coords_yx = [],[],[],[]
    sqrt_2 = math.sqrt(2)
    for i, blob in enumerate(d_blobs):
        y, x, sigma, z_name = d_blobs[i]
        r = int(np.ceil(sigma * sqrt_2))
        if r < 3: r = 3
        z_loc = z_name-1

        point_lum = image_stack[z_loc , y , x]
        stack_lum = image_stack[:, y, x]
        norm_int = max(stack_lum) - min(stack_lum)

        local = image_stack[z_loc , y-(r):y+(r+1) , x-(r):x+(r+1)]

        try: local_circ = np.hstack([local[0,1:-1],local[:,0],local[-1,1:-1],local[:,-1]])
        except IndexError:
            local = np.full([r+1,r+1], point_lum)
            local_circ = np.hstack([local[0,1:-1],local[:,0],local[-1,1:-1],local[:,-1]])

        cv_bg = np.std(local_circ)/np.mean(local_circ)
        bg_val = np.median(local_circ)

        perc_contrast_pt = ((point_lum - bg_val) * 100) / bg_val
        perc_contrast.append(perc_contrast_pt)
        cv_background.append(cv_bg)
        coords_yx.append((y,x))
        norm_intensity.append(norm_int)

    particle_df = pd.DataFrame(data = d_blobs, columns = ['y','x','sigma','z'])
    particle_df['pc'] = perc_contrast
    particle_df['cv_bg'] = cv_background
    particle_df['coords_yx'] = coords_yx
    particle_df['norm_int'] = norm_intensity

    particle_df = particle_df[particle_df.pc > 0]
    particle_df = particle_df[particle_df.cv_bg <= cv_thresh]
    particle_df.reset_index(drop = True, inplace = True)

    if len(particle_df) == 0:
        particle_df = pd.DataFrame(data = [[0,0,0,0,0,0,0,0]],
                                   columns = ['y','x','sigma','z','pc','cv_bg','norm_int','coords_yx'])

    return particle_df
#*********************************************************************************************#
def coord_rounder(DFrame, val = 10):
    """Identifies duplicate coordinates for particles, which inevitably occurs in multi-image stacks"""
    DFrame2 = DFrame.copy(deep=False)
    xrd = (DFrame.x/val).round()*val
    yrd = (DFrame.y/val).round()*val
    xceil = np.ceil(DFrame.x/val)*val
    yceil = np.ceil(DFrame.y/val)*val
    xfloor = np.floor(DFrame.x/val)*val
    yfloor = np.floor(DFrame.y/val)*val

    DFrame2['yx_'+str(val)] = pd.Series(list(zip(yrd,xrd)))
    DFrame2['yx_cc'] = pd.Series(list(zip(yceil,xceil)))
    DFrame2['yx_ff'] = pd.Series(list(zip(yfloor,xfloor)))
    DFrame2['yx_cf'] = pd.Series(list(zip(yceil,xfloor)))
    DFrame2['yx_fc'] = pd.Series(list(zip(yfloor,xceil)))
    rounding_cols = ['yx_'+str(val),'yx_cc','yx_ff','yx_cf','yx_fc']
    return DFrame2, rounding_cols
#*********************************************************************************************#
def dupe_dropper(DFrame, rounding_cols, sorting_col):
    """Removes duplicate particles while keeping the highest contrast particle for each duplicate"""
    DFrame.sort_values([sorting_col], kind = 'quicksort', inplace = True)
    for column in rounding_cols:
        DFrame.drop_duplicates(subset = (column), keep = 'last', inplace = True)
    DFrame.reset_index(drop = True, inplace = True)
    # DFrame.drop(columns = rounding_cols, inplace = True)
    return DFrame
#*********************************************************************************************#
def vir_csv_reader(csv_list, cont_window):
    """Reads the csvs generated by VIRAGO to determine final counts based on contrast window"""
    particle_list = ([])
    particle_dict = {}

    min_cont = float(cont_window[0])
    max_cont = float(cont_window[1])
    for csvfile in csv_list:
        csv_df = pd.read_csv(csvfile, error_bad_lines = False, header = 0)

        kept_vals = [val for val in csv_df.pc if min_cont < val <= max_cont]
        val_count = len(kept_vals)
        csv_info = csvfile.split(".")
        csv_id = '{}.{}'.format(csv_info[1],csv_info[2])
        particle_dict[csv_id] = kept_vals
        particle_list.append(val_count)
        print('File scanned:  '+ csvfile + '; Particles counted: ' + str(val_count))
    return particle_list, particle_dict

#*********************************************************************************************#
def vir_csv_reader(csv_list, cont_window):
    """Reads the csvs generated by VIRAGO to determine final counts based on contrast window"""
    particle_list = ([])
    particle_dict = {}

    min_cont = float(cont_window[0])
    max_cont = float(cont_window[1])
    for csvfile in csv_list:
        csv_df = pd.read_csv(csvfile, error_bad_lines = False, header = 0)

        kept_vals = [val for val in csv_df.pc if min_cont < val <= max_cont]
        val_count = len(kept_vals)
        csv_info = csvfile.split(".")
        csv_id = '{}.{}'.format(csv_info[1],csv_info[2])
        particle_dict[csv_id] = kept_vals
        particle_list.append(val_count)
        print('File scanned:  '+ csvfile + '; Particles counted: ' + str(val_count))
    return particle_list, particle_dict
#*********************************************************************************************#
def vir2_csv_reader(csv_list, cont_window):
    """Reads the csvs generated by VIRAGO to determine final counts based on contrast window"""
    particle_list = ([])
    particle_dict = {}

    min_cont = float(cont_window[0])
    max_cont = float(cont_window[1])
    for csvfile in csv_list:
        csv_df = pd.read_csv(csvfile, error_bad_lines = False, header = 0)

        all_particles = [val for val in csv_df.perc_contrast if min_cont < val <= max_cont]
        val_count = len(kept_vals)
        csv_info = csvfile.split(".")
        csv_id = '{}.{}'.format(csv_info[1],csv_info[2])
        particle_dict[csv_id] = kept_vals
        particle_list.append(val_count)
        print('File scanned:  '+ csvfile + '; Particles counted: ' + str(val_count))
    return particle_list, particle_dict
#*********************************************************************************************#
# def nano_csv_reader(chip_name, spot_data, csv_list):
#     """Deprecated"""
#     min_corr = input("\nWhat is the correlation cutoff for particle count?"+
#                      " (choose value between 0.5 and 1)\t")
#     if min_corr == "": min_corr = 0.75
#     min_corr = float(min_corr)
#     contrast_window = input("\nEnter the minimum and maximum percent contrast values," +
#                             " separated by a comma (for VSV, 0-6% works well)\t")
#     assert isinstance(contrast_window, str)
#     contrast_window = contrast_window.split(",")
#     cont_0 = (float(contrast_window[0])/100)+1
#     cont_1 = (float(contrast_window[1])/100)+1
#     min_corr_str = str("%.2F" % min_corr)
#     particles_list = []
#     particle_dict = {}
#     nano_csv_list = [csvfile for csvfile in csv_list if csvfile.split(".")[-2].isdigit()]
#     for csvfile in nano_csv_list: ##This pulls particle data from the CSVs generated by nanoViewer
#         csv_data = pd.read_table(
#                              csvfile, sep = ',',
#                              error_bad_lines = False, usecols = [1,2,3,4,5],
#                              names = ("contrast", "correlation", "x", "y", "slice")
#                              )
#         filtered = csv_data[(csv_data['contrast'] <= cont_1)
#                     & (csv_data['contrast'] > cont_0)
#                     & (csv_data['correlation'] >= min_corr)][['contrast','correlation']]
#         particles = len(filtered)
#         csv_id = csvfile.split(".")[1] + "." + csvfile.split(".")[2]
#         particle_dict[csv_id] = list(round((filtered.contrast - 1) * 100, 4))
#         particles_list.append(particles)
#         print('File scanned: '+ csvfile + '; Particles counted: ' + str(particles))
#         particle_count_col = str('particle_count_'+ min_corr_str
#                            + '_' + contrast_window[0]
#                            + '_' + contrast_window[1]+ '_')
#     spot_data[particle_count_col] = particles_list
#     #for row in spot_data.iterrows():
#     filtered_density = spot_data[particle_count_col] / spot_data.area * 0.001
#     spot_data = pd.concat([spot_data, filtered_density.rename('kparticle_density')], axis = 1)
#     dict_file = pd.io.json.dumps(particle_dict)
#     with open('../virago_output/' + chip_name + '/' + chip_name
#               + '_particle_dict_' + min_corr_str + 'corr.txt', 'w') as f:
#               f.write(dict_file)
#     print("Particle dictionary file generated")
#
#     return min_corr, spot_data, particle_dict, contrast_window
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
        kp_df = spot_df['kparticle_density'][(spot_df.spot_number == x)].reset_index(drop=True)

        norm_list = [round(kp_df[i] - kp_df[0],3) for i in range(0,len(kp_df))]
        normalized_density.append(norm_list)

    normalized_density = [item for sublist in normalized_density for item in sublist]

    return normalized_density
#*********************************************************************************************#
def vdata_reader(vdata_list, prop_list):
    vdata_dict = {}
    for prop in prop_list:
        d_list = []
        for file in vdata_list:
            full_text = {}
            with open(file) as f:
                for line in f:
                    # key, val = line.split(":")
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
        print(d_list)
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
def spot_remover(spot_df, particle_dict, mod_pgm = False):
    excise_toggle = input("Would you like to remove any spots from the analysis? (y/[n])\t")
    assert isinstance(excise_toggle, str)
    if excise_toggle.lower() in ('y','yes'):
        excise_spots = input("Which spots? (Separate all spot numbers by a comma)\t")
        excise_spots = excise_spots.split(",")
        for ex_spot in excise_spots:
            spot_df.loc[spot_df.spot_number == int(ex_spot), 'valid'] = False
            for key in particle_dict.keys():
                spot_num = int(key.split(".")[0])
                if spot_num == ex_spot:
                    particle_dict[key] = 0

            if mod_pgm == True:
                os.chdir(iris_path)
                bad_pgms = sorted(glob.glob('*.'+three_digs(ex_spot)+'.*.*.pgm'))
                for pgm in bad_pgms:
                    pgm_split = pgm.split('.')
                    pgm_split.insert(-1, 'bad')
                    new_pgm = '.'.join(pgm_split)
                    os.rename(pgm, new_pgm)


    return spot_df, particle_dict
#*********************************************************************************************#
