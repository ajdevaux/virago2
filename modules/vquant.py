from __future__ import division
from future.builtins import input
from scipy.ndimage.filters import gaussian_filter
# from scipy.ndimage.morphology import binary_dilation
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import pdist, squareform

from scipy.sparse import csr_matrix, csgraph
from skimage import img_as_float
from skimage.filters import sobel_h, sobel_v
from skimage.feature import peak_local_max
from skimage.morphology import medial_axis, skeletonize
from skimage.measure import label, regionprops, perimeter

import pandas as pd
import numpy as np
import itertools as itt
import math, warnings, re, os, glob
from modules import vpipes,vimage
#*********************************************************************************************#
#
#           SUBROUTINES
#
#*********************************************************************************************#
def psf_sine(x,a,b,c):
    return a * np.sin(b * x + c)
#*********************************************************************************************#
def _multistep_z(z,zero_pt):
    """
    Helper function to fix the inconsistent steps the stage makes when acquiring images
    for certain experiments
    """
    if z < zero_pt: return z
    else: return z+z-zero_pt

#*********************************************************************************************#
def eccentricity(mu):
    """Measures the eccentricity of a binary object when given the central moments *mu* of that object"""
    mu02 = mu[0,2]
    mu20 = mu[2,0]

    A = (mu20 - mu02)**2
    B = 4*(mu[1,1]**2)
    C = (mu20 + mu02)**2

    return abs((A - B) / C)
#*********************************************************************************************#
def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges
#*********************************************************************************************#
def bbox_verts(bbox):
    bbox0 = bbox[0] - 2
    bbox1 = bbox[1] - 2
    bbox2 = bbox[2] + 2
    bbox3 = bbox[3] + 2

    return np.array([(bbox0,bbox1),(bbox0, bbox3),(bbox2, bbox3),(bbox2, bbox1)])
#*********************************************************************************************#
def measure_focal_plane(pic3D_norm, marker_locs, exo_toggle, marker_shape):
    """
    Uses the markers in the 3D image to determine the best image in the z stack to collect data from
    based on the defocus curve
    """
    marker_h, marker_w =  marker_shape
    hmarker_h = marker_h // 2
    hmarker_w = marker_w // 2
    pos_plane_list = []
    if exo_toggle == True:

        for loc in marker_locs:

            if (loc[0] < hmarker_h) or (loc[1] < hmarker_w):
                pass

            else:
                marker3D_img = pic3D_norm[:,
                                          loc[0]-hmarker_h : loc[0]+hmarker_h,
                                          loc[1]-hmarker_w : loc[1]+hmarker_w
                ]

                teng_vals = [np.mean(sobel_h(img)**2 + sobel_v(img)**2) for img in marker3D_img]
                min_plane = teng_vals.index(min(teng_vals))
                pos_vals = teng_vals[:min_plane]
                if not pos_vals == []:

                    pos_plane_list.append(teng_vals.index(max(pos_vals)))

                # neg_vals = teng_vals[min_plane:]
                # neg_vals_diff = list(np.diff(neg_vals))
                # neg_plane_list = [neg_vals_diff.index(val) for val in neg_vals_diff if val < 0]

    return pos_plane_list
#*********************************************************************************************#
def classify_shape(shapedex, pic2D, shape, delta, intensity, operator = 'greater'):
    with warnings.catch_warnings():
        ##RuntimeWarning ignored: invalid values are expected
        warnings.simplefilter("ignore")
        warnings.warn(RuntimeWarning)
        if operator == 'greater':
            shape_y, shape_x = np.where((np.abs(shapedex - shape) <= delta) & (pic2D >= intensity))
        else:
            shape_y, shape_x = np.where((np.abs(shapedex - shape) <= delta) & (pic2D <= intensity))
    return list(zip(shape_y, shape_x))
#*********************************************************************************************#
def binary_data_extraction(pic_binary, intensity_img, prop_list, pix_range):

    binary_props = regionprops(label(pic_binary, connectivity=2), intensity_img,
                             coordinates='xy', cache=True
    )

    # with warnings.catch_warnings():
    #     ##UserWarning ignored
    #     warnings.simplefilter("ignore")
    #     warnings.warn(UserWarning)
    #     shape_df = pd.DataFrame([[region['label'],
    #                               #[tuple(coords) for coords in region['coords']],
    #                               region['coords'],
    #                               region['moments'],
    #                               region['moments_central'],
    #                               region['moments_normalized'],
    #                               region['bbox'],
    #
    #
    #                               # region['centroid'],
    #
    #                               # region['area'],
    #                               region['perimeter'],
    #                               region['convex_image'],
    #                               region['major_axis_length'],
    #                               region['minor_axis_length']]
    #                               for region in binary_props
    #                               if (region['area'] > pix_range[0]) & (region['area'] <= pix_range[1])],
    #                               columns=['label','coords','moments','moments_central','moments_normalized','bbox',
    #                                        'perimeter','convex_image','major_axis_length','minor_axis_length']
    #     )

    shape_df = pd.DataFrame([[region[prop] for prop in prop_list] for region in binary_props
                              if (region['area'] > pix_range[0]) & (region['area'] <= pix_range[1])],
                              columns=prop_list
    )







    return shape_df
#*********************************************************************************************#
def particle_masker(pic_binary, shape_df, pass_num, first_scan = 1):
    particle_mask = np.zeros_like(pic_binary, dtype=int)
    rows, cols = zip(*[item for sublist in shape_df.coords.tolist() for item in sublist])
    particle_mask[rows,cols] = 1

    return particle_mask
#*********************************************************************************************#
def density_normalizer(spot_df, spot_counter):
    """Particle count normalizer so pass 1 = 0 particle density"""
    normalized_density = []
    for x in range(1, spot_counter + 1):
        kp_df = spot_df.kparticle_density[(spot_df.spot_number == x)].reset_index(drop=True)
        pass_count = len(kp_df)
        print(kp_df)
        j = 0
        if pass_count > 1:
            while np.isnan(kp_df[j]):
                print(kp_df)
                j += 1
                if (j == pass_count - 1):
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
    vdata_df.columns = props
    vdata_df = vdata_df.applymap(lambda x: x.strip(' \n'))

    vdata_df['validity'] = vdata_df['validity'].apply(lambda x: eval(x))

    return vdata_df

#*********************************************************************************************#
def _pixel_graph_easy(image, steps, distances, num_edges, height=None):
    row = np.empty(num_edges, dtype=int)
    col = np.empty(num_edges, dtype=int)
    data = np.empty(num_edges, dtype=float)
    image = image.ravel()
    n_neighbors = steps.size
    start_idx = np.max(steps)
    end_idx = image.size + np.min(steps)
    k = 0
    for i in range(start_idx, end_idx + 1):
        if image[i] != 0:
            for j in range(n_neighbors):
                n = steps[j] + i
                if image[n] != 0 and image[n] != image[i]:
                    row[k] = image[i]
                    col[k] = image[n]
                    if height is None:
                        data[k] = distances[j]

                    else:
                        data[k] = np.sqrt(distances[j] ** 2 + (height[i] - height[n]) ** 2)
                    k += 1

    graph = sparse.coo_matrix((data[:k], (row[:k], col[:k]))).tocsr()

    return graph
#*********************************************************************************************#
def measure_filo_length(coords, pix_per_um):
    sparse_matrix = csr_matrix(squareform(pdist(coords,metric='euclidean')))
    # vimage.gen_img(sparse_matrix)

    distances = csgraph.shortest_path(sparse_matrix,method = 'FW',return_predecessors=False)

    ls_path = np.max(distances)

    farpoints = np.where(distances == ls_path)

    filo_len = float(round(ls_path / pix_per_um, 3))
    vertices = [coords[farpoints[0][0]],coords[farpoints[0][len(farpoints[0]) // 2]]]

    return filo_len, vertices
#*********************************************************************************************#
def measure_defocus(z_stack, std_z_stack, measure_corr=True,
                    a0=0.1, b0=0.1, c0=1, show = False):
    """
    Measures through the z stack and collects intensity data
    """

    zstack_len = len(z_stack)
    x_data = np.arange(1,zstack_len+1)

    if zstack_len <= 10:
        vzfunc = np.vectorize(_multistep_z)
        x_data = vzfunc(x_data, 7)

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

    top_bot = max_z_img[[top_row, bot_row], lft_col:rgt_col+1]

    lft_rgt = max_z_img[top_row+1:bot_row, [lft_col, rgt_col]]

    return np.concatenate((top_bot.ravel(), lft_rgt.ravel()))
#*********************************************************************************************#
def _overlap_tol(z_i, z_j):
    if z_i == z_j:
        return 1.0
    else:
        return 1/abs(z_i - z_j)
#*********************************************************************************************#
def _intersection_of_bbox(bb1, bb2, get_iou = False):
    """
    Calculate the Intersection over union or smaller of two bounding boxes.

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
    bb1_toprow, bb1_leftcol = bb1[0]
    bb1_bottomrow, bb1_rightcol = bb1[2]
    assert (bb1_leftcol < bb1_rightcol) & (bb1_toprow < bb1_bottomrow)

    bb2_toprow, bb2_leftcol = bb2[0]
    bb2_bottomrow, bb2_rightcol = bb2[2]
    assert (bb2_leftcol < bb2_rightcol) & (bb2_toprow < bb2_bottomrow)

    ##check if one box is completely inside the other
    if (  (bb1_rightcol <= bb2_rightcol)
        & (bb1_leftcol >= bb2_leftcol)
        & (bb1_toprow >= bb2_toprow)
        & (bb1_bottomrow <= bb2_bottomrow)):
        return 1.0
    # determine the coordinates of the intersection box
    col_left = max(bb1_leftcol, bb2_leftcol)
    row_top = max(bb1_toprow, bb2_toprow)
    col_right = min(bb1_rightcol, bb2_rightcol)
    row_bottom = min(bb1_bottomrow, bb2_bottomrow)

    if (col_right < col_left) | (row_bottom < row_top):
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (col_right - col_left) * (row_bottom - row_top)

    # compute the area of both AABBs
    bb1_area = (bb1_rightcol - bb1_leftcol) * (bb1_bottomrow - bb1_toprow)
    bb2_area = (bb2_rightcol - bb2_leftcol) * (bb2_bottomrow - bb2_toprow)

    combined_area = (bb1_area + bb2_area) - intersection_area
    #determine which bounding box is smaller
    #then divide by the smaller bounding box to get the amount
    #the smaller bounding box is enveloped by the larger one
    if get_iou == True:
        return intersection_area / (combined_area + 0.000001)
    elif bb1_area > bb2_area:
        return intersection_area / bb2_area
    else:
        return intersection_area / bb1_area

#*********************************************************************************************#
def mark_overlaps(neighbor_tree_dist, shape_df, iou=False):
    try:
        z_series = shape_df.z_intensity
    except AttributeError:
        z_series = shape_df.area

    bbox_series = shape_df.bbox

    overlap_ix_list = []

    for i,j in neighbor_tree_dist:
        z_i,z_j = z_series[[i,j]]
        bb_i, bb_j = bbox_series[[i,j]]

        overlap = _overlap_tol(z_i, z_j)
        ios = _intersection_of_bbox(bb_i, bb_j, get_iou=iou)

        if ios >= overlap:

            if z_i < z_j:
                overlap_ix_list.append(i)
            else:
                overlap_ix_list.append(j)

    return overlap_ix_list
#*********************************************************************************************#
def remove_overlapping_objs(shape_df, radius=20):

    neighbor_tree = cKDTree(np.array(shape_df.centroid.tolist()))

    neighbor_tree_dist = neighbor_tree.query_pairs(radius, output_type='ndarray')

    overlap_ix = mark_overlaps(neighbor_tree_dist, shape_df)

    return shape_df.drop(overlap_ix).reset_index(drop=True)
#*********************************************************************************************#
def spot_remover(spot_df, contrast_df, vcount_dir, iris_path, quarantine_img = False):
    excise_toggle = input("Would you like to remove any spots from the analysis? (y/[n])\t")
    assert isinstance(excise_toggle, str)
    if excise_toggle.lower() in ('y','yes'):
        excise_spots = input("Which spots? (Separate all spot numbers by a comma)\t")
        excise_spots = [int(x) for x in excise_spots.split(',')]

        spot_df.loc[spot_df.spot_number.isin(excise_spots), 'validity'] = False

        drop_col_list = [col for col in contrast_df.columns if int(col.split(".")[0]) in excise_spots]
        contrast_df.drop(columns=drop_col_list, inplace=True)

        os.chdir(vcount_dir)

        vdata_list = glob.glob('*.vdata.txt')
        bad_vfiles = [vf for vf in vdata_list if int(vf.split('.')[1]) in excise_spots]
        print(bad_vfiles)
        for vf in bad_vfiles:
            old_vf = vf+'~'
            os.rename(vf, old_vf)
            with open(old_vf, 'r+') as ovf, open(vf, 'w') as nvf:
                lines = ovf.readlines()
                # print(lines)
                for line in lines:
                    if line.split(':')[0] == 'validity':
                        # print(line)
                        nvf.write('validity: False\n')

                    else:
                        nvf.write(line)
            os.remove(old_vf)

        if quarantine_img == True:
            os.chdir(iris_path)
            if not os.path.exists('bad_imgs'):
                os.makedirs('bad_imgs')

            bad_imgs = [img for sublist in
                       [glob.glob('*.' + str(spot).zfill(3) + '.*.tif') for spot in excise_spots]
                       for img in sublist
            ]
            for img in bad_imgs:
                os.rename(img, "{}/bad_imgs/{}".format(iris_path, img))


    return spot_df, contrast_df
#*********************************************************************************************#
