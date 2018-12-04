#! /usr/local/bin/python3
from __future__ import division
from future.builtins import input
from lxml import etree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, csgraph
from scipy.stats import norm, gamma
from scipy import ndimage as ndi
# from skimage import img_as_float
# from skimage.filters import gaussian
from skimage import exposure, feature, transform, filters, util, measure, morphology, io, img_as_float
import os, json, math, warnings, sys, glob
#*********************************************************************************************#
#
#           SUBROUTINES
#
#*********************************************************************************************#
def _hysteresis_th(image, low, high):
    """Ripped from https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/thresholding.py#L885"""
    low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
    mask_low = image > low
    mask_high = image > high
    # Connected components of mask_low
    labels_low, num_labels = ndi.label(mask_low)
    # Check which connected components contain pixels from mask_high
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]
    return thresholded
#*********************************************************************************************#
def filobinarize(filo_pic, pic_orig, thresh_scalar, return_props = True, show_hist = False):

    if not type(filo_pic) is np.ndarray:
        spot_median = np.ma.median(filo_pic)
        filo_pic = filo_pic.filled(0)
    else:
        spot_median = np.median(filo_pic)
    thresh_high = spot_median + thresh_scalar
    thresh_low = spot_median + (thresh_scalar / 2)
    if show_hist == True:
        # plt.xticks(np.arange(0,np.max(filo_pic)), size = 10)
        plt.xlim((0,thresh_high+0.05))
        plt.axvline(thresh_high, color = 'r')
        plt.axvline(thresh_low, color = 'r', alpha = 0.5)
        plt.axvline(spot_median, color = 'b')
        sns.distplot(filo_pic.ravel(),
                     kde = False,
                     bins = int(np.ceil(np.max(filo_pic)*1000)),
                     norm_hist = True)
        plt.show()
        plt.clf()

    # pic_binary = (filo_pic > thresh).astype(int)

    pic_binary = _hysteresis_th(filo_pic, low = thresh_low, high = thresh_high)

    pic_binary = ndi.binary_fill_holes(pic_binary)

    if return_props == True:
        pic_binary_label = measure.label(pic_binary, connectivity = 2)
        binary_props = measure.regionprops(pic_binary_label, pic_orig, cache = True)
        return pic_binary, binary_props, thresh_high
    else:
        return pic_binary, thresh_high
#*********************************************************************************************#
def filoskel(pic_binary, pic_orig):
    pic_skel = morphology.medial_axis(pic_binary)
    pic_skel_label, labels = measure.label(pic_skel,
                                              return_num = True,
                                              connectivity = 2)
    skel_props = measure.regionprops(pic_skel_label, pic_orig, cache = True)

    return pic_skel, skel_props
#*********************************************************************************************#
def _measure_filament(coords_dict, res):
    filo_lengths, vertex1, vertex2 = [],[],[]
    for key in coords_dict:
        fiber_coords = coords_dict[key]
        dist_matrix = pdist(fiber_coords, metric='cityblock')
        sparse_matrix = csr_matrix(squareform(dist_matrix))
        distances, preds = csgraph.shortest_path(sparse_matrix,
                                                 method = 'FW',
                                                 return_predecessors=True)
        ls_path = np.max(distances)
        farpoints = np.where(distances == ls_path)
        endpt_loc = len(farpoints[0]) // 2
        v1 = fiber_coords[farpoints[0][0]]
        v2 = fiber_coords[farpoints[0][endpt_loc]]
        filo_lengths.append(float(round(ls_path / res, 3)))
        vertex1.append(tuple(v1))
        vertex2.append(tuple(v2))

    return filo_lengths, vertex1, vertex2
#*********************************************************************************************#

#*********************************************************************************************#
def filoskel_quant(regionprops, res, area_filter = (4,1500)):
    coords_dict = {}
    label_list, centroid_list = [],[]
    skel_df = pd.DataFrame()
    for region in regionprops:
        if (region['area'] > area_filter[0]) & (region['area'] < area_filter[1]):
            label_list.append(region['label'])
            coords_dict[region['label']] = region['coords']
            centroid_list.append(region['centroid'])

    skel_df['label_skel'] = label_list
    skel_df['centroid_skel'] = centroid_list

    filo_lengths, vertex1, vertex2 = _measure_filament(coords_dict, res)

    skel_df['filament_length_um'] = filo_lengths
    skel_df['vertex1'] = vertex1
    skel_df['vertex2'] = vertex2
    skel_df.reset_index(drop = True, inplace = True)
    return skel_df
#*********************************************************************************************#
def filobinary_quant(regionprops, pic_orig, res, area_filter = (12,210)):
    label_list, centroid_list, area_list, coords_list, bbox_list, perim_list = [],[],[],[],[],[]
    binary_df = pd.DataFrame()
    for region in regionprops:
        if (region['area'] > area_filter[0]) & (region['area'] < area_filter[1]):
            label_list.append(region['label'])
            coords_list.append(region['coords'])
            centroid_list.append(region['centroid'])
            bbox_list.append((region['bbox'][0:2], region['bbox'][2:]))
            area_list.append(region['area'])
            perim_list.append(region['perimeter'])

    roundness_list = _roundness_measure(area_list, perim_list)

    binary_df['label_bin'] = label_list
    binary_df['centroid_bin'] = centroid_list
    binary_df['area'] = area_list
    binary_df['roundness'] = roundness_list
    med_intensity_list = [np.median([pic_orig[tuple(coords)]
                          for coords in coord_array])
                          for coord_array in coords_list]
    binary_df['median_intensity'] = med_intensity_list

    median_bg_list, bbox_vert_list = [],[]
    for bbox in bbox_list:
        top_left = (bbox[0][0],bbox[0][1])
        top_rt = (bbox[0][0], bbox[1][1])
        bot_rt = (bbox[1][0], bbox[1][1])
        bot_left = (bbox[1][0], bbox[0][1])
        bbox_verts = np.array([top_left,top_rt,bot_rt,bot_left])
        bbox_vert_list.append(bbox_verts)

        top_edge = pic_orig[bbox[0][0],bbox[0][1]:bbox[1][1]+1]
        bottom_edge = pic_orig[bbox[1][0]-1,bbox[0][1]:bbox[1][1]+1]
        rt_edge = pic_orig[bbox[0][0]:bbox[1][0]+1,bbox[1][1]]
        left_edge = pic_orig[bbox[0][0]:bbox[1][0]+1,bbox[0][1]]
        all_edges = np.hstack([top_edge, bottom_edge, rt_edge, left_edge])

        median_bg = np.median(all_edges)
        median_bg_list.append(median_bg)

    binary_df['median_background'] = median_bg_list
    binary_df['filo_pc'] = (((binary_df.median_intensity - binary_df.median_background) * 100)
                            / binary_df.median_background)

    binary_df['bbox_verts'] = bbox_vert_list

    binary_df.reset_index(drop = True, inplace = True)

    return binary_df,bbox_vert_list
#*********************************************************************************************#
def boxcheck_merge(df1, df2, pointcol, boxcol, dropcols = False):
    new_df = pd.DataFrame()
    for i, point in enumerate(df1[pointcol]):
        arr_point = np.array(point).reshape(1,2)
        for j, bbox in enumerate(df2[boxcol]):
            boxcheck = measure.points_in_poly(arr_point,bbox)
            if boxcheck == True:
                series1 = df1.loc[i]
                series2 = df2.loc[j]
                combo_series = series1.append(series2)
                new_df = new_df.append(combo_series, ignore_index = True)
                df1.drop([i], inplace = True)
                break
    if (dropcols == True) & (not new_df.empty):
        new_df.drop(columns = ['centroid_skel',
                               'label_bin',
                               'median_background',
                               'median_intensity',
                               'pc'],
                    inplace = True)
    return new_df
#*********************************************************************************************#
def no_filos(filo_dir, png):
    filo_df = pd.DataFrame(columns = ['centroid_bin',
                                      'label_skel',
                                      'filament_length_um',
                                      'roundness',
                                      'pc',
                                      'vertex1',
                                      'vertex2',
                                      'area',
                                      'bbox_verts'])
    filo_df.to_csv(filo_dir + '/' + png + '.filocount.csv')
    return filo_df
#*********************************************************************************************#
