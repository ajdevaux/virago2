#! /usr/local/bin/python3
from __future__ import division
import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
from scipy.ndimage import zoom
from scipy.stats import norm, gamma
from skimage.exposure import cumulative_distribution, equalize_adapthist, rescale_intensity
from skimage.feature import match_template, peak_local_max, canny
from skimage.transform import hough_circle, hough_circle_peaks
import math, warnings
# from vpipes import _dict_matcher
#*********************************************************************************************#
#
#           SUBROUTINES
#
#*********************************************************************************************#
def image_details(fig1, fig2, fig3, pic_edge, chip_name, png, save = False, dpi = 96):
    """A subroutine for debugging contrast adjustment"""
    bin_no = 55
    nrows, ncols = fig1.shape
    figsize = (ncols/dpi/2, nrows/dpi/2)
    fig = plt.figure(figsize = figsize, dpi = dpi)

    ax_img = plt.Axes(fig,[0,0,1,1])
    ax_img.set_axis_off()
    fig.add_axes(ax_img)

    fig3[pic_edge] = fig3.max()*2

    ax_img.imshow(fig3, cmap = 'gray')

    pic_cdf1, cbins1 = cumulative_distribution(fig1, bin_no)
    pic_cdf2, cbins2 = cumulative_distribution(fig2, bin_no)
    pic_cdf3, cbins3 = cumulative_distribution(fig3, bin_no)

    ax_hist1 = plt.axes([.05, .05, .25, .25])
    ax_cdf1 = ax_hist1.twinx()
    ax_hist2 = plt.axes([.375, .05, .25, .25])
    ax_cdf2 = ax_hist2.twinx()
    ax_hist3 = plt.axes([.7, .05, .25, .25])
    ax_cdf3 = ax_hist3.twinx()

    # hist1, hbins1 = np.histogram(fig1.ravel(), bins = bin_no)
    # hist2, hbins2 = np.histogram(fig2.ravel(), bins = bin_no)
    # hist3, hbins3 = np.histogram(fig3.ravel(), bins = bin_no)
    fig1r = fig1.ravel(); fig2r = fig2.ravel(); fig3r = fig3.ravel()

    hist1, hbins1, __ = ax_hist1.hist(fig1r, bin_no, facecolor = 'r', normed = True)
    hist2, hbins2, __ = ax_hist2.hist(fig2r, bin_no, facecolor = 'b', normed = True)
    hist3, hbins3, __ = ax_hist3.hist(fig3r, bin_no, facecolor = 'g', normed = True)
    # hist_dist1 = scipy.stats.rv_histogram(hist1)

    ax_hist1.patch.set_alpha(0); ax_hist2.patch.set_alpha(0); ax_hist3.patch.set_alpha(0)

    ax_cdf1.plot(cbins1, pic_cdf1, color = 'w')
    ax_cdf2.plot(cbins2, pic_cdf2, color = 'c')
    ax_cdf3.plot(cbins3, pic_cdf3, color = 'y')

    bin_centers2 = 0.5*(hbins2[1:] + hbins2[:-1])
    m2, s2 = norm.fit(fig2r)
    pdf2 = norm.pdf(bin_centers2, m2, s2)
    ax_hist2.plot(bin_centers2, pdf2, color = 'm')
    mean, var, skew, kurt = gamma.stats(fig2r, moments='mvsk')
    print(mean, var, skew, kurt)

    ax_hist1.set_title("Normalized", color = 'r')
    ax_hist2.set_title("CLAHE Equalized", color = 'b')
    ax_hist3.set_title("Contrast Stretched", color = 'g')
    ax_hist1.set_ylim([0,max(hist1)])
    ax_hist3.set_ylim([0,max(hist3)])
    ax_hist1.set_xlim([np.median(fig1)-0.25,np.median(fig1)+0.25])
    #ax_cdf1.set_ylim([0,1])
    ax_hist2.set_xlim([np.median(fig2)-0.5,np.median(fig2)+0.5])
    ax_hist3.set_xlim([0,1])
    if save == True:
        plt.savefig('../virago_output/' + chip_name
                    + '/processed_images/' + png
                    + '_image_details.png',
                    dpi = dpi)
    plt.show()

    plt.close('all')
    return hbins2, pic_cdf1
#*********************************************************************************************#
def gen_img(image, name = 'default', savedir = '', cmap = 'gray', dpi = 96, show = True, zoom_amt = 1):
    print(zoom_amt)
    nrows, ncols = image.shape[0], image.shape[1]
    figsize = ((ncols/dpi), (nrows/dpi))
    fig = plt.figure(figsize = figsize, dpi = dpi)
    axes = plt.Axes(fig,[0,0,1,1])
    fig.add_axes(axes)
    axes.set_axis_off()
    axes.imshow(image, cmap = cmap)
    # if zoom_amt > 1:
    #     image = zoom(image, zoom_amt)
    if savedir:
        plt.savefig('{}/{}.png'.format(savedir, name), dpi = dpi)
        print("\nFile generated: {}.png\n".format(name))
    if show == True: plt.show()
    plt.close('all')
#*********************************************************************************************#
def gen_img3D(im3D, cmap = "gray", step = 1):
    """Debugging function for viewing all image files in a stack"""
    _, axes = plt.subplots(nrows = int(np.ceil(zslice_count/4)),
                           ncols = 4,
                           figsize = (16, 14))
    vmin = im3D.min()
    vmax = im3D.max()

    for ax, image in zip(axes.flatten(), im3D[::step]):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()

    plt.show()
    plt.close('all')
#*********************************************************************************************#
def marker_finder(image, marker, thresh = 0.9, gen_mask = False):
    """This locates the "backwards-L" shapes in the IRIS images"""
    marker_match = match_template(image, marker, pad_input = True)
    # gen_img(marker_match, show=True)
    locs = peak_local_max(marker_match,
                                  min_distance = 800,
                                  threshold_rel = thresh,
                                  exclude_border = False,
                                  num_peaks = 4
    )
    locs = [tuple(coords) for coords in locs]
    locs.sort(key = lambda coord: coord[1])

    mask = None
    if gen_mask == True:
        mask = np.zeros(shape = image.shape, dtype = bool)
        h, w = marker.shape
        h += 5; w += 5
        for coords in locs:
            marker_w = (np.arange(coords[1] - w/2,coords[1] + w/2)).astype(int)
            marker_h = (np.arange(coords[0] - h/2,coords[0] + h/2)).astype(int)
            mask[marker_h[0]:marker_h[-1],marker_w[0]:marker_w[-1]] = True

        return locs, mask
    else:
        return locs
#*********************************************************************************************#
def spot_finder(image, canny_sig = 2, rad_range = (525, 651), center_mode = False):
    """Locates the antibody spot convalently bound to the SiO2 substrate
    where particles of interest should be accumulating"""
    nrows, ncols = image.shape
    pic_canny = canny(image, sigma = canny_sig)
    if center_mode == True:
        xyr = (536, 540, 500)
    else:
        hough_radius = range(rad_range[0], rad_range[1], 25)
        hough_res = hough_circle(pic_canny, hough_radius)
        accums, cx, cy, rad = hough_circle_peaks(hough_res, hough_radius, total_num_peaks=1)
        xyr = tuple((int(cx), int(cy), int(rad)))
    print("Spot center coordinates (row, column, radius): {}\n".format(xyr))
    return xyr, pic_canny
#*********************************************************************************************#
def clahe_3D(img_stack, cliplim = 0.003, recs = 0):
    """Performs the contrast limited adaptive histogram equalization on the stack of images"""
    # shape =
    # mid_pic = int(np.ceil(shape[0]/2))
    if img_stack.ndim == 2: img_stack = np.array([img_stack])

    img3D_clahe = np.empty_like(img_stack).astype('float64')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn(UserWarning)##Images are acutally converted to uint16 for some reason
        for plane,image in enumerate(img_stack):
            img3D_clahe[plane] = equalize_adapthist(image, clip_limit = cliplim)
            image_r = img3D_clahe[plane].ravel()
                # hist1, hbins1 = np.histogram(image_r, bins = 55)
                # mean, std = norm.fit(image_r)
                # mean, var, skew, kurt = norm.stats(moments='mvsk')

                # var = np.var(img3D_clahe[plane])
                # print(var)
                # if var < 0.012:
                #     recs += 1
                #     print("Recursing %d" % recs)
                #     mult = 3.3 - (0.3 * recs)
                #     cliplim = round(cliplim * mult,3)
                #     img3D_clahe[plane] = clahe_3D(img3D_clahe[plane], cliplim, recs = recs)
                #
                # else: print("Sweet Distribution!")
    return img3D_clahe
#*********************************************************************************************#
def rescale_3D(img_stack, perc_range = (2,98)):
    """Streches the histogram for all images in stack to further increase contrast"""
    img3D_rescale = np.empty_like(img_stack)
    for plane, image in enumerate(img_stack):
        p1,p2 = np.percentile(image, perc_range)
        img3D_rescale[plane] = rescale_intensity(image, in_range=(p1,p2))
    return img3D_rescale
#*********************************************************************************************#
def masker_3D(image_stack, mask, filled = False, fill_val = 0):
    """Masks all images in stack so only areas not masked (the spot) are quantified.
    Setting filled = True will return a normal array with fill_val filled in on the masked areas.
    Default filled = False returns a numpy masked array."""

    pic3D_masked = np.ma.empty_like(image_stack)
    pic3D_filled = np.empty_like(image_stack)

    for plane, image in enumerate(image_stack):
        pic3D_masked[plane] = np.ma.array(image, mask = mask)
        if filled == True:
            pic3D_filled[plane] = pic3D_masked[plane].filled(fill_value = fill_val)

    if filled == False:
        return pic3D_masked
    else:
        return pic3D_filled
#*********************************************************************************************#
def measure_rotation(marker_dict, spot_pass_str):
    """Measures how rotated the image is compared to the previous scan"""
    marker_ct = len(marker_dict[spot_pass_str])
    if marker_ct == 2:
        r1 = marker_dict[spot_pass_str][0][0]
        r2 = marker_dict[spot_pass_str][1][0]
        c1 = marker_dict[spot_pass_str][0][1]
        c2 = marker_dict[spot_pass_str][1][1]
        row_diff = abs(r1 - r2)
        col_diff = abs(c1 - c2)
        if (col_diff < row_diff) & (col_diff < 15):
            print("Markers vertically aligned")
            img_rot_deg = math.degrees(math.atan(col_diff / row_diff))
            print("\nImage rotation: {} degrees\n".format(round(img_rot_deg,3)))
        elif (row_diff < col_diff) & (row_diff < 15):
            print("Markers horizontally aligned")
            img_rot_deg = math.degrees(math.atan(row_diff / col_diff))
            print("\nImage rotation: {} degrees\n".format(round(img_rot_deg,3)))
        else:
            print("Markers unaligned; cannot compute rotation")
            img_rot_deg = np.nan
    else:
        print("Wrong number of markers ({}); cannot compute rotation".format(marker_ct))
        img_rot_deg = np.nan
    return img_rot_deg
#*********************************************************************************************#
def _dict_matcher(_dict, spot_num, pass_num, mode = 'series'):
    if mode == 'baseline': prev_pass = 1
    elif mode == 'series': prev_pass = pass_num - 1
    for key in _dict.keys():
        split_key = key.split('.')
        if (split_key[0] == str(spot_num)) &  (split_key[1] == str(prev_pass)):
            prev_vals = _dict[key]
        elif (split_key[0] == str(spot_num)) &  (split_key[1] == str(pass_num)):
            new_vals = _dict[key]
    return prev_vals, new_vals
#*********************************************************************************************#
def measure_shift(marker_dict, pass_num, spot_num, mode = 'baseline'):
    overlay_toggle = True
    if pass_num > 1:
        prev_locs, new_locs = _dict_matcher(marker_dict, spot_num, pass_num, mode = mode)
        plocs_ct = len(prev_locs)
        nlocs_ct = len(new_locs)
        if (plocs_ct > 0) & (nlocs_ct > 0) & (plocs_ct != nlocs_ct):
            shift_array = [np.subtract(coords1, coords0)
                           for coords0 in prev_locs
                           for coords1 in new_locs
                           if np.all(abs(np.subtract(coords1, coords0)) <= 50)
                          ]
        elif (plocs_ct > 0) & (nlocs_ct > 0) & (plocs_ct == nlocs_ct):
            shift_array = [np.subtract(coords1, coords0)
                           for coords0 in prev_locs
                           for coords1 in new_locs
                           if np.all(abs(np.subtract(coords1, coords0)) <= 75)
                           ]
        else:
            shift_array = []

        shift_array = np.asarray(shift_array)
        if (shift_array.size > 0) & (shift_array.ndim == 1):
            mean_shift = shift_array
            print("Image shift: {}".format(mean_shift))
            overlay_toggle = True
        elif shift_array.size == 0:
            mean_shift = np.array([0, 0])
            print("No compatible markers, cannot compute shift")
            overlay_toggle = False
        else:
            mean_shift = np.mean(shift_array, axis = 0)
            print("Image shift: {}\n".format(mean_shift))
            overlay_toggle = True
    else:
        mean_shift = tuple((0,0))
        overlay_toggle = False

    return tuple(mean_shift), overlay_toggle
#*********************************************************************************************#
def overlayer(overlay_dict, overlay_toggle, spot_num, pass_num,
              mean_shift, mode ='baseline'):
    vshift = int(np.ceil(mean_shift[0]))
    hshift = int(np.ceil(mean_shift[1]))
    if (pass_num > 1) & (overlay_toggle == True):
        bot_img, top_img = _dict_matcher(overlay_dict, spot_num, pass_num, mode = mode)

        try: bot_img
        except NameError: print("Cannot overlay images")
        else:
            if vshift < 0:
                bot_img = np.delete(bot_img, np.s_[0:abs(vshift)], axis = 0)
                top_img = np.delete(top_img, np.s_[-abs(vshift):], axis = 0)
            elif vshift > 0:
                bot_img = np.delete(bot_img, np.s_[-abs(vshift):], axis = 0)
                top_img = np.delete(top_img, np.s_[0:abs(vshift)], axis = 0)
            if hshift < 0:
                bot_img = np.delete(bot_img, np.s_[0:abs(hshift)], axis = 1)
                top_img = np.delete(top_img, np.s_[-abs(hshift):], axis = 1)
            elif hshift >0:
                bot_img = np.delete(bot_img, np.s_[-abs(hshift):], axis = 1)
                top_img = np.delete(top_img, np.s_[0:abs(hshift)], axis = 1)

            image_overlay = np.dstack((bot_img, top_img, np.zeros_like(bot_img)))
            # image_overlay = top_img - bot_img
            return image_overlay

    elif pass_num == 1:
        print("First scan of spot...")
    else:
        print("Cannot overlay images")
#*********************************************************************************************#
def cropper(pic, coords, img_dir, crop_pix = 150, zoom_amt = 1):
    crop_dict = {}
    cx = int(coords[0])
    cy = int(coords[1])
    rad = int(coords[2])
    for i in range(-300,301,150):

        sx = cx + i
        sy = cy + i

        print(i)
        print(sx, cy)
        print(cx, sy)
        if not (sx,cy) == (cx,sy):
            gen_img(pic[cy-crop_pix: cy+crop_pix, sx-crop_pix: sx+crop_pix],
                            name = "{}.{}-{}".format(img_name,cy,sx),
                            savedir = img_dir,
                            # zoom_amt = zoom_amt

            )
            gen_img(pic[sy-crop_pix: sy+crop_pix, cx-crop_pix: cx+crop_pix],
                            name = "{}.{}-{}".format(img_name,sy,cx),
                            savedir = img_dir,
            )
        else:
            gen_img(pic[cy-crop_pix: cy+crop_pix, cx-crop_pix: cx+crop_pix],
                            name = "{}.{}-{}".format(img_name,cy,cx),
                            savedir = img_dir
            )
#*********************************************************************************************#
