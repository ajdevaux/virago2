from future.builtins import input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import seaborn as sns
import warnings
#*********************************************************************************************#
def _color_mixer(zlen,c1,c2,c3,c4):
    """A function to create color gradients from 4 input colors"""
    if zlen > 1:
        cmix_r1 = np.linspace(c1[0],c2[0],int(zlen//2),dtype = np.float16)
        cmix_g1 = np.linspace(c1[1],c2[1],int(zlen//2),dtype = np.float16)
        cmix_b1 = np.linspace(c1[2],c2[2],int(zlen//2),dtype = np.float16)
        cmix_r2 = np.linspace(c3[0],c4[0],int(zlen//2),dtype = np.float16)
        cmix_g2 = np.linspace(c3[1],c4[1],int(zlen//2),dtype = np.float16)
        cmix_b2 = np.linspace(c3[2],c4[2],int(zlen//2),dtype = np.float16)
        cnew1 = [(cmix_r1[c], cmix_g1[c], cmix_b1[c]) for c in range(0,(zlen)//2,1)]
        cnew2 = [(cmix_r2[c], cmix_g2[c], cmix_b2[c]) for c in range(0,(zlen)//2,1)]
        cnew3 = [(np.mean(list([c2[0],c3[0]]),dtype = np.float16),
                  np.mean(list([c2[1],c3[1]]),dtype = np.float16),
                  np.mean(list([c2[2],c3[2]]),dtype = np.float16))]
        color_list = cnew1 + cnew3 + cnew2
    else:
        color_list = ['white']
    return color_list
#*********************************************************************************************#
def _circle_particles(particle_df, axes, exo_toggle):
    z_list = [z for z in list(set(particle_df.z))]# if str(z).isdigit()]
    zlen = len(z_list)
    dark_red = (0.645, 0, 0.148); pale_yellow = (0.996, 0.996, 0.746)
    pale_blue = (0.875, 0.949, 0.969); dark_blue = (0.191, 0.211, 0.582)
    blueflame_cm = _color_mixer(zlen, c1=dark_red, c2=pale_yellow, c3=pale_blue, c4=dark_blue)
    pc_hist = list()
    ax_hist = plt.axes([.7, .06, .25, .25])
    hist_max = 6
    for c, zslice in enumerate(z_list):
        circ_color = blueflame_cm[c]
        y = particle_df.loc[particle_df.z == zslice].y.reset_index(drop = True)
        x = particle_df.loc[particle_df.z == zslice].x.reset_index(drop = True)
        pc = particle_df.loc[particle_df.z == zslice].pc.reset_index(drop = True)
        try:
            if max(pc) > hist_max: hist_max = max(pc)
        except: ValueError
        if exo_toggle == True: crad = 0.2
        else: crad = 2
        # try:
        #     if max(pc) > 25: crad = 0.25
        # except: ValueError
        pc_hist.append(np.array(pc))
        for i in range(0,len(pc)):
            point = plt.Circle((x[i], y[i]), pc[i] * crad,
                                color = circ_color, linewidth = 1,
                                fill = False, alpha = 0.75)
            axes.add_patch(point)

    hist_color = blueflame_cm[:len(pc_hist)]
    hist_vals, hbins, hist_patches = ax_hist.hist(pc_hist, bins = 200, range = [0,30],
                                                  linewidth = 2, alpha = 0.5, stacked = True,
                                                  color = hist_color,
                                                  label = z_list)
    ax_hist.patch.set_alpha(0.5)
    ax_hist.patch.set_facecolor('black')
    ax_hist.legend(loc = 'best', fontsize = 8)
    if exo_toggle == True: ax_hist.set_xlim([0,25])
    else: ax_hist.set_xlim([0,15])

    for spine in ax_hist.spines: ax_hist.spines[spine].set_color('k')
    ax_hist.tick_params(color = 'k')
    plt.xticks(size = 10, color = 'w')
    plt.xlabel("% CONTRAST", size = 12, color = 'w')
    plt.yticks(size = 10, color = 'w')
    plt.ylabel("PARTICLE COUNT", color = 'w')
#*********************************************************************************************#
def processed_image_viewer(image, particle_df, spot_coords, res, chip_name,
                            filo_df = pd.DataFrame([]),
                            cmap = 'gray', dpi = 96, markers = [],
                            im_name = "",
                            show_particles = True, show_fibers = False,
                            show_filaments = False, exo_toggle = False,
                            show_markers = True, show_info = False,
                            show_image = False, scale = 15,
                            crosshairs = False, invert = False):
    """Generates a full-resolution PNG image after, highlighting features, showing counted particles,
    and a particle contrast histogram"""
    nrows, ncols = image.shape
    cx,cy,rad = spot_coords
    true_radius = round((rad - 20) / res,2)
    figsize = (ncols/dpi, nrows/dpi)
    fig = plt.figure(figsize = figsize, dpi = dpi)
    axes = plt.Axes(fig,[0,0,1,1])
    fig.add_axes(axes)
    axes.set_axis_off()
    if invert == True:
        image = util.invert(image)
    axes.imshow(image, cmap = cmap)

    ab_spot = plt.Circle((cx, cy), rad, color='#5A81BB',
                  linewidth=5, fill=False, alpha = 0.5)
    axes.add_patch(ab_spot)
    if show_info == True:
        scalebar_len_pix = res * scale
        scalebar_len = scalebar_len_pix / ncols
        scalebar_xcoords = ((0.98 - scalebar_len), 0.98)
        scale_text_xloc = np.mean(scalebar_xcoords) * ncols
        plt.axhline(y=100, xmin=scalebar_xcoords[0], xmax=scalebar_xcoords[1],
                    linewidth = 8, color = "red")
        plt.text(y = 85, x = scale_text_xloc, s = (str(scale)+ " " + r'$\mu$' + "m"),
                 color = 'red', fontsize = '20', horizontalalignment = 'center')
        plt.text(y = 120, x = scalebar_xcoords[0] * ncols, s = im_name,
                 color = 'red', fontsize = '10', horizontalalignment = 'left')
        plt.text(y = 140, x = scalebar_xcoords[0] * ncols, s = "Radius = " + str(true_radius)+ " " + r'$\mu$' + "m",
                 color = 'red', fontsize = '10', horizontalalignment = 'left')

    if show_particles == True:
         _circle_particles(particle_df, axes, exo_toggle)
    if show_fibers == True:
        def fiber_points(particle_df, axes):
            for v1 in particle_df.vertex1:
                v1point = plt.Circle((v1[1], v1[0]), 0.5,
                                      color = 'red', linewidth = 0,
                                      fill = True, alpha = 1)
                axes.add_patch(v1point)
            for v2 in particle_df.vertex2:
                v2point = plt.Circle((v2[1], v2[0]), 0.5,
                                      color = 'm', linewidth = 0,
                                      fill = True, alpha = 1)
                axes.add_patch(v2point)
            # for centroid in particle_df.centroid:
            #     centpoint = plt.Circle((centroid[1], centroid[0]), 2,
            #                             color = 'g', fill = False, alpha = 1)
            #     axes.add_patch(centpoint)
        fiber_points(particle_df, axes)
    if (show_filaments == True) & (not filo_df.empty):
        for v1 in filo_df.vertex1:
            v1point = plt.Circle((v1[1], v1[0]), 0.5,
                                  color = 'red', linewidth = 0,
                                  fill = True, alpha = 1)
            axes.add_patch(v1point)
        for v2 in filo_df.vertex2:
            v2point = plt.Circle((v2[1], v2[0]), 0.5,
                                  color = 'm', linewidth = 0,
                                  fill = True, alpha = 1)
            axes.add_patch(v2point)
        for box in filo_df.bbox_verts:
            low_left_xy = (box[3][1]-1, box[3][0]-1)
            h = box[0][0] - box[2][0]
            w = box[1][1] - box[0][1]
            filobox = plt.Rectangle(low_left_xy, w, h, fill = False, ec = 'm', lw = 0.5, alpha = 0.8)
            axes.add_patch(filobox)

    if show_markers == True:
        for coords in markers:
            # mark_cm = plt.scatter(x = coords[1], y=coords[0],s=10, c = 'g',marker='+')
            mark_box = plt.Rectangle((coords[1]-58,coords[0]-78), 114, 154,
                                  fill = False, ec = 'green', lw = 1)
            # axes.add_patch(mark_cm)
            axes.add_patch(mark_box)
    if crosshairs == True:
        plt.axhline(y = cy, color = 'red', linewidth = 3)
        plt.axvline(x = cx, color = 'red', linewidth = 3)

    plt.savefig('../virago_output/' + chip_name + '/processed_images/' + im_name +'.png', dpi = dpi)
    print("Processed image generated: " + im_name + ".png")
    if show_image == True:
        plt.show()
    plt.clf(); plt.close('all')
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

    pic_cdf1, cbins1 = exposure.cumulative_distribution(fig1, bin_no)
    pic_cdf2, cbins2 = exposure.cumulative_distribution(fig2, bin_no)
    pic_cdf3, cbins3 = exposure.cumulative_distribution(fig3, bin_no)

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
def gen_particle_image(pic_to_show, shape_df, spot_coords, pix_per_um, cv_cutoff,
                       show_particles=True, scalebar = 0, markers = []):
    nrows, ncols = pic_to_show.shape
    cx,cy,rad = spot_coords
    true_radius = round((rad - 20) / pix_per_um,2)
    dpi = 96
    fig = plt.figure(figsize = (ncols/dpi, nrows/dpi), dpi = dpi)
    axes = plt.Axes(fig,[0,0,1,1])
    fig.add_axes(axes)
    axes.set_axis_off()

    axes.imshow(pic_to_show, cmap = plt.cm.gray)

    ab_spot = plt.Circle((cx, cy), rad, color='#5A81BB',
                         linewidth=5, fill=False, alpha = 0.5
    )
    axes.add_patch(ab_spot)
    if markers != []:
        for coords in markers:
            mark_box = plt.Rectangle((coords[1]-58,coords[0]-78), 114, 154,
                                      fill = False, ec = 'green', lw = 1)
            axes.add_patch(mark_box)
    if show_particles == True:
        patch_settings = dict(fill=False, linewidth=1)
        line_settings = dict(lw=1,color='purple',alpha=0.6)

        total_passes = np.max(shape_df.pass_number)

        for val in shape_df.index.values:
            filo_score = shape_df.filo_score.loc[val]
            pass_num = shape_df.pass_number.loc[val]



            if (pass_num == total_passes):
                color = 'c'
                alpha = 0.75
            else:
                color = 'r'
                alpha = 0.45

            box = shape_df.bbox_verts.loc[val]
            # low_left_xy   =   (box[3][1], box[3][0])
            # up_left_xy  =   (box[0][1], box[0][0])
            # low_rt_xy   =   (box[2][1], box[2][0])
            # up_rt_xy    =   (box[1][1], box[1][0])
            if shape_df.cv_bg.loc[val] >= cv_cutoff:
                line1 = lines.Line2D([box[3][1],box[1][1]],[box[3][0],box[1][0]], **line_settings)
                line2 = lines.Line2D([box[0][1],box[2][1]],[box[0][0],box[2][0]], **line_settings)
                axes.add_line(line1)
                axes.add_line(line2)

            else:
                pc = shape_df.perc_contrast.loc[val]
                centroid = shape_df.centroid.loc[val]
                # round_circ = plt.Circle((centroid[1], centroid[0]),
                #                          pc*3,
                #                          color=color, alpha = alpha,
                #                          **patch_settings
                # )
                # axes.add_patch(round_circ)
                axes.text(y = centroid[0], x = centroid[1], s = str(round(pc,2)),
                         color = 'red', fontsize = '5', horizontalalignment = 'right')

    if scalebar > 0:
        scalebar_len_pix = pix_per_um * scalebar
        scalebar_len = scalebar_len_pix / ncols
        scalebar_xcoords = ((0.98 - scalebar_len), 0.98)
        scale_text_xloc = np.mean(scalebar_xcoords) * ncols
        axes.axhline(y=100, xmin=scalebar_xcoords[0], xmax=scalebar_xcoords[1],
                    linewidth = 8, color = "red")
        axes.text(y = 85, x = scale_text_xloc, s = (str(scalebar)+' ' + r'$\mu$' + 'm'),
                 color = 'red', fontsize = '20', horizontalalignment = 'center')
        # axes.text(y = 120, x = scalebar_xcoords[0] * ncols, s = im_name,
        #          color = 'red', fontsize = '10', horizontalalignment = 'left')
        # axes.text(y = 140, x = scalebar_xcoords[0] * ncols, s = "Radius = {} " + r'$\mu$' + "m".format(true_radius),
        #              color = 'red', fontsize = '10', horizontalalignment = 'left')


#*********************************************************************************************#
def histogrammer(particle_dict, spot_counter, cont_window, baselined = True):
    """Returns a DataFrame of histogram data from the particle dictionary. If baselined = True, returns a DataFrame where the histogram data has had pass 1 values subtracted for all spots"""
    baseline_histogram_df, histogram_df =  pd.DataFrame(), pd.DataFrame()
    cont_0 = float(cont_window[0])
    cont_1 = float(cont_window[1])
    bin_no = int((cont_1 - cont_0) * 10)
    for x in range(1,spot_counter+1):
        hist_df, base_hist_df = pd.DataFrame(), pd.DataFrame()
        histo_listo = [particle_dict[key]
                       for key in sorted(particle_dict.keys())
                       if int(key.split(".")[0]) == x]
        for y, vals in enumerate(histo_listo):
            hist_df[str(x)+'_'+str(y+1)], hbins = np.histogram(vals,
                                                            bins = bin_no,
                                                            range = (cont_0,cont_1))

        for col in hist_df:
            zerocheck = (hist_df[col].nonzero() or hist_df[col].notna())
            if not zerocheck[0].size:
                hist_df.drop(columns=col, inplace=True)
                # hist_df[col].replace(0, np.nan, inplace = True)
            else:
                spot_num = str(col.split('_')[0])
                try:
                    base_hist_df[col] = (hist_df[col] - hist_df[spot_num+'_'+str(1)])
                except KeyError:
                    hist_df.drop(columns=col, inplace=True)
                    print("Missing baseline data for spot {}".format(spot_num))


        baseline_histogram_df = pd.concat([baseline_histogram_df, base_hist_df], axis = 1)
        histogram_df = pd.concat([histogram_df, hist_df], axis = 1)
    baseline_histogram_df['bins'] = hbins[:-1]
    histogram_df['bins'] = hbins[:-1]
    if baselined == False:
        return histogram_df
    else:
        return baseline_histogram_df
#*********************************************************************************************#
def histogram_averager(histogram_df, mAb_dict_rev, pass_counter, smooth = False, sm_window = 7):
    """Returns an DataFrame representing the average histogram of all spots of the same type."""
    mean_histo_df = histogram_df[['bins']]
    # smooth_histo_df = mean_histo_df.copy()
    for key in mAb_dict_rev:
        mAb_split_df, mean_spot_df, smooth_spot_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for col in histogram_df:
            if str(col).replace('_','').isdigit():
                spot_num = int(col.split("_")[0])
                if spot_num in mAb_dict_rev[key]:
                    mAb_split_df = pd.concat((mAb_split_df,histogram_df[col]), axis = 1)
                    named_col = key+'_'+str(col)
                    histogram_df.rename({col:named_col}, axis='columns', inplace=True)

        for i in range(1,pass_counter+1):
            pass_df = pd.DataFrame()
            for col in mAb_split_df:
                pass_num = int(col.split("_")[1])
                if pass_num == i:
                    pass_df = pd.concat((pass_df, mAb_split_df[col]), axis = 1)
            # print(pass_df)
            new_name = key +'_'+ str(i)
            mean_pass_df = (pass_df.mean(axis = 1)).rename(new_name)
                # print(col)
                    # print(mean_pass_df)
                    # print(col)
            mean_spot_df = pd.concat((mean_spot_df, mean_pass_df), axis = 1)

            if smooth == True:
                smooth_pass_df = mean_pass_df.rolling(window = sm_window,
                                                      center = True).mean().rename('sm-'+new_name)
                smooth_spot_df = pd.concat((smooth_spot_df, smooth_pass_df), axis = 1)
                        # def smoothing(mean_pass_df, val = 5):
                        #     smooth_pass_list = []
                        #     for x in range(val, len(mean_pass_df) - val-1):
                        #         print(x)
                        #         smooth_val = np.mean(mean_pass_df[x-val:x+val+1])
                        #         smooth_pass_list.append(smooth_val)

                # print(mean_spot_df)
        mean_histo_df = pd.concat((mean_histo_df, mean_spot_df), axis = 1)
        if smooth == True:
            mean_histo_df = pd.concat((mean_histo_df, smooth_spot_df), axis = 1)

    return mean_histo_df
#*********************************************************************************************#
def generate_combo_hist(mean_histo_df, chip_name, pass_counter,
                        cont_window, cmap, baselined = True, savedir = ''):
    """Generates a histogram figure for each pass in the IRIS experiment from a DataFrame representing the average data for every spot type"""

    cont_str = '{0}-{1}'.format(*cont_window)
    bin_series = mean_histo_df.pop('bins')
    smooth_histo_df = mean_histo_df.filter(regex='sm-')

    histo_min = int(np.min(np.min(mean_histo_df)))
    histo_max = int(np.max(np.max(mean_histo_df)))
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
    for i in range(2, pass_counter+1, passes_to_show):
        sns.set(style='darkgrid')
        c = 0
        for col in smooth_histo_df:
            spot_type = col.split("_")[0]
            pass_num = int(col.split("_")[-1])

            if pass_num == i:

                plt.plot(bin_series,
                         smooth_histo_df[col],
                         linewidth = 2,
                         color = cmap[c],
                         alpha = 0.75,
                         label = spot_type)


                c += 1
            # plt.clf()
        if baselined == True:
            plt.title(chip_name+" Pass "+str(i)+" Average Histograms - Baseline Subtracted")
        else:
            plt.title(chip_name+" Pass "+str(i)+" Average Histograms")
        plt.axhline(y=0, ls='dotted', c='black', alpha=0.75)
        plt.legend(loc = 'best', fontsize = 14)
        plt.ylabel("Particle Count", size = 14)

        plt.yticks(range(y_min,y_max+5,y_grid), size = 12)
        plt.xlabel("Percent Contrast", size = 14)
        if (len(bin_series) >= 100) & (len(bin_series) < 200): x_grid = 10
        elif len(bin_series) >= 200: x_grid = 20
        else: x_grid = 5
        plt.xticks(bin_series[::x_grid], size = 12, rotation = 30)

        figname = ('{}_combohisto_pass_{}_contrast_{}.png'.format(chip_name,i,cont_str))
        plt.savefig('{}/{}'.format(savedir,figname), bbox_inches = 'tight', dpi = 150)
        print("File generated: {}".format(figname))
        plt.clf()
#*********************************************************************************************#
def average_spot_data(spot_df, spot_tuple, pass_counter):
    """Creates a dataframe containing the average data for each antibody spot type"""
    averaged_df = pd.DataFrame()
    for i, spot in enumerate(spot_tuple):
        sub_df = spot_df[(spot_df.spot_type == spot_tuple[i]) & (spot_df.valid == True)]
        avg_time, avg_kpd, avg_nd, std_kpd, std_nd = [],[],[],[],[]
        for i in range(1,pass_counter+1):
            subsub_df = sub_df[sub_df.scan_number == i]
            avg_time.append(
                     round(
                     np.nanmean(subsub_df.scan_time.iloc[subsub_df.scan_time.nonzero()]),2))
            avg_kpd.append(round(np.nanmean(subsub_df.kparticle_density),2))
            std_kpd.append(round(np.nanstd(subsub_df.kparticle_density),3))
            avg_nd.append(round(np.nanmean(subsub_df.normalized_density),2))
            std_nd.append(round(np.nanstd(subsub_df.normalized_density),3))
        avg_df = pd.DataFrame({
                              'scan_number': np.arange(1,pass_counter+1),
                              'spot_type': [spot]* pass_counter,
                              'avg_time': avg_time,
                              'avg_density': avg_kpd,
                              'std_density': std_kpd,
                              'avg_norm_density':avg_nd,
                              'std_norm_density':std_nd
                              })
        averaged_df = averaged_df.append(avg_df).reset_index(drop=True)

    return averaged_df
#*********************************************************************************************#
def dict_joy_trans(particle_dict, spot_counter):
    """generates a dictionary for creating a joyplot-type histogram"""
    big_df = pd.DataFrame()
    for key in particle_dict.keys():
        spot_num = int(key.split(".")[0])
        pass_num = key.split(".")[1]
        if spot_num == spot_counter:
            small_df = pd.DataFrame()
            small_df['Percent_Contrast'] = particle_dict[key]
            small_df['scan_ID'] = np.array(['SCAN_' + pass_num] * len(particle_dict[key]))
            big_df = big_df.append(small_df, ignore_index=True)

    return big_df
#*********************************************************************************************#
def generate_joyplot(joy_df, spot_counter, cont_window, chip_name, savedir =''):
    """Generates a joyplot-style histogram showing how the peak changes for each scan of a spot"""
    scans = sorted(list(set(joy_df.scan_ID)), reverse = True)

    num_graphs = len(scans)
    if num_graphs > 10:
        scans_to_show = range(1,num_graphs, num_graphs//10)
    max_contrast = int(cont_window[1])
    sns.set(style=("white"), rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.cubehelix_palette(num_graphs, rot=-.25, light=.7)
    g = sns.FacetGrid(joy_df, row="scan_ID",
                      hue="scan_ID", aspect=(1.5 * num_graphs), size= (5 / num_graphs),
                      palette=pal, row_order = scans)

    # Draw the densities in a few steps
    # g.map(sns.distplot, 'Percent_Contrast',  color = 'b', bins=60, norm_hist = False)
    g.map(sns.distplot, 'Percent_Contrast', bins=max_contrast*10, norm_hist=False, kde = False,
        hist_kws={"ec": 'k'},
        # kde_kws={"bw":0.1}
        ).set(xlim=(0, max_contrast),xticks=np.arange(0,max_contrast+1,2))

    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0.9, 0.2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, 'Percent_Contrast')

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.5)

    # Remove axes details that don't play will with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)


    figname = (chip_name+'_series-histogram_spot-'+str(spot_counter)+'.png')
    g.savefig('{}/{}'.format(savedir,figname), bbox_inches = 'tight', dpi = 150)
    print("File generated: {}".format(figname))
#*********************************************************************************************#
def generate_timeseries(spot_df, averaged_df, mAb_dict, spot_tuple,
                         chip_name, sample_name, vhf_colormap, cont_window, version,
                         scan_or_time = 'scan', baseline = True, savedir=''):
    """Generates a timeseries for the cumulate particle counts for each spot, and plots the average
    for each spot type"""
    # baseline_toggle = input("Do you want the time series chart normalized to baseline? ([y]/n)\t")
    # assert isinstance(baseline_toggle, str)
    # if baseline_toggle.lower() in ('no', 'n'):

    if baseline == True: filt_toggle = 'normalized_density'
    else: filt_toggle = 'kparticle_density'

    sns.set(style="ticks")
    fig = plt.figure(figsize = (8,6))
    ax1 = fig.add_subplot(111)

    for key in mAb_dict.keys():
        if key == 1: c = 0
        elif (mAb_dict[key-1] != mAb_dict[key]): c += 1

        solo_spot_df = spot_df[(spot_df.spot_number == key)
                                & (spot_df.valid == True)].reset_index(drop = True)
        if not solo_spot_df.empty:
            if scan_or_time == 'scan':
                x_axis = solo_spot_df['scan_number']
            elif scan_or_time == 'time':
                x_axis = solo_spot_df['scan_time']
            density_y = solo_spot_df[filt_toggle]
            ax1.plot(x_axis, density_y, lw = 1, c = vhf_colormap[c], alpha = 0.5, label = '_nolegend_')

    ax2 = fig.add_subplot(111)
    for n, spot in enumerate(spot_tuple):
        avg_data = averaged_df[averaged_df['spot_type'] == spot]
        if scan_or_time == 'scan': avg_x = avg_data['scan_number']
        else: avg_x = avg_data['avg_time']

        ax2.errorbar(avg_x, avg_data['avg_norm_density'],
                        yerr = avg_data['std_norm_density'], label = spot_tuple[n],
                        lw = 2, elinewidth = 1,
                        c = vhf_colormap[n], aa = True)

    ax2.legend(loc = 'upper left', fontsize = 12, ncol = 1)
    if max(spot_df.scan_number) < 10: x_grid = 1
    else: x_grid = max(spot_df.scan_number) // 10
    if scan_or_time == 'scan':
        plt.xlabel("Scan Number", size = 14)
        plt.xticks(np.arange(1, max(spot_df.scan_number) + 1, x_grid), size = 12, rotation = 30)

    elif scan_or_time == 'time':
        plt.xlabel("Time (min)", size = 14)
        plt.xticks(np.arange(0, max(spot_df.scan_time) + 1, 5), size = 12, rotation = 30)
    cont_str = '{0}-{1}'.format(*cont_window)
    plt.ylabel("Particle Density (kparticles/sq. mm)\n {} % Contrast".format(cont_str), size = 12)
    plt.yticks(color = 'k', size = 12)
    plt.title("{} Time Series of {} - v{}".format(chip_name, sample_name, version), size = 14)
    plt.axhline(linestyle = '--', color = 'gray')

    plot_name = "{}_timeseries.{}contrast.v{}.png".format(chip_name, cont_str,version)

    plt.savefig('{}/{}'.format(savedir, plot_name),
                bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
    print('File generated: {}'.format(plot_name))
    plt.clf(); plt.close('all')
#*********************************************************************************************#
def generate_barplot(spot_df, pass_counter, cont_window, chip_name, sample_name, version, savedir=''):
    """
    Generates a barplot for the dataset.
    Most useful for before and after scans (pass_counter == 2)
    """
    firstlast_spot_df = spot_df[(spot_df.scan_number == 1) | (spot_df.scan_number == pass_counter)]
    cont_str = '{0}-{1}'.format(*cont_window)
    final_spot_df = firstlast_spot_df[firstlast_spot_df.scan_number == pass_counter]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    sns.set(style = 'darkgrid', font_scale = 0.75)

    sns.barplot(y='kparticle_density',x='spot_type',hue='scan_number',data=firstlast_spot_df,
                 errwidth = 2, ci = 'sd', ax=ax1)
    ax1.set_ylabel("Particle Density (kparticles/sq.mm)\n"+"Contrast = "+cont_str+ '%', fontsize = 10)
    ax1.set_xlabel("Prescan & Postscan", fontsize = 8)

    sns.barplot(y='normalized_density',x='spot_type',data=final_spot_df, color ='purple',
              errwidth = 2, ci = 'sd', ax=ax2)
    ax2.set_ylabel("")
    ax2.set_xlabel("Difference", fontsize = 8)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=30, fontsize = 6)

    plt.suptitle("{} {} - v{}".format(chip_name,sample_name,version), y = 1.04, fontsize = 14)

    plt.tight_layout()
    plot_name = "{}_barplot.{}contrast.v{}.png".format(chip_name, cont_str, version)

    plt.savefig('{}/{}'.format(savedir,plot_name),
                bbox_inches = 'tight', pad_inches = 0.1, dpi = 300)
    print('File generated: {}'.format(plot_name))
    plt.clf(); plt.close('all')
#*********************************************************************************************#
