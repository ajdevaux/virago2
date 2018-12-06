from future.builtins import input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Polygon, Circle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgba_array
from skimage import io as skio
from skimage.exposure import rescale_intensity
import seaborn as sns
import warnings
from modules.vimage import gen_img

#*********************************************************************************************#
def _color_mixer(zlen,c1,c2,c3,c4):
    """A function to create color gradients from 4 input colors"""
    if zlen > 1:
        cmix_r1=np.linspace(c1[0],c2[0],int(zlen//2),dtype=np.float16)
        cmix_g1=np.linspace(c1[1],c2[1],int(zlen//2),dtype=np.float16)
        cmix_b1=np.linspace(c1[2],c2[2],int(zlen//2),dtype=np.float16)
        cmix_r2=np.linspace(c3[0],c4[0],int(zlen//2),dtype=np.float16)
        cmix_g2=np.linspace(c3[1],c4[1],int(zlen//2),dtype=np.float16)
        cmix_b2=np.linspace(c3[2],c4[2],int(zlen//2),dtype=np.float16)
        cnew1=[(cmix_r1[c], cmix_g1[c], cmix_b1[c]) for c in range(0,(zlen)//2,1)]
        cnew2=[(cmix_r2[c], cmix_g2[c], cmix_b2[c]) for c in range(0,(zlen)//2,1)]
        cnew3=[(np.mean(list([c2[0],c3[0]]),dtype=np.float16),
                  np.mean(list([c2[1],c3[1]]),dtype=np.float16),
                  np.mean(list([c2[2],c3[2]]),dtype=np.float16))]
        color_list=cnew1 + cnew3 + cnew2
    else:
        color_list=['white']
    return color_list
#*********************************************************************************************#
def _circle_particles(particle_df, axes, exo_toggle):
    z_list=[z for z in list(set(particle_df.z))]# if str(z).isdigit()]
    zlen=len(z_list)
    dark_red=(0.645, 0, 0.148); pale_yellow=(0.996, 0.996, 0.746)
    pale_blue=(0.875, 0.949, 0.969); dark_blue=(0.191, 0.211, 0.582)
    blueflame_cm=_color_mixer(zlen, c1=dark_red, c2=pale_yellow, c3=pale_blue, c4=dark_blue)
    pc_hist=list()
    ax_hist=plt.axes([.7, .06, .25, .25])
    hist_max=6
    for c, zslice in enumerate(z_list):
        circ_color=blueflame_cm[c]
        y=particle_df.loc[particle_df.z == zslice].y.reset_index(drop=True)
        x=particle_df.loc[particle_df.z == zslice].x.reset_index(drop=True)
        pc=particle_df.loc[particle_df.z == zslice].pc.reset_index(drop=True)
        try:
            if max(pc) > hist_max: hist_max=max(pc)
        except: ValueError
        if exo_toggle == True: crad=0.2
        else: crad=2
        # try:
        #     if max(pc) > 25: crad=0.25
        # except: ValueError
        pc_hist.append(np.array(pc))
        for i in range(0,len(pc)):
            point=plt.Circle((x[i], y[i]), pc[i] * crad,
                                color=circ_color, linewidth=1,
                                fill=False, alpha=0.75)
            axes.add_patch(point)

    hist_color=blueflame_cm[:len(pc_hist)]
    hist_vals, hbins, hist_patches=ax_hist.hist(pc_hist, bins=200, range=[0,30],
                                                  linewidth=2, alpha=0.5, stacked=True,
                                                  color=hist_color,
                                                  label=z_list)
    ax_hist.patch.set_alpha(0.5)
    ax_hist.patch.set_facecolor('black')
    ax_hist.legend(loc='best', fontsize=8)
    if exo_toggle == True: ax_hist.set_xlim([0,25])
    else: ax_hist.set_xlim([0,15])

    for spine in ax_hist.spines: ax_hist.spines[spine].set_color('k')
    ax_hist.tick_params(color='k')
    plt.xticks(size=10, color='w')
    plt.xlabel("% CONTRAST", size=12, color='w')
    plt.yticks(size=10, color='w')
    plt.ylabel("PARTICLE COUNT", color='w')
#*********************************************************************************************#
def gen_particle_image(pic_to_show, shape_df, spot_coords, pix_per_um, cv_cutoff,
                        show_particles=True, scalebar=0, markers=[], exo_toggle=False):
    nrows, ncols=pic_to_show.shape
    try: current_pass = max(shape_df.pass_number)
    except ValueError: current_pass = 0
    curr_pass_df = shape_df[shape_df.pass_number == current_pass]

    cx,cy,rad=spot_coords
    true_radius=round((rad - 20) / pix_per_um,2)
    dpi=96
    fig=plt.figure(figsize=(ncols/dpi, nrows/dpi), dpi=dpi)
    axes=plt.Axes(fig,[0,0,1,1])
    fig.add_axes(axes)
    axes.set_axis_off()

    axes.imshow(pic_to_show, cmap=plt.cm.gray)

    ab_spot=plt.Circle((cx, cy), rad, color='#5A81BB',
                         linewidth=5, fill=False, alpha=0.5
    )
    axes.add_patch(ab_spot)
    if markers != []:
        for coords in markers:
            mark_box=plt.Rectangle((coords[1]-58,coords[0]-78), 114, 154,
                                      fill=False, ec='green', lw=1)
            axes.add_patch(mark_box)
    if (show_particles == True) & (current_pass > 0):
        patch_settings=dict(fill=False, linewidth=1, alpha=0.75)
        line_settings=dict(lw=1,color='purple',alpha=0.6)

        # total_passes=np.max(shape_df.pass_number)

        for val in curr_pass_df.index.values:
            filo_score=curr_pass_df.filo_score.loc[val]
            # pass_num=curr_pass_df.pass_number.loc[val]
            #
            # if (pass_num == total_passes):
            #     color='c'
            #     alpha=0.75
            # else:
            #     color='r'
            #     alpha=0.45

            box=curr_pass_df.bbox_verts.loc[val]
            # low_left_xy  =  (box[3][1], box[3][0])
            # up_left_xy =  (box[0][1], box[0][0])
            # low_rt_xy  =  (box[2][1], box[2][0])
            # up_rt_xy   =  (box[1][1], box[1][0])
            if curr_pass_df.cv_bg.loc[val] >= cv_cutoff:
                line1=lines.Line2D([box[3][1],box[1][1]],[box[3][0],box[1][0]], **line_settings)
                line2=lines.Line2D([box[0][1],box[2][1]],[box[0][0],box[2][0]], **line_settings)
                axes.add_line(line1)
                axes.add_line(line2)




            else:
                pc=curr_pass_df.perc_contrast.loc[val]
                centroid=curr_pass_df.centroid.loc[val]
                chisq=curr_pass_df.chisq.loc[val]
                if exo_toggle == True: circ_radius = pc*0.5
                else: circ_radius = pc*3
                if chisq < 0.075: chicolor='cyan'
                else: chicolor='red'

                round_circ=plt.Circle((centroid[1], centroid[0]),
                                      circ_radius, color=chicolor, **patch_settings
                )
                axes.add_patch(round_circ)
                axes.text(y=centroid[0], x=centroid[1], s=str(round(pc,2)),
                         color='red', fontsize='8', horizontalalignment='right')

    if scalebar > 0:
        scalebar_len_pix=pix_per_um * scalebar
        scalebar_len=scalebar_len_pix / ncols
        scalebar_xcoords=((0.98 - scalebar_len), 0.98)
        scale_text_xloc=np.mean(scalebar_xcoords) * ncols
        axes.axhline(y=100, xmin=scalebar_xcoords[0], xmax=scalebar_xcoords[1],
                    linewidth=8, color="red")
        axes.text(y=85, x=scale_text_xloc, s=(str(scalebar)+' ' + r'$\mu$' + 'm'),
                 color='red', fontsize='20', horizontalalignment='center')
        # axes.text(y=120, x=scalebar_xcoords[0] * ncols, s=im_name,
        #          color='red', fontsize='10', horizontalalignment='left')
        # axes.text(y=140, x=scalebar_xcoords[0] * ncols, s="Radius={} " + r'$\mu$' + "m".format(true_radius),
        #              color='red', fontsize='10', horizontalalignment='left')


#*********************************************************************************************#
def histogrammer(particle_dict, spot_counter, cont_window, norm_to_area=True):
    """Returns a DataFrame of histogram data from the particle dictionary. If baselined=True, returns a DataFrame where the histogram data has had pass 1 values subtracted for all spots"""
    histogram_df= pd.DataFrame()
    # cont_0=float(cont_window[0])
    cont_0 = 0
    cont_1=float(cont_window[1])
    bin_no=int((cont_1 - cont_0) * 10)
    area_list = [float(key.split('.')[-1])*1e-6 for key in particle_dict.keys()]

    for x in range(1,spot_counter+1):
        hist_df, base_hist_df=pd.DataFrame(), pd.DataFrame()
        histo_listo=[particle_dict[key]
                     for key in sorted(particle_dict.keys())
                     if int(key.split(".")[0]) == x
        ]
        for y, vals in enumerate(histo_listo):
            hist_df[str(x)+'_'+str(y+1)], hbins=np.histogram(vals,
                                                             bins=bin_no,
                                                             range=(cont_0,cont_1)
            )

        histogram_df=pd.concat([histogram_df, hist_df], axis=1)

    if norm_to_area == True:
        for i, col in enumerate(histogram_df.columns):
            histogram_df[col] = round((histogram_df[col] / area_list[i])*1e-3, 5)

    for col in histogram_df:
        nancheck = np.all(np.isnan(histogram_df[col]))
        if nancheck == True:
            histogram_df.drop(columns=col, inplace=True)

    histogram_df['bins']=hbins[:-1]

    return histogram_df

#*********************************************************************************************#
def sum_histogram(raw_histogram_df, spot_counter, pass_counter):
    sum_histogram_df=raw_histogram_df.pop('bins')

    for y in range(1, spot_counter+1):
        spot_histogram_df = pd.DataFrame()
        for col in raw_histogram_df:
            if int(col.split('_')[0]) == y:
                spot_histogram_df=pd.concat([spot_histogram_df, raw_histogram_df[col]], axis=1)

        pass_list = [col.split('_')[1] for col in spot_histogram_df]
        if not pass_list == []:
            init_scan = int(min(pass_list))
            init_scan_cols=[col for col in spot_histogram_df.columns if int(col.split('_')[1]) == init_scan]
            spot_histogram_df.drop(init_scan_cols, axis=1, inplace=True)
            for i in pass_list[1:]:
                new_name=str(y)+'_'+str(i)
                cols_to_sum=[col for col in spot_histogram_df if int(col.split('_')[1]) <= int(i)]
                sum_histogram_df= pd.concat([sum_histogram_df,
                                         spot_histogram_df[cols_to_sum].sum(axis=1).rename(new_name)], axis=1
                )

    return sum_histogram_df
#*********************************************************************************************#
def average_histogram(sum_histogram_df, spot_df, pass_counter, smooth_window=7):
    avg_histogram_df=sum_histogram_df['bins']
    spot_type_list = []
    spot_df = spot_df[spot_df.valid==True]
    for val in spot_df.spot_type:
        if val not in spot_type_list: spot_type_list.append(val)
    for x in range(2,pass_counter+1):
        pass_histogram_df=pd.DataFrame()
        for col in sum_histogram_df.columns[1:]:
            if int(col.split('_')[1]) == x:
                pass_histogram_df=pd.concat([pass_histogram_df, sum_histogram_df[col]], axis=1)
        for spot_type in spot_type_list:
            cols_to_avg=[col for col in pass_histogram_df
                               if int(col.split('_')[0])
                               in spot_df.spot_number[(spot_df.spot_type == spot_type)].tolist()
            ]


            sqrt_n = np.sqrt(len(cols_to_avg))
            new_name=spot_type+'_'+str(x)
            mean_df=sum_histogram_df[cols_to_avg].mean(axis=1).rename(new_name)
            sdm_df=sum_histogram_df[cols_to_avg].std(axis=1).apply(lambda x: x/sqrt_n).rename(new_name + '_sdm')
            avg_histogram_df=pd.concat([avg_histogram_df, mean_df, sdm_df], axis=1)

    for col in avg_histogram_df.columns:
        if col[-1].isdigit():
            smooth_df=avg_histogram_df[col].rolling(window=smooth_window,
                                                    center=True).mean().rename(col+'_rollingmean')

            avg_histogram_df=pd.concat([avg_histogram_df,smooth_df], axis=1)

    return avg_histogram_df
#*********************************************************************************************#
def average_spot_data(spot_df, pass_counter):
    """Creates a dataframe containing the average data for each antibody spot type"""
    averaged_df=pd.DataFrame()
    spot_list=[]
    for val in spot_df.spot_type:
        if val not in spot_list:
            spot_list.append(val)

    for i, spot in enumerate(spot_list):
        sub_df=spot_df[(spot_df.spot_type == spot_list[i]) & (spot_df.valid == True)]
        avg_time, avg_kpd, avg_nd, std_kpd, std_nd=[],[],[],[],[]
        for i in range(1,pass_counter+1):
            subsub_df=sub_df[sub_df.scan_number == i]
            avg_time.append(
                     round(
                     np.nanmean(subsub_df.scan_time.iloc[subsub_df.scan_time.nonzero()]),2))
            avg_kpd.append(round(np.nanmean(subsub_df.kparticle_density),2))
            std_kpd.append(round(np.nanstd(subsub_df.kparticle_density),3))
            avg_nd.append(round(np.nanmean(subsub_df.normalized_density),2))
            std_nd.append(round(np.nanstd(subsub_df.normalized_density),3))
        avg_df=pd.DataFrame({
                              'scan_number': np.arange(1,pass_counter+1),
                              'spot_type': [spot]* pass_counter,
                              'avg_time': avg_time,
                              'avg_density': avg_kpd,
                              'std_density': std_kpd,
                              'avg_norm_density':avg_nd,
                              'std_norm_density':std_nd
                              })
        averaged_df=averaged_df.append(avg_df).reset_index(drop=True)

    return averaged_df
#*********************************************************************************************#
def generate_timeseries(spot_df, averaged_df, cont_window, mAb_dict,
                        chip_name, sample_name, vhf_colormap, version,
                         scan_or_time='scan', baseline=True, savedir=''):
    """Generates a timeseries for the cumulate particle counts for each spot, and plots the average
    for each spot type"""
    # baseline_toggle=input("Do you want the time series chart normalized to baseline? ([y]/n)\t")
    # assert isinstance(baseline_toggle, str)
    # if baseline_toggle.lower() in ('no', 'n'):

    if baseline == True: filt_toggle='normalized_density'
    else: filt_toggle='kparticle_density'
    spot_list=[]
    for val in spot_df.spot_type:
        if val not in spot_list:
            spot_list.append(val)

    sns.set(style="ticks")
    fig=plt.figure(figsize=(8,6))
    ax1=fig.add_subplot(111)

    for key in mAb_dict.keys():
        print(key)
        if key == 1:
            c = 0
        elif (mAb_dict[key-1][0] != mAb_dict[key][0]):
            c += 1

        solo_spot_df=spot_df[(spot_df.spot_number == key)
                                & (spot_df.valid == True)].reset_index(drop=True)
        print(solo_spot_df)
        if not solo_spot_df.empty:
            if scan_or_time == 'scan':   x_axis=solo_spot_df['scan_number']
            elif scan_or_time == 'time': x_axis=solo_spot_df['scan_time']

            ax1.plot(x_axis, solo_spot_df[filt_toggle], lw=1, c=vhf_colormap[c], alpha=0.5, label='_nolegend_')

    ax2=fig.add_subplot(111)
    for n, spot in enumerate(spot_list):
        avg_data=averaged_df[averaged_df['spot_type'] == spot]
        if scan_or_time == 'scan': avg_x=avg_data['scan_number']
        else: avg_x=avg_data['avg_time']

        ax2.errorbar(avg_x, avg_data['avg_norm_density'],
                        yerr=avg_data['std_norm_density'], label=spot_list[n],
                        lw=2, elinewidth=1,
                        c=vhf_colormap[n], aa=True)

    ax2.legend(loc='upper left', fontsize=24, ncol=1)
    if max(spot_df.scan_number) < 10: x_grid=1
    else: x_grid=max(spot_df.scan_number) // 10

    if scan_or_time == 'scan':
        plt.xlabel("Scan Number", size=24)
        plt.xticks(np.arange(1, max(spot_df.scan_number) + 1, x_grid), size=24, rotation=30)

    elif scan_or_time == 'time':
        plt.xlabel("Time (min)", size=24)
        plt.xticks(np.arange(0, max(spot_df.scan_time) + 1, 5), size=24, rotation=30)
    cont_str='{0}-{1}'.format(*cont_window)
    plt.ylabel("Particle Density (kparticles/mm" + r'$^2$'+")\n {} % Contrast".format(cont_str), size=24)
    plt.yticks(color='k', size=24)
    plt.title("{} Time Series of {} - v{}".format(chip_name, sample_name, version), size=28)
    plt.axhline(linestyle='--', color='gray')

    plot_name="{}_timeseries.{}contrast.v{}.png".format(chip_name, cont_str,version)

    plt.savefig('{}/{}'.format(savedir, plot_name),
                bbox_inches='tight', pad_inches=0.1)#, dpi=300)
    print('File generated: {}'.format(plot_name))
    plt.clf(); plt.close('all')
#*********************************************************************************************#
def generate_barplot(spot_df, pass_counter, cont_window,  chip_name, sample_name, vhf_colormap, version, savedir='', plot_3sigma=False):
    """
    Generates a barplot for the dataset.
    Most useful for before and after scans (pass_counter == 2)
    """
    firstlast_spot_df=spot_df[  ((spot_df.scan_number == 1)
                              | (spot_df.scan_number == pass_counter))
                              & (spot_df.valid == True)
    ]
    cont_str='{0}-{1}'.format(*cont_window)
    final_spot_df=spot_df[  (spot_df.scan_number == pass_counter)
                          & (spot_df.valid == True)
    ]

    fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    sns.set_style('darkgrid')

    sns.barplot(y='kparticle_density',x='spot_type',hue='scan_number',data=firstlast_spot_df,
                 palette = 'binary', errwidth=2, ci='sd', ax=ax1
    )
    ax1.set_ylabel("Particle Density (kparticles/mm" + r'$^2$'+")\n"+"Contrast="+cont_str+ '%', fontsize=12)
    ax1.set_xlabel("Prescan & Postscan", fontsize=12)

    sns.barplot(y='normalized_density',x='spot_type',data=final_spot_df, palette = vhf_colormap,
                errwidth=2, ci='sd', ax=ax2
    )
    ax2.set_ylabel("")
    ax2.set_xlabel("Difference", fontsize=10)
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=30, fontsize=10)
    if plot_3sigma == True:
        neg_control_vals = final_spot_df.normalized_density[final_spot_df.spot_type.str.contains("8G5")]
        neg_control_mean = np.mean(neg_control_vals)
        neg_control_std = np.std(neg_control_vals)
        three_sigma = (neg_control_std * 3) + neg_control_mean
        ax2.axhline(y=three_sigma,ls='--',lw=2,color='r', label='3'+r'$\sigma$'+' Signal Threshold')
    plt.legend(loc='upper right', fontsize=12, ncol=1)

    plt.suptitle("{} {} - v{}".format(chip_name,sample_name,version), y=1.04, fontsize=14)

    plt.tight_layout()
    plot_name="{}_barplot.{}contrast.v{}.png".format(chip_name, cont_str, version)

    plt.savefig('{}/{}'.format(savedir,plot_name),
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    print('File generated: {}'.format(plot_name))
    plt.close('all')
#*********************************************************************************************#
def filo_image_gen(shape_df, pic1, pic2, pic3,
                  ridge_list, sphere_list, ridge_list_s,
                  cv_cutoff=0.02, show=True
):

    fig=plt.figure(figsize=(24, 8))

    ax1=fig.add_subplot(1, 3, 1)
    ax1.imshow(pic1, cmap=plt.cm.magma)
    ax1.axis('off')
    ax1.set_title('Contrast-Adjusted Image', fontsize=18)
    patch_settings=dict(fill=False, linewidth=1, alpha=0.75)
    line_settings=dict(lw=1,color='purple',alpha=0.6)
    scatter_settings=dict(s=12, linewidths=0)

    for val in shape_df.index.values:
        filo_score = shape_df.filo_score.loc[val]
        # perim_area_ratio=shape_df.perim_area_ratio.loc[val]
        # elong=shape_df.elongation.loc[val]
        verts    = shape_df.vertices[val]
        box      = shape_df.bbox_verts.loc[val]
        centroid = shape_df.centroid.loc[val]
        pc       = shape_df.perc_contrast.loc[val]
        centroid = shape_df.centroid.loc[val]
        filo_len = shape_df.filo_lengths.loc[val]

        if (filo_score > 0.2):
            color='r'
        elif (filo_score <= 0.1):
            color='c'
        else: color='y'

        # low_left_xy  =  (box[3][1], box[3][0])
        # up_left_xy =  (box[0][1], box[0][0])
        # low_rt_xy  =  (box[2][1], box[2][0])
        # up_rt_xy   =  (box[1][1], box[1][0])

        # ax1.scatter(w_centroid[1],w_centroid[0], color='g', marker='+')

        # if shape_df.cv_bg.loc[al] >= cv_cutoff:
        # if filo_score < 0.2:
            # line1=lines.Line2D([box[3][1],box[1][1]],[box[3][0],box[1][0]], **line_settings)
            # line2=lines.Line2D([box[0][1],box[2][1]],[box[0][0],box[2][0]], **line_settings)
            # ax1.add_line(line1)
            # ax1.add_line(line2)
        # elif not np.isnan(verts).any():


        if not verts is np.nan:

            # centroid=shape_df.centroid.loc[val]
            # round_circ=plt.Circle((centroid[1], centroid[0]),
            #                          shape_df.perc_contrast.loc[val]*3,
            #                          color=color, **patch_settings
            # )
            # ax1.add_patch(round_circ)
            ax1.scatter(int(verts[0][1]),int(verts[0][0]), color='red', marker='.')
            ax1.scatter(int(verts[1][1]),verts[1][0], color='magenta', marker='+')



            ax1.text(y=centroid[0], x=centroid[1], s=str(round(filo_len,2)),
                     color='red', fontsize='10', horizontalalignment='right')

        else:
            round_circ=plt.Circle((centroid[1], centroid[0]),
                                     pc*0.5,
                                     color=color,
                                     **patch_settings
            )
            ax1.add_patch(round_circ)
            # ax1.scatter(centroid[1],centroid[0], **scatter_settings, color=color, marker='o')

    ax2=fig.add_subplot(1, 3, 2, sharex=ax1, sharey=ax1)

    ax2.imshow(pic2, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Shape Index', fontsize=18)


    ax3=fig.add_subplot(1, 3, 3, sharex=ax1, sharey=ax1)
    ax3.imshow(pic3, plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('Scored Binary Image', fontsize=18)


    ridge_y,ridge_x=zip(*ridge_list)
    ax3.scatter(ridge_x,ridge_y, color='magenta', **scatter_settings, marker='^')

    sphere_y, sphere_x=zip(*sphere_list)
    ax3.scatter(sphere_x, sphere_y, color='cyan', **scatter_settings, marker='o')

    ridge_y_s, ridge_x_s=zip(*ridge_list_s)
    ax3.scatter(ridge_x_s,ridge_y_s, color='purple', **scatter_settings, marker='^')


    fig.tight_layout()
    if show == True:
        plt.show()
    plt.clf()
#*********************************************************************************************#
def chipArray_graph(spot_df, vhf_colormap, chip_name='IRIS chip',
                    sample_name ='',exo_toggle=False, cont_str='', version='',
                    savedir='/Users/dejavu/Desktop'
                    ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    prescan = min(spot_df.scan_number)
    last_scan = max(spot_df.scan_number)
    spot_array = np.array(spot_df.spot_number[1::last_scan])
    total_spots = max(spot_array)

    linx,liny= zip(*(spot_df.chip_coords_xy[1::last_scan]))
    linx = np.array(linx,dtype='float') / 1000
    liny = np.array(liny,dtype='float') / 1000
    ax.plot(linx,liny,0, color='k', linewidth=1, linestyle='--', alpha=0.5)
    for j,spot_num in enumerate(spot_array):
        ax.text(linx[j],liny[j],0, s=str(spot_num),
                fontsize=8, fontname='monospace',color='k', alpha=1,
                horizontalalignment='left', verticalalignment='top',
        )

    spot_list=[]
    for val in spot_df.spot_type:
        if val not in spot_list:
            spot_list.append(val)

    for n, spot in enumerate(spot_list):
        prescan_df = spot_df[(spot_df.scan_number==prescan)&(spot_df.spot_type==spot)].reset_index()
        scan_df = spot_df[(spot_df.scan_number==last_scan)&(spot_df.spot_type==spot)].reset_index()
        xpos,ypos = zip(*(scan_df.chip_coords_xy))
        dx=dy=0.04

        xpos=np.array(xpos,dtype='float') / 1000
        xpos_bar=xpos-(dx/2)
        ypos=np.array(ypos,dtype='float') / 1000
        ypos_bar=ypos-(dy/2)
        zpos=0
        dz=np.array(scan_df.normalized_density)

        rad_list = np.sqrt(np.array(scan_df.area,dtype='float')/np.pi)
        for i,r in enumerate(rad_list):
            Ab_spot=Circle((xpos[i],ypos[i]),r,facecolor=vhf_colormap[n], edgecolor ='k',alpha=0.4)
            ax.add_patch(Ab_spot)
            art3d.pathpatch_2d_to_3d(Ab_spot, z=0, zdir="z")

        bar = ax.bar3d(xpos_bar, ypos_bar, zpos, dx, dy, dz,
                       color=vhf_colormap[n], edgecolor='k', label=spot, alpha=0.5)
        ##bug fixes with matplotlib 3d axes
        bar._facecolors3d=to_rgba_array(bar._facecolors3d, bar._alpha)
        bar._edgecolors3d=to_rgba_array(bar._edgecolors3d, bar._alpha)
        bar._facecolors2d=bar._facecolors3d
        bar._edgecolors2d=bar._edgecolors3d
        ##
    # surface = Polygon(np.array([[-1000,-750],[-1000,6000],[1000,6000],[1000,-750]]), color='gray')
    # ax.add_patch(surface)
    # art3d.pathpatch_2d_to_3d(surface, z=-5, zdir="z")
    tri = Polygon(np.array([[-.25,-.5],[0,0],[.25,-.5]]), color='k')
    ax.add_patch(tri)
    art3d.pathpatch_2d_to_3d(tri, z=0, zdir="z")


    ax.set_xlabel("Lat. Axis (mm)",fontsize=8)
    ax.set_ylabel("Long. Axis (mm)", fontsize=8)

    ax.set_zlabel("Particle Density (kparticles/mm" + r'$^2$'+")\n{}% Contrast".format(cont_str))
    ax.set_title("{} Microarray Bar Graph\n{}".format(chip_name, sample_name), fontsize=14)
    ax.legend(loc='upper left')

    if exo_toggle == False:
        ax.set_xlim3d(-1.125,1.125)
        ax.set_ylim3d(2.25,4.50)
    else:
        ax.set_xlim3d(-1.75,1.75)
        ax.set_ylim3d(0,3.5)
    ax.set_zlim(0,)

    plt.tight_layout()
    plot_name="{}_arrayplot.{}contrast.v{}.png".format(chip_name, cont_str, version)

    plt.savefig('{}/{}'.format(savedir,plot_name),
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    print('File generated: {}'.format(plot_name))
    plt.show()
    plt.clf()
#*********************************************************************************************#
def fluor_overlayer(fluor_img, vis_img, fluor_filename, savedir=''):
    """
    Creates an overlay of the fluorescent images on the visible light images.
    Fluorescent signal appears green, whilst visible light signal is red.
    """

    filesplit = fluor_filename.split('.')
    img_name = ".".join(filesplit[:-1])
    channel = filesplit[-2]

    fluor_norm = fluor_img / np.median(fluor_img) * 2

    p1, p2 = np.percentile(fluor_norm, (0.25, 99.75))
    fluor_rescale = rescale_intensity(fluor_norm, in_range=(p1,p2))

    img_overlay = np.dstack((vis_img, fluor_rescale, np.zeros_like(vis_img)))
    gen_img(img_overlay, name=img_name, savedir=savedir, show=False)
    print("Fluorescent overlay generated for {} channel\n".format(channel))

    return fluor_rescale
#*********************************************************************************************#
def intensity_profile_graph(shape_df, pass_num,zslice_count, img_name):
    # df_len = len(shape_df)

    shape_df=shape_df[shape_df.pass_number == pass_num]
    # for arr in shape_df['mean_intensity_profile_z']:
    intensity_df= pd.DataFrame({'intensity':shape_df.mean_intensity_profile_z.apply(pd.Series).stack()
    })
    intensity_df['max_z'] = [y for x in [[z]*zslice_count for z in shape_df.max_z] for y in x]
    intensity_df.reset_index(inplace=True)
    intensity_df.rename(index=str, columns={'level_1': 'z'}, inplace=True)
    intensity_df['z'] = intensity_df['z'] +1

    sns.set_style('darkgrid')
    sns.lineplot(x='z',
                 y='intensity',
                 hue='max_z',
                 markers=True,palette="ch:2.5,.25", lw=1,
                 ci='sd',
                 data=intensity_df
    )
    plt.title(img_name)
    plt.show()
