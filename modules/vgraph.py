from future.builtins import input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Polygon, Patch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgba_array
from skimage import io as skio
from skimage.exposure import rescale_intensity
import seaborn as sns
import warnings
from modules.vimage import _gen_img_fig
import sys

"""
A suite of graphing subroutines for IRIS images
"""
#*********************************************************************************************#
def get_vhf_colormap():
    vhf_colormap = ('#e41a1c',
                    '#377eb8',
                    '#4daf4a',
                    '#984ea3',
                    '#ff7f00',
                    '#cccc00',
                    '#a65628',
                    '#f781bf',
                    'gray',
                    'black',
                    '#a6cee3',
                    '#1f78b4',
                    '#b2df8a',
                    '#33a02c',
                    '#fb9a99',
                    '#e31a1c',
                    '#fdbf6f',
                    '#ff7f00',
                    '#cab2d6',
                    '#6a3d9a',
                    '#ffff99',
                    '#b15928'
    )
    return vhf_colormap
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
def gen_particle_image(pic_to_show, shape_df, spot_coords, pix_per_um, cv_cutoff=1,
                        r2_cutoff=0, show_particles=True, scalebar=0, markers=[], exo_toggle=False):

    nrows, ncols=pic_to_show.shape

    fig, axes = _gen_img_fig(pic_to_show)

    cx,cy,rad=spot_coords
    true_radius=round((rad - 20) / pix_per_um,2)
    ab_spot=plt.Circle((cx, cy), rad, color='#5A81BB', linewidth=5, fill=False, alpha=0.5)
    axes.add_patch(ab_spot)

    if markers != []:
        for coords in markers:
            mark_box=plt.Rectangle((coords[1]-58,coords[0]-78), 114, 154,
                                      fill=False, ec='green', lw=1)
            axes.add_patch(mark_box)

    if show_particles == True:
        try: current_pass = max(shape_df.pass_number)
        except ValueError: current_pass = 0
        curr_pass_df = shape_df[shape_df.pass_number == current_pass]

        patch_settings=dict(fill=False, color='r',linewidth=1, alpha=0.75)
        line_settings=dict(lw=1,color='purple',alpha=0.25)
        text_settings=dict(fontsize='6', alpha = 0.8, horizontalalignment='right')

        for val in curr_pass_df.index.values:
            label,filo_score,bbox,validity,z_int,centroid = curr_pass_df[['label',
                                                                          'filo_score',
                                                                          'bbox',
                                                                          'validity',
                                                                          'z_intensity',
                                                                          'centroid']].loc[val]
            lowleft_x = bbox[0][1]
            lowleft_y = bbox[0][0]

            if validity == True:
                box_w = bbox[2][1] - lowleft_x
                box_h = bbox[2][0] - lowleft_y
                particle_box = plt.Rectangle((lowleft_x-1,lowleft_y-2), box_w, box_h, **patch_settings)
                axes.add_patch(particle_box)
                datastr = '{}: {}'.format(label, round(z_int,1))
                axes.text(y=centroid[0], x=centroid[1], s=datastr, color='c', **text_settings)

            else:
                line1=lines.Line2D([bbox[3][1],bbox[1][1]],[bbox[3][0],bbox[1][0]], **line_settings)
                line2=lines.Line2D([lowleft_x, bbox[2][1]],[lowleft_y, bbox[2][0]], **line_settings)
                axes.add_line(line1)
                axes.add_line(line2)

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
def histogrammer(value_df, spot_counter, metric_window, bin_size=0.1,norm_to_area=True):
    """Returns a DataFrame of histogram data from the particle dictionary.
    """
    metric_1=float(metric_window[1])
    bin_no=int(metric_1 / bin_size)
    area_list = [float(col.split('.')[-1])*1e-3 for col in value_df.columns]

    histogram_df= pd.DataFrame()
    for col in value_df:
        new_col = '_'.join([str(int(x)) for x in col.split('.')[:-1]])
        histogram_df[new_col], hbins=np.histogram(value_df[col].dropna(), bins=bin_no, range=(0,metric_1))

    if norm_to_area == True:
        histogram_df = histogram_df.div(area_list,axis=1)
        # for i, col in enumerate(histogram_df.columns):
        #     histogram_df[col] = round((histogram_df[col] / area_list[i])*1e-3, 5)

    for col in histogram_df:
        if np.all(np.isnan(histogram_df[col])) == True:
            histogram_df.drop(columns=col, inplace=True)

    histogram_df.set_index(hbins[:-1], inplace=True)

    return histogram_df

#*********************************************************************************************#
def sum_histogram(raw_histogram_df, spot_counter):

    sum_histogram_df = pd.DataFrame(index=raw_histogram_df.index)

    for y in range(1, spot_counter+1):
        spot_histogram_df = pd.DataFrame()
        for col in raw_histogram_df:

            if int(col.split('_')[0]) == y:
                spot_histogram_df=pd.concat([spot_histogram_df, raw_histogram_df[col]], axis=1)

        for i in range(1,len(spot_histogram_df.columns)+1):
            new_name = '{}_{}'.format(y,i)
            if i == 1:
                sum_histogram_df[new_name] = spot_histogram_df.pop(new_name)
            else:
                cols_to_sum = [col for col in spot_histogram_df if int(col.split('_')[1]) <= i]

                sum_histogram_df= pd.concat([sum_histogram_df,
                                         spot_histogram_df[cols_to_sum].sum(axis=1).rename(new_name)], axis=1
                )

    return sum_histogram_df
#*********************************************************************************************#
def average_histogram(sum_histogram_df, spot_df, pass_counter, smooth_window=5, all_locs = False):
    avg_histogram_df = pd.DataFrame(index=sum_histogram_df.index)
    spot_type_list = []
    spot_df = spot_df[spot_df.validity==True]
    for val in spot_df.spot_type:
        if val.startswith('LOC') & (all_locs == True):
            val = 'ALL LOCs'
            spot_df.spot_type = 'ALL LOCs'

        if val not in spot_type_list:
            spot_type_list.append(val)

    for x in range(1,pass_counter+1):
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

            new_name = spot_type+'_'+str(x)

            mean_df = sum_histogram_df[cols_to_avg].mean(axis=1).rename(new_name)

            sdm_df = sum_histogram_df[cols_to_avg].std(axis=1).apply(lambda x: x/sqrt_n).rename(new_name + '_sdm')

            avg_histogram_df = pd.concat([avg_histogram_df, mean_df, sdm_df], axis=1)

    for col in avg_histogram_df.columns:
        if col[-1].isdigit():
            smooth_df=avg_histogram_df[col].rolling(window=smooth_window,
                                                    center=True).mean().rename(col+'_rollingmean')

            avg_histogram_df=pd.concat([avg_histogram_df,smooth_df], axis=1)

    return avg_histogram_df
#*********************************************************************************************#
def generate_histogram(avg_histogram_df, pass_counter, chip_name, metric_str, histo_metric, histo_dir):
    """Generates a histogram figure for each pass in the IRIS experiment from a
    DataFrame representing the average data for every spot type"""


    bin_array = np.array(avg_histogram_df.index, dtype='float')

    smooth_histo_df = avg_histogram_df.filter(regex='rollingmean').rename(columns=lambda x: x[:-12])

    sdm_histo_df = avg_histogram_df.filter(regex='sdm').rename(columns=lambda x: x[:-4])

    smooth_max = np.max(np.max(smooth_histo_df))

    sdm_max = np.max(np.max(sdm_histo_df))

    if np.isnan(sdm_max): sdm_max = 0

    histo_max = np.round(smooth_max+sdm_max,2)

    min_cont, max_cont = metric_str.split("-")

    if pass_counter < 10:
        passes_to_show = 1
    else:
        passes_to_show = 2
    pass_counter // 10
    line_settings = dict(alpha=0.75, elinewidth = 0.5)
    vhf_colormap = get_vhf_colormap()



    for i in range(1, pass_counter+1, passes_to_show):
        sns.set_style('darkgrid')
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        # sns.set(style='ticks')

        c = 0
        for j, col in enumerate(smooth_histo_df):
            splitcol = col.split("_")
            if len(splitcol) == 2:
                spot_type, pass_num = splitcol
            else:
                spot_type, pass_num = splitcol[::2]
            pass_num = int(pass_num)
            if pass_num == i:
                ax.errorbar(x=bin_array,
                             y=smooth_histo_df[col],
                             yerr=sdm_histo_df[col],
                             color = vhf_colormap[c],
                             label = None,
                             lw = 0,
                            **line_settings
                )
                ax.step(x=bin_array,
                        y=smooth_histo_df[col],
                        color = vhf_colormap[c],
                        label = spot_type,
                        lw = 1,
                        where= 'mid',
                        alpha=0.75
                )
                c += 1

        ax.axhline(y=0, ls='dotted', c='k', alpha=0.75)
        ax.axvline(x=float(min_cont), ls='dashed', c='k', alpha=0.8)

        plt.legend(loc = 'best', fontsize = 10)

        plt.ylabel("Frequency (kparticles/mm" + r'$^2$'+")", size = 14)
        plt.xlabel("{} (%)".format(histo_metric), size = 14)

        if histo_max < 0.5:
            ysteps = 0.1
        else:
            ysteps = round(histo_max/10,1)

        plt.yticks(np.arange(0, histo_max, ysteps), size = 12)

        xlabels = np.append(bin_array, int(max_cont))[::(len(bin_array) // 10)]
        plt.xticks(xlabels, size = 12, rotation = 90)

        plt.title(chip_name+" Pass "+str(i)+" Average Histograms")

        figname = ('{}_combohisto_pass_{}_{}_{}.png'.format(chip_name,i,histo_metric,metric_str))
        plt.savefig('{}/{}'.format(histo_dir,figname), bbox_inches = 'tight', dpi = 300)
        print("File generated: {}".format(figname))
        plt.clf()
#*********************************************************************************************#
def defocus_profile_graph(shape_df, pass_num, zslice_count, dir, exo_toggle, img_name=''):
    inflection_pt = (zslice_count // 2)
    pass_df = shape_df[shape_df.pass_number == pass_num]

    def_df = pass_df.max_z_stack.apply(pd.Series)
    norm_def_df = def_df.sub(def_df[inflection_pt], axis=0)

    defocus_df= pd.DataFrame({'defocus': def_df.stack(),
                              'norm_defocus': norm_def_df.stack()

    })
    defocus_df['pc'] = [y for x in [[pc]*zslice_count
                          for pc in pass_df.perc_contrast]
                          for y in x
    ]

    defocus_df.reset_index(inplace=True)

    defocus_df.rename(index=str, columns={'level_0':'label',
                                            'level_1':'z'}, inplace=True)
    defocus_df['z'] = defocus_df['z'] + 1
    if max(defocus_df.z) <= 10:
        defocus_df['z'] = defocus_df['z'].apply(lambda x: x if x <= 7 else int(x + x - 7))

    if exo_toggle == True:
        plot0, plot1, plot2 = (5,80,5)
        defocus_df = defocus_df[(defocus_df.pc >= plot0) & (defocus_df.pc <= plot1)]
        defocus_df['pc_bin'] = round(defocus_df.pc*0.02,1)*50
    else:
        plot0, plot1, plot2 = (0.5,10, 0.1)
        defocus_df = defocus_df[(defocus_df.pc >= plot0) & (defocus_df.pc <= plot1)]
        defocus_df['pc_bin'] = round(defocus_df.pc,1)

    defocus_df.sort_values(by=['pc'], ascending=False, inplace=True)
    if len(defocus_df) > 0:
        sns.set_style('darkgrid')
        sns.lineplot(x='z',
                     y='norm_defocus',
                     hue='pc_bin',
                     markers=True, palette="Blues_r", lw=1,
                     ci='sd',
                     data=defocus_df
        )
        plt.title(img_name)
        plt.xticks(np.arange(1, zslice_count+1, 1))
        # plt.legend(np.arange(plot0,plot1,plot2),loc='lower left', ncol=2)
        # plt.ylim(min(defocus_df.norm_intensity),max(defocus_df.norm_intensity))
        # plt.show()
        plt.savefig('{}/{}.png'.format(dir, img_name))
        plt.clf()
    else:
        print("No valid particles to graph the profile of\n")
#*********************************************************************************************#
def average_spot_data(spot_df, pass_counter):
    """Creates a dataframe containing the average data for each antibody spot type"""
    averaged_df=pd.DataFrame()
    spot_list=[]
    for val in spot_df.spot_type:
        if val not in spot_list:
            spot_list.append(val)

    for i, spot in enumerate(spot_list):
        sub_df=spot_df[(spot_df.spot_type == spot_list[i]) & (spot_df.validity == True)]
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
def generate_timeseries(spot_df, averaged_df, metric_window, mAb_dict,
                        chip_name, sample_name, version,
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

    vhf_colormap = get_vhf_colormap()

    for key in mAb_dict.keys():
        print(key)
        if key == 1:
            c = 0
        elif (mAb_dict[key-1][0] != mAb_dict[key][0]):
            c += 1

        solo_spot_df=spot_df[(spot_df.spot_number == key)
                                & (spot_df.validity == True)].reset_index(drop=True)
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

    ax2.legend(loc='upper left', fontsize=16, ncol=1)
    if max(spot_df.scan_number) < 10: x_grid=1
    else: x_grid=max(spot_df.scan_number) // 10

    if scan_or_time == 'scan':
        plt.xlabel("Scan Number", size=24)
        plt.xticks(np.arange(1, max(spot_df.scan_number) + 1, x_grid), size=24, rotation=30)

    elif scan_or_time == 'time':
        plt.xlabel("Time (min)", size=24)
        plt.xticks(np.arange(0, max(spot_df.scan_time) + 1, 5), size=24, rotation=30)
    metric_str='{0}-{1}'.format(*metric_window)
    plt.ylabel("Particle Density (kparticles/mm" + r'$^2$'+")\n {} % Contrast".format(metric_str), size=24)
    plt.yticks(color='k', size=24)
    plt.title("{} Time Series of {} - v{}".format(chip_name, sample_name, version), size=28)
    plt.axhline(linestyle='--', color='gray')

    plot_name="{}_timeseries.{}contrast.v{}.png".format(chip_name, metric_str,version)

    plt.savefig('{}/{}'.format(savedir, plot_name),
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    print('File generated: {}'.format(plot_name))
    plt.clf(); plt.close('all')
#*********************************************************************************************#
# def old_gen_barplot(spot_df, pass_counter, metric_window,  chip_name, sample_name, version,
#                      savedir='', plot_3sigma=False, neg_ctrl_str=''):
#     """
#     Generates a barplot for the dataset.
#     Most useful for before and after scans (pass_counter == 2)
#     """
#     firstlast_spot_df=spot_df[  ((spot_df.scan_number == 1)
#                               | (spot_df.scan_number == pass_counter))
#                               & (spot_df.validity == True)
#     ]
#     metric_str='{0}-{1}'.format(*metric_window)
#     final_spot_df=spot_df[  (spot_df.scan_number == pass_counter)
#                           & (spot_df.validity == True)
#     ]
#
#     fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(8, 6), sharey=True)
#     sns.set_style('darkgrid')
#     vhf_colormap = get_vhf_colormap()
#
#     sns.barplot(y='kparticle_density',x='spot_type',hue='scan_number',data=firstlast_spot_df,
#                  palette = 'binary', errwidth=2, ci='sd', ax=ax1
#     )
#     ax1.set_ylabel("Particle Density (kparticles/mm" + r'$^2$'+")\n"+"Contrast="+metric_str+ '%', fontsize=14)
#     ax1.set_xlabel("Prescan & Postscan", fontsize=14)
#
#     sns.barplot(y='normalized_density',x='spot_type',data=final_spot_df, palette = vhf_colormap,
#                 errwidth=2, ci='sd', ax=ax2
#     )
#     ax2.set_ylabel("")
#     plt.yticks(fontsize=12)
#     ax2.set_xlabel("Difference", fontsize=14)
#     for ax in fig.axes:
#         plt.sca(ax)
#         plt.xticks(rotation=30, fontsize=12)
#     if plot_3sigma == True:
#         neg_control_vals = final_spot_df.normalized_density[final_spot_df.spot_type.str.contains(neg_ctrl_str)]
#         neg_control_mean = np.mean(neg_control_vals)
#         neg_control_std = np.std(neg_control_vals)
#         three_sigma = (neg_control_std * 3) + neg_control_mean
#         ax2.axhline(y=three_sigma,ls='--',lw=2,color='r', label='3'+r'$\sigma$'+' Signal Threshold')
#     plt.legend(loc='upper right', fontsize=14, ncol=1)
#
#     plt.suptitle("{} {} - v{}".format(chip_name,sample_name,version), y=1.04, fontsize=16)
#
#     plt.tight_layout()
#     plot_name="{}_barplot.{}contrast.v{}.png".format(chip_name, metric_str, version)
#
#     plt.savefig('{}/{}'.format(savedir,plot_name),
#                 bbox_inches='tight', pad_inches=0.1, dpi=300)
#     print('File generated: {}'.format(plot_name))
#     plt.close('all')
#*********************************************************************************************#
def iris_barplot_gen(spot_df, pass_counter, metric_window, chip_name, version,
                     savedir='', plot_3sigma=False, neg_ctrl_str='8G5|MOUSE IGG|muIgG|GFP'):
    """
    Generates a barplot for the dataset.
    Most useful for before and after scans (pass_counter == 2)
    """
    vhf_colormap = get_vhf_colormap()
    metric_str='{0}-{1}'.format(*metric_window)
    sns.set_style('darkgrid')
    labels = [Patch(color=vhf_colormap[c], label=val.split('_')[0])
              for c, val in enumerate(spot_df.spot_type.unique())
    ]

    first_df = spot_df[(spot_df.scan_number == 1)  & (spot_df.validity == True)]
    ax1_max = int(round(max(first_df['kparticle_density']),0)) + 1
    if ax1_max < 10: ax1_max = 10

    last_df = spot_df[(spot_df.scan_number == pass_counter) & (spot_df.validity == True) ]
    ax2_max = int(round(max(last_df['normalized_density']),0)) + 1
    if ax2_max < 10: ax2_max = 10

    fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(8, 6), sharey=True,
                                 gridspec_kw = {'width_ratios':[ax1_max, ax2_max]}
    )
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.suptitle("{}".format(chip_name), y=1, fontsize=20)
    plt.xlabel("Particle Density (kparticles/mm" + r'$^2$'+")\n"+"Contrast="+metric_str+ '%', fontsize=14)


    ax1 = sns.barplot(x='kparticle_density',y='spot_type',data=first_df,
                      palette = vhf_colormap, errwidth=2, ci='sd', ax=ax1, alpha=0.5
    )
    ax1.set_xlim([ax1_max,0])
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.set_title("Prescan", fontsize=14)
    ax1.set_ylabel('')
    ax1.set_xlabel('')
    ax1.set_yticklabels('')

    ax2 = sns.barplot(x='normalized_density',y='spot_type',data=last_df,
                      palette = vhf_colormap, errwidth=2, ci='sd', ax=ax2
    )
    ax2.set_xlim([0,ax2_max])
    ax2.xaxis.set_tick_params(labelsize=14)
    ax2.set_title("Postscan", fontsize=14)
    ax2.set_ylabel('')
    ax2.set_xlabel('')
    ax2.set_yticklabels('')

    if plot_3sigma == True:
        neg_control_vals = last_df.normalized_density[last_df.spot_type.str.contains(neg_ctrl_str)]
        neg_control_mean = np.mean(neg_control_vals)
        neg_control_std = np.std(neg_control_vals)
        three_sigma = (neg_control_std * 3) + neg_control_mean
        ax2.axvline(x=three_sigma,ls=':',lw=2,color='k', label='3'+r'$\sigma$'+' Signal Threshold')
        line_legend = ax2.get_legend_handles_labels()
        labels = labels+line_legend[0]

    if ax1_max >= ax2_max:
        ax1.legend(handles=labels, fontsize=10, loc ='best')
    else:
        ax2.legend(handles=labels, fontsize=10, loc ='best')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()

    plot_name="{}_barplot.{}contrast.v{}.png".format(chip_name, metric_str, version)

    plt.savefig('{}/{}'.format(savedir, plot_name), bbox_inches='tight', pad_inches=0.1, dpi=300)
    print('File generated: {}'.format(plot_name))
    plt.close('all')
#*********************************************************************************************#
def filo_image_gen(shape_df, pic1, pic2, pic3,
                  ridge_list, sphere_list, other_list,
                  cv_cutoff=0.1, r2_cutoff=0.85, show=True):

    fig=plt.figure(figsize=(24, 8))

    ax1=fig.add_subplot(1, 3, 1)
    ax1.imshow(pic1, cmap=plt.cm.magma)
    ax1.axis('off')
    ax1.set_title('Contrast-Adjusted Image', fontsize=18)
    patch_settings=dict(fill=False, linewidth=1, alpha=0.75)
    line_settings=dict(lw=1,color='purple',alpha=0.6)
    scatter_settings=dict(s=12, linewidths=0)

    filo_df = shape_df[pd.notna(shape_df.vertices)]
    if not filo_df.empty:
        v1,v2 = map(np.array, zip(*filo_df.vertices))
        ax1.scatter(v1[:,1], v1[:,0], color='y', marker='v')
        ax1.scatter(v2[:,1], v2[:,0], color='y', marker='+')

    for t in filo_df.index.values:
        ax1.text(y=filo_df.centroid[t][0], x=filo_df.centroid[t][1], s=str(round(filo_df.filo_lengths[t],2)),
                 color='y', fontsize='9', horizontalalignment='right')

    for i in shape_df.index:
        bbox,centroid,z_int,validity = shape_df[['bbox','centroid','z_intensity','validity']].loc[i]

        lowleft_x = bbox[0][1]
        lowleft_y = bbox[0][0]

        if validity == True:
            color = 'c'
            alpha = 1
        else:
            color = 'purple'
            alpha = 0.5

        ax1.text(y=centroid[0], x=centroid[1], s=str(round(z_int,2)),
                 color=color, alpha=alpha,fontsize='9', horizontalalignment='left')

        particle_box = plt.Rectangle((lowleft_x-1,lowleft_y-2),
                                      bbox[2][1] - lowleft_x,
                                      bbox[2][0] - lowleft_y,
                                      fill=False, color=color, alpha=1)
        ax1.add_patch(particle_box)

    ax2=fig.add_subplot(1, 3, 2, sharex=ax1, sharey=ax1)

    ax2.imshow(pic2, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('Shape Index', fontsize=18)


    ax3=fig.add_subplot(1, 3, 3, sharex=ax1, sharey=ax1)
    ax3.imshow(pic3, plt.cm.gray)
    ax3.axis('off')
    ax3.set_title('Scored Binary Image', fontsize=18)

    if not ridge_list == []:
        ridge_y,ridge_x = zip(*ridge_list)
        ax3.scatter(ridge_x,ridge_y, color='magenta', **scatter_settings, marker='^')
    if not sphere_list == []:
        sphere_y,sphere_x = zip(*sphere_list)
        ax3.scatter(sphere_x, sphere_y, color='cyan', **scatter_settings, marker='o')
    if not other_list == []:
        other_list_y_s,other_list_x_s = zip(*other_list)
        ax3.scatter(other_list_x_s,other_list_y_s, color='gray', **scatter_settings, marker='X')

    fig.tight_layout()
    if show == True:
        plt.show()
    plt.clf()
    # sys.exit()
#*********************************************************************************************#
def _bugfix_bar3d(bar3d):
    """bug fixes for bar3d plots"""
    bar3d._facecolors3d=to_rgba_array(bar3d._facecolors3d, bar3d._alpha)
    bar3d._edgecolors3d=to_rgba_array(bar3d._edgecolors3d, bar3d._alpha)
    bar3d._facecolors2d=bar3d._facecolors3d
    bar3d._edgecolors2d=bar3d._edgecolors3d

    return bar3d
#*********************************************************************************************#
def chipArray_graph(spot_df, chip_name='IRIS chip',
                    sample_name ='',exo_toggle=False, metric_str='', version='',
                    savedir='/Users/dejavu/Desktop'
                    ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    prescan = min(spot_df.scan_number)
    last_scan = max(spot_df.scan_number)
    spot_array = np.array(spot_df.spot_number[::last_scan])
    total_spots = max(spot_array)
    vhf_colormap = get_vhf_colormap()

    linx,liny= zip(*(spot_df.chip_coords_xy[::last_scan]))
    linx = np.array(linx,dtype='float') / 1000
    liny = np.array(liny,dtype='float') / 1000
    ax.plot(linx,liny,0, color='k', linewidth=1, linestyle='--', alpha=0.5)
    for j,spot_num in enumerate(spot_array):
        ax.text(linx[j],liny[j],0, s=str(spot_num),
                fontsize=8, fontname='monospace',color='k', alpha=1,
                horizontalalignment='left', verticalalignment='top',
        )
    spot_type_list = []
    for val in spot_df.spot_type:
        if val not in spot_type_list:
            spot_type_list.append(val)

    zpos = 0
    for c, spot in enumerate(spot_type_list):
        prescan_df = spot_df[(spot_df.scan_number==prescan)&(spot_df.spot_type==spot)].reset_index()

        xpos,ypos = zip(*(prescan_df.chip_coords_xy))
        dx = dy = 0.08
        offset = dx * 0.5

        xpos = np.array(xpos,dtype='float') * 0.001
        xpos_bar = xpos - offset

        ypos = np.array(ypos,dtype='float') * 0.001
        ypos_bar = ypos - offset

        rad_list = np.sqrt(np.array(prescan_df.area_sqmm,dtype='float')/np.pi)
        color = vhf_colormap[c]
        for i,r in enumerate(rad_list):
            Ab_spot = plt.Circle((xpos[i],ypos[i]),r,facecolor=color, edgecolor ='k',alpha=0.75)
            ax.add_patch(Ab_spot)
            art3d.pathpatch_2d_to_3d(Ab_spot, z=0, zdir="z")

        dz_pre = np.array(prescan_df.kparticle_density)
        prebar = ax.bar3d(xpos_bar, ypos_bar, zpos, dx, dy, dz_pre,
                         color=color, edgecolor='k', label=None, alpha=0.15)

        prebar = _bugfix_bar3d(prebar)

        if last_scan > 1:
            scan_df = spot_df[(spot_df.scan_number==last_scan)&(spot_df.spot_type==spot)].reset_index()
            dz = np.array(scan_df.normalized_density)
            dx = dy = 0.04
            offset = dx * 0.5
            xpos_bar = xpos - offset
            ypos_bar = ypos - offset

            bar = ax.bar3d(xpos_bar, ypos_bar, zpos, dx, dy, dz,
                           color=color, edgecolor='k', label=spot, alpha=0.75)

            bar = _bugfix_bar3d(bar)

    tri = Polygon(np.array([[-.25,-.5],[0,0],[.25,-.5]]), color='k')
    ax.add_patch(tri)
    art3d.pathpatch_2d_to_3d(tri, z=0, zdir="z")


    ax.set_xlabel("Lat. Axis (mm)",fontsize=8)
    ax.set_ylabel("Long. Axis (mm)", fontsize=8)

    ax.set_zlabel("Particle Density (kparticles/mm" + r'$^2$'+")\n{}% Contrast".format(metric_str))
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
    plot_name="{}_arrayplot.{}contrast.v{}.png".format(chip_name, metric_str, version)

    plt.savefig('{}/{}'.format(savedir,plot_name),
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    print('File generated: {}'.format(plot_name))
    plt.show()
    plt.clf()
#*********************************************************************************************#
def fluor_overlayer(fluor_df, g_img, r_img=np.array([]),b_img=np.array([]), show_particles=False):
    """
    Creates an overlay of the fluorescent images on the visible light images.
    Fluorescent signal appears green, whilst visible light signal is red.
    """
    if r_img.size == 0:
        r_img = np.zeros_like(g_img)
    if b_img.size == 0:
        b_img = np.zeros_like(g_img)

    img_overlay = np.dstack((r_img, g_img, b_img))

    fig, axes = _gen_img_fig(img_overlay)

    if show_particles == True:
        for val in fluor_df.index.values:
            bbox,norm_int,centroid,chan = fluor_df[[ 'bbox','fl_intensity','centroid','channel']].loc[val]

            lowleft_x = bbox[0][1]
            lowleft_y = bbox[0][0]

            box_w = bbox[2][1] - lowleft_x
            box_h = bbox[2][0] - lowleft_y
            if chan == 'A':
                box_color = 'y'
                align = 'left'
            else:
                box_color = 'm'
                align = 'right'
            particle_box = plt.Rectangle((lowleft_x-1,lowleft_y-2), box_w, box_h,
                                          fill=False, color=box_color, alpha=0.75)
            axes.add_patch(particle_box)

            axes.text(y=centroid[0], x=centroid[1], s=str(round(norm_int,1)),
                     color='c', fontsize='6', alpha = 0.8, horizontalalignment=align)


#*********************************************************************************************#
