import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os,glob
from modules import vpipes, vimage, vquant, vgraph

# csv_list = []
# while (csv_list == []): ##Keep repeating until files are found
#     v2_path = input("\nPlease type in the path to the folder that contains the VIRAGO2 data:\n")
#     if v2_path == 'test':
#         v2_path = '/Volumes/KatahdinHD/ResilioSync/DATA/IRIS/tCHIP_results/v2results/tCHIP004'
#     else:
#         v2_path = v2_path.strip('"')##Point to the correct directory
#     os.chdir(v2_path)
#     csv_list = sorted(glob.glob('*.vcount.csv'))
# chip_name = csv_list[0].split('.')[0]
#
#
# xml_list = []
# for root, dirs, files in os.walk('../..'):
#     for file in files:
#         if file.endswith(".xml"):
#             xml_list.append('{}/{}'.format(root,file))
#
# xml_file = [file for file in xml_list if chip_name in file]
#
# chip_file = vpipes.chip_file_reader(xml_file[0])
#
# mAb_dict, mAb_dict_rev = vpipes.dejargonifier(chip_file)
#
# expt_dict = {'tCHIP004':'EBOVmay@1E6',
#              'tCHIP008':'VSV-EBOVmay@1E6',
#              'tCHIP007':'EBOVmay@1E6',
#              'pCHIP004':'MARVang@1E6',
#              'aaCHIP006':'LASVjos@8E5',
#              'tCHIP014':'EBOVmay@3E5',
#              'mCHIP007':'MARVang@1E6',
#              'mCHIP015':'MARVang@1E6'
#
# }
#
#
# violin_df = pd.DataFrame()
# for filename in csv_list:
#
#     vcount_name = filename.split('/')[-1]
#     expt_name = vcount_name.split('.')[0]
#     spot_num = int(vcount_name.split('.')[1])
#     pass_num = int(vcount_name.split('.')[2])
#
#     vcount_df =pd.read_csv(filename)
#
#     vcount_df['expt_name'] = [expt_dict[expt_name]]*len(vcount_df)
#     vcount_df['spot_name'] = [mAb_dict[spot_num]]*len(vcount_df)
#     vcount_df['pass_num'] = ['final_pass']*len(vcount_df)
#     violin_df = violin_df.append(vcount_df)
#
# violin_df.reset_index(inplace=True, drop=True)
def calc_quartile(filo_score):
    first = filo_score.quantile(.75)
    median = filo_score.quantile(.5)
    third = filo_score.quantile(.25)
    quartile_list = []
    for val in filo_score:
        if val > first:
            quartile_list.append('First')
        elif (val <= first) & (val > median):
            quartile_list.append('Second')
        elif (val <= median) & (val > third):
            quartile_list.append('Third')
        else:
            quartile_list.append('Fourth')

    return quartile_list

def filohisto(shape_df, filo_cutoff = 0.25, irreg_cutoff = -0.25, range = (0,10), show = False):
    data = shape_df[shape_df.pass_number > 1].copy()

    data['quartile'] = calc_quartile(data['filo_score'])
    data.sort_values(by=['filo_score'], ascending=False, inplace=True)
    # title= csv.split('/')[-1]
    fig, (ax_viol, ax_hist) = plt.subplots(1,2, figsize=(12, 8))
    sns.set_context(context='talk', font_scale=1)

    sns.violinplot(x = data.pass_number, y= data.filo_score, inner=None, cut=0, bw= 0.15,
                    color = 'black', ax=ax_viol)
    sns.swarmplot(x = data.pass_number, y= data.filo_score, hue = data.quartile,
                  palette = ['m','y','c','c'], orient = 'v', ax = ax_viol)

    sns.distplot(data.perc_contrast[data.quartile == 'First'],
                                    bins=100,color='m',kde=False,norm_hist=False,
                                    label="Filo score > {} (Filamentous)".format(filo_cutoff),
                                    ax = ax_hist,
                                    hist_kws={"range":range,
                                              "histtype":'step',
                                              "lw":2
                                    }
    )
    sns.distplot(data.perc_contrast[data.quartile == 'Second'],
                                    bins=100,color='y',kde=False,norm_hist=False,
                                    label="{} >= Filo score > {} (Irregular)".format(filo_cutoff, irreg_cutoff),
                                    ax = ax_hist,
                                    hist_kws={"range":range,
                                              "histtype":'step',
                                              "lw":2
                                    }
    )
    sns.distplot(data.perc_contrast[(data.quartile == 'Fourth') | (data.quartile == 'Third')],
                                    bins=100,color='c',kde=False,norm_hist=False,
                                    label="Filo score <= {} (Round)".format(irreg_cutoff),
                                    ax = ax_hist,
                                    hist_kws={"range":range,
                                              "histtype":'step',
                                              "lw":2
                                    }
    )
    sns.distplot(data.perc_contrast,bins=100,color='k',kde=False,norm_hist=False,
                                    label="All particles",
                                    ax = ax_hist,
                                    hist_kws={"range":range,
                                              "histtype":'step',
                                              "lw":0.5
                                    }
    ).set(xlim=range)

    ax_hist.set_ylabel("Particle Count")
    ax_hist.set_xlabel("Percent Contrast")
    # ax_hist.legend()
    # plt.title(title)

    plt.tight_layout()
    if show==True:
        plt.show()


def filoviolin(violin_df, v2_path =''):
    vhf_colormap = ('#e41a1c','#377eb8','#4daf4a',
                '#984ea3','#ff7f00','#ffff33',
                '#a65628','#f781bf','gray','black')
    chip_name = v2_path.split('/')[-1]

    filo_cutoff = 0.2
    irreg_cutoff = 0.1
    plt.figure(figsize = (12,9))
    sns.violinplot(x='expt_name', y='filo_score', palette=vhf_colormap,
                   data=violin_df, cut=0, scale='width', inner='quartile',
                   bw=0.2

                    # bins=100,color='m',kde=False,norm_hist=False,
                    # label="Filo score > {} (Filamentous)".format(filo_cutoff),


    )


    plt.ylabel("filo_score")
    plt.ylim(-1,1)
    plt.xlabel("pass_number")
    # plt.legend()
    plt.title('Multi')
    sns.set_context(context='talk', font_scale=1)
    plt.tight_layout()
    plt.show()
    # plt.savefig('{}/filoviolin.svg'
    #             .format(v2_path))
#
# filoviolin(violin_df, v2_path)

def filojoint(df):
    sns.jointplot(violin_df.round_points, violin_df.filo_points, kind='kde', height=7, space=0)


def calc_quartile(filo_score):
    first = filo_score.quantile(.75)
    median = filo_score.quantile(.5)
    third = filo_score.quantile(.25)
    quartile_list = []
    for val in filo_score:
        if val > first:
            quartile_list.append('First')
        elif (val <= first) & (val > median):
            quartile_list.append('Second')
        elif (val <= median) & (val > third):
            quartile_list.append('Third')
        else:
            quartile_list.append('Fourth')

    return quartile_list
