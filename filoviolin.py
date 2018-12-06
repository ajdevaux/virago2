import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os,glob
from modules import vpipes, vimage, vquant, vgraph

csv_list = []
while (csv_list == []): ##Keep repeating until files are found
    v2_path = input("\nPlease type in the path to the folder that contains the VIRAGO2 data:\n")
    if v2_path == 'test':
        v2_path = '/Volumes/KatahdinHD/ResilioSync/DATA/IRIS/tCHIP_results/v2results/tCHIP004'
    else:
        v2_path = v2_path.strip('"')##Point to the correct directory
    os.chdir(v2_path)
    csv_list = sorted(glob.glob('vcounts/*.vcount.csv'))
chip_name = v2_path.split('/')[-1]


xml_list = []
for root, dirs, files in os.walk('../..'):
    for file in files:
        if file.endswith(".xml"):
            xml_list.append('{}/{}'.format(root,file))

xml_file = [file for file in xml_list if chip_name in file]

chip_file = vpipes.chip_file_reader(xml_file[0])

mAb_dict, mAb_dict_rev = vpipes.dejargonifier(chip_file)

expt_dict = {'tCHIP004':'EBOVmay@1E6',
             'tCHIP008':'VSV-EBOVmay@1E6',
             'pCHIP004':'MARVang@1E6',
             'aaCHIP006':'LASVjos@8E5'
}


violin_df = pd.DataFrame()
for filename in csv_list:

    vcount_name = filename.split('/')[-1]
    expt_name = vcount_name.split('.')[0]
    spot_num = int(vcount_name.split('.')[1])
    pass_num = int(vcount_name.split('.')[2])

    vcount_df =pd.read_csv(filename)

    vcount_df['expt_name'] = [expt_dict[expt_name]]*len(vcount_df)
    vcount_df['spot_name'] = [mAb_dict[spot_num]]*len(vcount_df)
    vcount_df['pass_num'] = ['final_pass']*len(vcount_df)
    violin_df = violin_df.append(vcount_df)

violin_df.reset_index(inplace=True, drop=True)



def filoviolin(violin_df, v2_path):
    vhf_colormap = ('#e41a1c','#377eb8','#4daf4a',
                '#984ea3','#ff7f00','#ffff33',
                '#a65628','#f781bf','gray','black')
    chip_name = v2_path.split('/')[-1]

    filo_cutoff = 0.2
    irreg_cutoff = 0.1
    plt.figure(figsize = (12,9))
    sns.swarmplot(x='expt_name', y='filo_score', palette=vhf_colormap,
                  data=violin_df

                                    # bins=100,color='m',kde=False,norm_hist=False,
                                    # label="Filo score > {} (Filamentous)".format(filo_cutoff),


    )


    plt.ylabel("filo_score")
    plt.xlabel("pass_number")
    # plt.legend()
    plt.title('Multi')
    sns.set_context(context='talk', font_scale=1)
    plt.tight_layout()
    # plt.show()
    plt.savefig('{}/filoviolin.png'
                .format(v2_path))

filoviolin(violin_df, '/Volumes/KatahdinHD/ResilioSync/DATA/IRIS/FIGS4PAPER/vcount_files')

def filojoint(df):
    sns.jointplot(violin_df.round_points, violin_df.filo_points, kind='kde', height=7, space=0)
