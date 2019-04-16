from __future__ import division
from future.builtins import input
from lxml import etree
import pandas as pd
import numpy as np
import seaborn as sns
from skimage import io as skio
from skimage.external.tifffile import TiffWriter
import os, json, math, warnings, sys, glob, zipfile, re
#*********************************************************************************************#
#
#           SUBROUTINES
#
#*********************************************************************************************#
def three_digs(number):
    if not type(number) is str:
        number = str(number)

    return '0'*(3 - len(number)) + number
#*********************************************************************************************#
def xml_parser(xml_file):
    """XML file parser, reads the turns the chipFile into a list of dictionaries"""
    xml_raw = etree.iterparse(xml_file)
    spot_info_dict = {}
    chipFile = []
    for action, elem in xml_raw:
        if not elem.text:
            text = "None"
        else:
            text = elem.text
        spot_info_dict[elem.tag] = text
        if elem.tag == "spot":
            chipFile.append(spot_info_dict)
            spot_info_dict = {}
    print("Chip file read\n")
    return chipFile
#*********************************************************************************************#
def chipFile_reader(chipFile, remove_jargon = True):
    """This takes antibody names from the chip file and makes them more general for easier layperson understanding.
    It returns two dictionaries that match spot number with antibody name."""
    jargon_dict = {
                   '13F6': r'$\alpha$'+'-EBOV', '127-8': r'$\alpha$'+'-MARV', 'AGP127-8':r'$\alpha$'+'-MARV',
                   '6D8': r'$\alpha$'+'-EBOV', '8.9F': r'$\alpha$'+'-LASV',
                   '8G5': r'$\alpha$'+'-VSV', '4F3': r'$\alpha$'+'-EBOV',
                   '13C6': r'$\alpha$'+'-EBOV', '40-3': r'$\alpha$'+'-IAV-H3Nx',
                   'PA1-7221': r'$\alpha$'+'-H1N1',  'PA1-7222': r'$\alpha$'+'-H3N2',
                   'MOUSE': 'muIgG', 'CD81': r'$\alpha$'+'-CD81',
                   'CD63': r'$\alpha$'+'-CD63', 'CD9': r'$\alpha$'+'-CD9',
                   'CD41A': r'$\alpha$'+'-CD41A', 'GFP': r'$\alpha$'+'-GFP',
                   'SAV': r'$\alpha$'+'-SAV'
                   }
    mAb_dict = {}
    for q, spot in enumerate(chipFile):
        spot_info_dict = chipFile[q]
        mAb_name = spot_info_dict['spottype'].upper()
        xloc = spot_info_dict['xmicrons']
        yloc = spot_info_dict['ymicrons']
        spot_height = spot_info_dict['signalheight']

        for key in jargon_dict.keys():
            if (mAb_name.startswith(key) or mAb_name.endswith(key)) & (remove_jargon == True):
                new_name = jargon_dict[key] + '_(' + mAb_name + ')'
                break
            else:
                new_name = mAb_name
        mAb_dict[q + 1] = new_name, xloc, yloc, spot_height

    mAb_dict_rev = {val[0]: key for key, val in mAb_dict.items()}

    return mAb_dict, mAb_dict_rev
#*********************************************************************************************#
def sample_namer(iris_path):
    if sys.platform == 'win32':
        folder_name = iris_path.split("\\")[-1]
    elif sys.platform == 'darwin':
        folder_name = iris_path.split("/")[-1]
    else: folder_name = ''

    if len(folder_name.split("_")) == 2:
        sample_name = folder_name.split("_")[-1]
    else:
        sample_name = input("\nPlease enter a sample descriptor (e.g. VSV-MARV@1E6 PFU/mL)\n")
    return sample_name
#*********************************************************************************************#
# def write_vdata(dir, filename, list_of_vals):
#     with open(dir + '/' + filename + '.vdata.txt', 'w') as vdata_file:
#         vdata_file.write((
#                          'filename: {0}\n'
#                          +'spot_type: {1}\n'
#                          +'area_sqmm: {2}\n'
#                          +'image_shift_RC: {3}\n'
#                          +'overlay_mode: {4}\n'
#                          +'non-filo_ct: {5}\n'
#                          +'filo_ct: {6}\n'
#                          +'total_particles: {7}\n'
#                          +'slice_high_count: {8}\n'
#                          +'spot_coords_xyr: {9}\n'
#                          +'marker_coords_RC: {10}\n'
#                          +'binary_thresh: {11}\n'
#                          +'valid: {12}'
#                          ).format(*list_of_vals)
#                         )
        # write_list =[]
        # for val in list_of_vals:
        #     write_list.append(str(val)+': '+val+'\n')

#*********************************************************************************************#
# def missing_pgm_fixer(spot_to_scan, pass_counter, pass_per_spot_list,
#                       chip_name, marker_dict, filo_toggle = False, version = 1):
#     print("Missing pgm files... fixing...")
#     vcount_dir = '../virago_output/{}/vcounts'.format(chip_name)
#     scans_counted = [int(file.split(".")[-1]) for file in pass_per_spot_list]
#     scan_set = set(range(1,pass_counter+1))
#     missing_df = pd.DataFrame(np.zeros(shape = (1,6)),
#                          columns = ['y', 'x', 'r', 'z', 'pc', 'sdm'])
#
#     missing_csvs = scan_set.difference(scans_counted)
#
#     for scan in missing_csvs:
#         scan_str = str(scan)
#         spot_str = str(spot_to_scan)
#         spot_scan_str  = '{}.{}'.format(spot_str, scan_str)
#         marker_dict[spot_scan_str] = (0,0)
#         scan_data = [chip_name, vpipes.three_digs(spot_to_scan), vpipes.three_digs(scan)]
#         missing_scan = "{0}.{1}.{2}".format(*scan_data)#chip_name + '.' + '0' * (3 - len(spot_str)) + spot_str + '.' + '0' * (3 - len(scan_str)) + scan_str
#         print(missing_scan)
#         missing_df.to_csv('{}/{}.vcount.csv'.format(vcount_dir, missing_scan))
#         if filo_toggle == True:
#             filo_dir = '../virago_output/'+ chip_name + '/filo'
#             missing_filo_df = pd.DataFrame(columns = ['centroid_bin', 'label_skel',
#                                                       'filament_length_um', 'roundness',
#                                                       'pc', 'vertex1', 'vertex2',
#                                                       'area', 'bbox'])
#             missing_filo_df.to_csv('{}/{}.filocount.csv'.format(filo_dir,missing_scan))
#         missing_vals = list([missing_scan, 'N/A', 0, 'N/A', 'N/A', 'N/A',
#                             'N/A', 0, 'N/A', 'N/A', 'N/A', 'N/A', False])
#         write_vdata(vcount_dir, missing_scan, missing_vals)
#
#         print("Writing blank data files for {}".format(missing_scan))


#*********************************************************************************************#
def mirror_finder(pgm_list):
    regex = re.compile('000\.000')
    mirror_list = list(filter(regex.search, pgm_list))
    if mirror_list != []:
        mirror_file = mirror_list[0]
        pgm_list.remove(mirror_file)
        mirror = skio.imread(mirror_file)
        print("Mirror file detected\n")
        mirror_toggle = True
    else:
        print("Mirror file absent\n")
        mirror_toggle = False
        mirror = np.ones(shape = 1, dtype = int)
    return pgm_list, mirror
#*********************************************************************************************#
def sample_namer(iris_path):
    if sys.platform == 'win32': folder_name = iris_path.split("\\")[-1]
    elif sys.platform == 'darwin': folder_name = iris_path.split("/")[-1]
    else: folder_name = ''
    if len(folder_name.split("_")) == 2:
        sample_name = folder_name.split("_")[-1]
    else:
        sample_name = input("\nPlease enter a sample descriptor (e.g. VSV-MARV@1E6 PFU/mL)\n")
    return sample_name
#*********************************************************************************************#
def zipper(filename, filelist, compression='zip', iris_path=os.getcwd()):
    if compression == '7z':
        zMODE = zipfile.ZIP_LZMA
    elif compression =='zip':
        zMODE = zipfile.ZIP_DEFLATED
    elif compression == 'bz2':
        zMODE = zipfile.ZIP_BZIP2

    if not os.path.exists('archive'):
        os.makedirs('archive')

    zf = zipfile.ZipFile(iris_path +'/archive/'+filename+'.'+compression, mode='w')
    for file in filelist:
        zf.write(file,compress_type=zMODE)
        print("{} added to {}.{}".format(file, filename, compression))
#*********************************************************************************************#
def iris_txt_reader(iris_txt, mAb_dict, pass_counter):
    spot_df = pd.DataFrame()
    for ix, txtfile in enumerate(iris_txt):
        spot_ix = ix+1
        dict_vals = mAb_dict[spot_ix]
        spot_data_solo = pd.DataFrame({'spot_number'   : [spot_ix] * pass_counter,
                                       'scan_number'   : range(1,pass_counter + 1),
                                       'spot_type'     : [dict_vals[0]] * pass_counter,
                                       'chip_coords_xy': [dict_vals[1:3]] * pass_counter
                                       }
        )
        if type(txtfile) is str:
            txtdata = pd.read_table(txtfile, sep = ':', error_bad_lines = False,
                                header = None, index_col = 0, usecols = [0, 1])
            expt_date = txtdata.loc['experiment_start'][1].split(" ")[0]


            pass_labels = [row for row in txtdata.index if row.startswith('pass_time')]
            times_s = txtdata.loc[pass_labels].values.flatten().astype('float')
            pass_diff = pass_counter - len(pass_labels)
            if pass_diff > 0:
                times_s = np.append(times_s, [0] * pass_diff)
            spot_data_solo['scan_time'] = np.round(times_s * 0.0167,2)
            print('File scanned:  {}'.format(txtfile))

        else:
            print("Missing text file for spot {}\n".format(txtfile))
            spot_data_solo['scan_time'] = [0] * pass_counter
            expt_date = None
        spot_df = spot_df.append(spot_data_solo, ignore_index = True)

    return spot_df, expt_date
#*********************************************************************************************#
def version_finder(version):
    major, minor, micro = re.search('(\d+)\.(\d+)\.(\d+)', version).groups()

    return int(major), int(minor), int(micro)
#*********************************************************************************************#
def pgm_to_tiff(pic3D, img_name, stack_list, tiff_compression = 1, archive_pgm=False):
    # stack_name = '.'.join(pgm_name[:-2])
    tiff_name = img_name + '.tif'

    with TiffWriter(tiff_name, imagej=True) as tif_img:
        for i in range(pic3D.shape[0]):
            tif_img.save(pic3D[i], compress = tiff_compression)
        print("TIFF file generated: {}".format(tiff_name))

    if archive_pgm == True:
        zipper(img_name, stack_list, compression='bz2', iris_path=os.getcwd())
    for pgm in stack_list:
        os.remove(pgm)
#*********************************************************************************************#
def tiff_maker(pgm_list, tiff_compression = 1, archive = True):
    chip_name = pgm_list[0].split(".")[0]

    pgm_list, mirror = mirror_finder(pgm_list)
    fluor_files = [file for file in pgm_list if file.split(".")[-2] in 'ABC']
    if fluor_files:
        pgm_list = [file for file in pgm_list if file not in fluor_files]
        print("Fluorescent channel(s) detected, will not be converted to TIFF\n")



    pgm_set = set([".".join(file.split(".")[:3]) for file in pgm_list])

    spot_counter = max([int(pgmfile.split(".")[1]) for pgmfile in pgm_list])##Important
    zslice_count = max([int(pgmfile.split(".")[3]) for pgmfile in pgm_list])

    print("There are {} PGM images, in stacks of {}.\n".format(len(pgm_list), zslice_count))

    pass_counter = int(max([pgm.split(".")[2] for pgm in pgm_list]))##Important

    spot_to_scan = 1

    while spot_to_scan <= spot_counter:
        print(spot_to_scan)

        pps_list = sorted([file for file in pgm_set if int(file.split(".")[1]) == spot_to_scan])
        passes_per_spot = len(pps_list)

        if passes_per_spot == 0:
            print("No pgm files for spot {} \n".format(spot_to_scan))

        else:
            for scan in range(0,passes_per_spot,1):

                stack_list = [file for file in pgm_list if file.startswith(pps_list[scan])]
                pgm_name = ".".join(stack_list[0].split(".")[:3])

                pic3D = np.array([pic for pic in skio.imread_collection(stack_list)],
                                 dtype='uint16')

                pgm_to_tiff(pic3D, pgm_name, stack_list,
                            tiff_compression = tiff_compression, archive_pgm=archive)

        spot_to_scan += 1

    return sorted(glob.glob('{}.*.tif'.format(chip_name)))
