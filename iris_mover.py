import os, glob, sys, re
from modules.vpipes import tiff_maker


def fluor_cleaner(pgm_list, splitter ='/', fluor_clean_toggle = 'no'):
    bad_fluor_files = [file for file in pgm_list if file.split(".")[-2] in ['Aold', 'Cold']]
    if bad_fluor_files:
        print("\nFluorescent channel(s) detected: {}\n.".format(bad_fluor_files))
        fluor_clean_toggle = input("Remove fluorescent files? (y/[n])")
        if fluor_clean_toggle in ('yes', 'y'):
            for file in bad_fluor_files:
                os.remove(file)
                pgm_list.remove(file)
                print("Deleted {}\n".format(file))

    old_fluor_files = list(filter(re.compile(".*old").match, pgm_list))
    if old_fluor_files:
        olddir = 'old'
        if not os.path.exists(olddir): os.mkdir(olddir)
        print("Moving old fluorescent files")
        for file in old_fluor_files:
            dest = '{}{}{}'.format(olddir, splitter, file)
            os.rename(file, dest)
            pgm_list.remove(file)

    return pgm_list
#***********************************************************************************************#
def file_deleter(file_list, delete_toggle = 'no'):
    print(file_list)
    delete_toggle = input("Remove files? (y/[n])")
    if delete_toggle in ('yes','y'):
        delete_list = file_list.copy()

        for file in delete_list:
            os.remove(file)
            print("Deleted {}\n".format(file))
#***********************************************************************************************#
def txt_combiner(txt_list, splitter):
    # if sys.platform == 'win32': splitter = ('\\')
    # elif sys.platform == 'darwin': splitter = ('/')

    new_txts = ['..' +splitter + file for file in txt_list if len(file.split('.')) == 3]
    old_txts = [file for file in sorted(glob.glob('*.txt')) if len(file.split('.')) == 3]

    for newfile in new_txts:
        for oldfile in old_txts:
            if newfile.split(splitter)[1] == oldfile:
                print(newfile, '\n', oldfile)
                of_open = open(oldfile)
                oldlines = of_open.read()
                of_open.close()

                oldlines_list = oldlines.split("\n")
                oldlines_list.remove('')
                for line in oldlines_list:
                    if line.startswith('experiment'):
                        print(line)
                        expt_start1 = oldlines_list.pop(oldlines_list.index(line))
                        expt_start1 = expt_start1.split(' ')
                        start1_date = expt_start1[0].split(':')[1]
                        start1_timecode = expt_start1[1]
                        start1_hms = start1_timecode.split(':')
                        start1_sec = (int(start1_hms[0])*(60**2) + int(start1_hms[1])*60 + int(start1_hms[2]))
                    # else:
                    #     print('Cannot find experiment start time')

                old_dict = {k:v for k,v in (val.split(':') for val in oldlines_list)}

                last_pass = int([key[-1] for key in old_dict.keys() if 'pass_time' in key][-1])
                combo_dict = old_dict.copy()
    #-----------------------------------------------------------------------------------------------#
                nf_open = open(newfile)
                newlines = nf_open.read()
                nf_open.close()

                newlines_list = newlines.split("\n")
                newlines_list.remove('')
                for line in newlines_list:
                    if line.startswith('experiment'):
                        print(line)
                        expt_start2 = newlines_list.pop(newlines_list.index(line))
                        expt_start2 = expt_start2.split(' ')
                        start2_timecode = expt_start2[1]
                        start2_hms = start2_timecode.split(':')
                        start2_sec = (int(start2_hms[0])*(60**2) + int(start2_hms[1])*60 + int(start2_hms[2]))
                    # else:
                    #     print('Cannot find experiment start time')

                start_diff = start2_sec - start1_sec

                new_dict = {k:v for k,v in (val.split(':') for val in newlines_list)}

                pass_time_new = float(new_dict['pass_time001']) + start_diff
                new_pass = str(last_pass + 1)
                new_str = '0' * (3-len(new_pass)) + new_pass

                combo_dict['total_passes'] = new_pass
                combo_dict['pass_valid'+new_str] = new_dict['pass_valid001']
                combo_dict['pass_pos'+new_str] = new_dict['pass_pos001']
                combo_dict['pass_time'+new_str] = str(pass_time_new)
                combo_dict['experiment_start'] = (' ').join((start1_date, start1_timecode))

                # print(combo_dict)
                combo_list = [k+':'+v+'\n' for k,v in combo_dict.items()]
                # if not os.path.exists('test'): os.makedirs('test')
                with open(oldfile,'w') as file:
                    for line in combo_list:
                        file.write(line)
            else:
                new_str = '001'
    return new_str
#***********************************************************************************************#
iris_path = input("\nPlease type in the path to the folder that contains the IRIS data:\n")
if iris_path == 'test':
    iris_path = "/Volumes/KatahdinHD/ResilioSync/DATA/IRIS/tCHIP_results/tXbCHIP025_VSV-EBOVmak@1E6-Dylight549"
iris_path = iris_path.strip('"')##Point to the correct directory
os.chdir(iris_path)

if sys.platform == 'win32': splitter = ('\\')
elif sys.platform == 'darwin': splitter = ('/')

xml_list = [file for file in sorted(glob.glob('*/*.xml'))]
if not xml_list: xml_list = [file for file in sorted(glob.glob('../*/*.xml'))]
print("{} chipFiles detected".format(len(xml_list)))

for xfile in xml_list:
    chip_name = (xfile.split(splitter)[1]).split('.')[0]

    print(chip_name)
    folder_name = chip_name
    # number = 1
    all_files = sorted(glob.glob('{}.*'.format(chip_name)))
    pgm_list, csv_list, png_list, txt_list, tiff_list, fluor_files = [],[],[],[],[],[]
    for file in all_files:
        if file.endswith('.pgm'):
            pgm_list.append(file)
        elif file.endswith('.txt'):
            txt_list.append(file)
        elif file.endswith('.csv'):
            csv_list.append(file)
        elif file.endswith('.png'):
            png_list.append(file)
        elif file.endswith('.tif'):
            tiff_list.append(file)
            print("TIFF images detected")

    if pgm_list != []:
        fluor_files = [file for file in pgm_list if file.split(".")[-2] in 'ABC']
        if fluor_files != []:
            print("Fluorescent images detected")
            pgm_list = fluor_cleaner(pgm_list, splitter)
        total_spots = max([int(val.split('.')[1]) for val in pgm_list])
        if csv_list:
            print("\nNanoanalysis files detected: {}\n".format(csv_list))
            file_deleter(csv_list)
        if png_list:
            print("\nNanoanalysis files detected: {}\n".format(png_list))
            file_deleter(png_list)

        if (len(tiff_list) != total_spots):
            print("Converting TIFFs")
            img_list = tiff_maker(pgm_list, archive = True)
        else:
            img_list = tiff_list

        iris_data_list = img_list + txt_list + fluor_files

        if not iris_data_list == []:

            if os.path.exists(folder_name):
                os.chdir(folder_name)

                try: new_str = txt_combiner(txt_list, splitter)
                except UnboundLocalError: new_str = '002'

                for txtfile in txt_list:
                    if len(txtfile.split('.')) < 4:
                        os.remove('../' + txtfile)
                os.chdir(iris_path)

                print(iris_data_list)
                for filename in iris_data_list:
                    filesplit = filename.split('.')
                    if filesplit[2].isdigit():
                        filesplit[2] = new_str
                        newname = '.'.join(filesplit)
                        print(newname)
                        dest = '{}{}{}'.format(folder_name, splitter, newname)
                        os.rename(filename, dest)
            else:
                os.makedirs(folder_name)
                for filename in iris_data_list:
                    os.rename(filename, '{}{}{}'.format(folder_name, splitter, filename))

            mirror_file = str(glob.glob(folder_name+'.000.000.pgm')).strip("'[]'")
            final_mirror_loc = '{}{}{}'.format(folder_name, splitter, mirror_file)
            if not os.path.isfile(final_mirror_loc):
                os.rename(mirror_file, final_mirror_loc)
            else:
                os.remove(mirror_file)
