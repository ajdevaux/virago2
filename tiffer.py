#!/usr/bin/env python3
from future.builtins import input
from modules.vpipes import tiff_maker
from os import chdir
from glob import glob

def main():
    pgm_list = []
    while pgm_list == []: ##Keep repeating until pgm files are found
        iris_path = input("\nPlease type in the path to the folder that contains the IRIS data:\n")
        chdir(iris_path)
        pgm_list = sorted(glob('*.pgm'))

    tiff_list = tiff_maker(pgm_list, tiff_compression = 1, archive = False)

    print(tiff_list)

if __name__ == '__main__':
    main()
