# Shuffles dataset in place

import argparse
import glob
import random

from PIL import Image, ImageDraw
import os
import tempfile
import shutil
import csv


parser = argparse.ArgumentParser()

parser.add_argument(
    "-dir", "--directory", help="Input directory (should contain images)"
)
parser.add_argument("-out", "--output", help="Output directory")
args = parser.parse_args()


if args.directory == None or args.output == None:
    print("Missing parameters \n")
    parser.print_help()
    exit(1)

files = []
for file in os.listdir(args.directory):
    if ".jpg" in file:
        name = file.replace(".jpg", "")
        files.append(name)

if len(files) == 0:
    print("No files found")
    exit(1)

random.shuffle(files)


count = 1
for f in files:
    old_img_name = f + ".jpg"
    old_ann_name = f + ".txt"
    new_name = str(count).rjust(3, "0")
    new_img_name = new_name + ".jpg"
    new_ann_name = new_name + ".txt"
    shutil.copyfile(
        os.path.join(args.directory, old_img_name),
        os.path.join(args.output, new_img_name),
    )
    shutil.copyfile(
        os.path.join(args.directory, old_ann_name),
        os.path.join(args.output, new_ann_name),
    )

    count += 1

print("\n\nDONE")
