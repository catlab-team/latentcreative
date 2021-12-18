import os, sys, glob, argparse
import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--input_size', type=int, required=True)
parser.add_argument('--output_size', type=int, required=True)
parser.add_argument('--row', type=int, required=True)
parser.add_argument('--col', type=int, required=True)
args = parser.parse_args()

## PARAMETERS ##
image_path = args.image_path
input_size = args.input_size
output_size = args.output_size
row_i = args.row - 1 
col_i = args.col - 1
################

def partition_image(img):
    rows = []
    start_idx = 0
    while (start_idx+input_size<=img.shape[0]):
        row = img[start_idx:start_idx+input_size,:,:]
        start_idx+=input_size+1
        rows.append(row)
    return rows


def extract_images(row):
    images = []
    start_idx = 0
    while (start_idx+input_size<=row.shape[1]):
        img = row[:, start_idx:start_idx+input_size,:]
        start_idx+=input_size+1
        images.append(img)
    return images


whole_img = cv2.imread(image_path)
target_row = partition_image(whole_img)[row_i]
img = extract_images(target_row)[col_i]
rect_h = 30 if input_size == 256 else 60
rect_w = rect_h*2
img = cv2.rectangle(img, (0,0), (rect_w, rect_h), (0,0,0), -1)
img = cv2.resize(img, (output_size, output_size))
cv2.imwrite(args.output_path, img)
