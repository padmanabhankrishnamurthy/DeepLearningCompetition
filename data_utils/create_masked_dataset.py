import numpy as np
import os
import SITK
import argparse as argparse
from PIL import Image
from lungmask import mask
import SimpleITK as sitk

def get_masked_image(image_path, model):
    img = np.array(Image.open(image_path))

    # crop by white
    res = np.where(np.mean(img[:, :, 0], axis=1)>100)[0]
    res_img = img[res[0]:res[-1], :, :]
    img = res_img

    # get lung mask
    img_mask = sitk.GetImageFromArray(img)
    segmentation = mask.apply_fused(img, noHU=True)
    segmentation_relu = (segmentation > 0).transpose(1, 2, 0)

    result = np.multiply(segmentation_relu, res_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir')
    parser.add_argument('-output_dir')
    args = parser.parse_args()

    

    for file in os.listdir(args.input_dir):
        if '.png' not in file:
            continue

