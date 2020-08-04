import torch
import torch.nn as nn
import numpy as np
import argparse
from tensorboardX import *
from torch.utils.data import DataLoader
import datetime
from tqdm import *
import os
import cv2

from lit.model import unet
from lit.loss import dice_loss
from lit.metrics import get_ious

def calc(pred, y):
    smooth = 1.
    p_flat = pred.reshape(-1)
    t_flat = y.reshape(-1)
    intersection = sum((p_flat * t_flat))
    a_sum = sum(p_flat * p_flat)
    b_sum = sum(t_flat * t_flat)
    return 1. - ((2. * intersection + smooth) / (a_sum + b_sum + smooth))

def main(args):
    gt_path = args.gt_path
    gt_names = os.listdir(gt_path)
    gt_names.sort()
    gt_masks = [name for name in gt_names if 'mask' in name]
    gt_masks.sort()

    test_path = args.test_path
    test_names = os.listdir(test_path)
    test_names.sort()
    test_masks = [name for name in test_names if 'mask' in name]
    test_masks.sort()

    total = len(gt_masks) // 2
    acc = []

    print('\nBegin to calculate accuracy ...')
    for i in tqdm(range(total)):
        gt_name, test_name = gt_masks[i], test_masks[i]
        gt_img = cv2.imread(os.path.join(gt_path, gt_name), cv2.IMREAD_UNCHANGED)
        gt_img[gt_img != 0] = 1
        test_img = cv2.imread(os.path.join(test_path, test_name), cv2.IMREAD_UNCHANGED)
        test_img[test_img != 0] = 1
        acc.append(calc(gt_img, test_img))

    import xlwt
    wb = xlwt.Workbook()
    sheet1 = wb.add_sheet('sheet1', cell_overwrite_ok=True)
    for i, x in enumerate(acc):
        sheet1.write(i, 0, str(x))
    wb.save('temp.xls')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gt_path', type=str,
                        default='D:\\LitData\\DIC-C2DH-HeLa\\TEST\\01_RES_MU')
    parser.add_argument('--test_path', type=str,
                        default='D:\\LitData\\DIC-C2DH-HeLa\\TEST\\01_RES')

    args = parser.parse_args()

    main(args)