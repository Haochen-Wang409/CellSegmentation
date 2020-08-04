from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import *
import cv2
import numpy as np
import os
import torch
from PIL import Image
import argparse

def main(args):
    """Data augmentation by cv2.CenterCrop and cv2.Flip

    Load images from path:
        args.train_data/args.phase/args.sequence
    Load corresponding segmentations from path:
        args.train_data/args.phase/args.sequence_ST/SEG
    """

    data = args.train_data
    phase = args.phase
    sequences = args.sequence
    store_path = args.store_path
    temp = 0

    for sequence in sequences:
        print('\nBegin to preprocess sequence', sequence)

        IN_path = os.path.join(data, phase, sequence)
        SEG_path = os.path.join(data, phase, '{}_ST'.format(sequence), 'SEG')
        TRACK_path = os.path.join(data, phase, '{}_GT'.format(sequence), 'TRA')
        in_names = os.listdir(IN_path)
        in_names.sort()
        seg_names = os.listdir(SEG_path)
        seg_names.sort()
        track_names = os.listdir(TRACK_path)
        track_names.sort()
        track_names = track_names[1:]

        img_number = len(in_names)
        flg = np.zeros((img_number,), dtype=np.uint8)

        print('\nBegin to write original images ...')
        for i in tqdm(range(img_number)):
            in_name, seg_name, track_name = in_names[i], seg_names[i], track_names[i]
            in_path = os.path.join(IN_path, in_name)
            seg_path = os.path.join(SEG_path, seg_name)
            track_path = os.path.join(TRACK_path, track_name)

            in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            track_img = cv2.imread(track_path, cv2.IMREAD_UNCHANGED)

            seg_temp = np.array(seg_img)
            track_temp = np.array(track_img)

            seg_label = np.unique(seg_temp[seg_temp != 0])
            track_label = np.unique(track_temp[track_temp != 0])

            if abs(len(seg_label) - len(track_label)) <= 1:
                flg[i] = 1
                cv2.imwrite(os.path.join(store_path, 'image', 't{:03d}.tif'.format(i + temp)), np.array(in_img))
                cv2.imwrite(os.path.join(store_path, 'mask', 'man_seg{:03d}.tif'.format(i + temp)), np.array(seg_img))

        print('\nBegin to FLIP images and corresponding segmentations HORIZONTALLY ...')
        for i in tqdm(range(img_number)):
            in_name, seg_name = in_names[i], seg_names[i]
            in_path, seg_path = os.path.join(IN_path, in_name), os.path.join(SEG_path, seg_name)

            in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            cv2.flip(in_img, 1, in_img)

            seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            cv2.flip(seg_img, 1, seg_img)

            if flg[i] == 1:
                cv2.imwrite(os.path.join(store_path, 'image', 't{:03d}.tif'.format(i + img_number + temp)), np.array(in_img))
                cv2.imwrite(os.path.join(store_path, 'mask', 'man_seg{:03d}.tif'.format(i + img_number + temp)), np.array(seg_img))

        print('\nBegin to FLIP images and corresponding segmentations VERTICALLY ...')
        for i in tqdm(range(img_number)):
            in_name, seg_name = in_names[i], seg_names[i]
            in_path, seg_path = os.path.join(IN_path, in_name), os.path.join(SEG_path, seg_name)

            in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            cv2.flip(in_img, 0, in_img)

            seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            cv2.flip(seg_img, 0, seg_img)

            if flg[i] == 1:
                cv2.imwrite(os.path.join(store_path, 'image', 't{:03d}.tif'.format(i + 2*img_number + temp)), np.array(in_img))
                cv2.imwrite(os.path.join(store_path, 'mask', 'man_seg{:03d}.tif'.format(i + 2*img_number + temp)), np.array(seg_img))

        print('\nBegin to FLIP images and corresponding segmentations HORIZONTALLY AND VERTICALLY ...')
        for i in tqdm(range(img_number)):
            in_name, seg_name = in_names[i], seg_names[i]
            in_path, seg_path = os.path.join(IN_path, in_name), os.path.join(SEG_path, seg_name)

            in_img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            cv2.flip(in_img, -1, in_img)

            seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
            cv2.flip(seg_img, -1, seg_img)

            if flg[i] == 1:
                cv2.imwrite(os.path.join(store_path, 'image', 't{:03d}.tif'.format(i + 3*img_number + temp)), np.array(in_img))
                cv2.imwrite(os.path.join(store_path, 'mask', 'man_seg{:03d}.tif'.format(i + 3*img_number + temp)), np.array(seg_img))

        temp += 4 * img_number

        print('Sequence {} was completed, temp {}'.format(sequence, temp))

    print('\nData augmentation was COMPELETED!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', type=str,
                        default='D:\\LitData\\DIC-C2DH-HeLa',
                        help='path for the data to be preprocessed')
    parser.add_argument('--sequence', type=list,
                        default=['01', '02'],
                        help='sequence for the data to be preprocessed')
    parser.add_argument('--phase', type=str, default='TRAIN')
    parser.add_argument('--store_path', type=str,
                        default='D:\\LitData\\DIC-C2DH-HeLa\\TRAIN',
                        help='path for images to store after preprocessing')

    args = parser.parse_args()

    main(args)
