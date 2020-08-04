from scipy.interpolate import interp1d
import numpy as np
import PIL.Image as Image
import SimpleITK as sitk
import argparse
import os

from tqdm import *


def main(args):
    temp = 0
    data_path = args.path
    sequences = args.sequence

    for sequence in sequences:
        print('\nBegin to preprocess sequence {}'.format(sequence))

        img_path = os.path.join(data_path, sequence)
        seg_path = os.path.join(data_path, '{}_GT'.format(sequence), 'SEG')
        img_names = os.listdir(img_path)
        img_names.sort()
        seg_names = os.listdir(seg_path)
        seg_names.sort()

        total = len(seg_names)
        for i in tqdm(range(total)):
            seg_name = seg_names[i]
            img_name = 't' + str(seg_name[7:])

            itk_in_img = sitk.ReadImage(os.path.join(img_path, img_name))
            in_img = sitk.GetArrayFromImage(itk_in_img).astype(np.float32)
            # sitk.WriteImage(itk_in_img, os.path.join(data_path, 'image', 't{:03d}.tif'.format(i + temp)))

            itk_seg_img = sitk.ReadImage(os.path.join(seg_path, seg_name))
            seg_img = sitk.GetArrayFromImage(itk_seg_img).astype(np.float32)
            # sitk.WriteImage(itk_seg_img, os.path.join(data_path, 'mask', 'man_seg{:03d}.tif'.format(i + temp)))

            depth, width, height = seg_img.shape
            img = np.zeros((64, 256, 256), dtype=np.float32)
            seg = np.zeros((64, 256, 256), dtype=np.float32)

            img[0, :, :] = Image.fromarray(in_img[0, :, :]).resize((256, 256))
            seg[0, :, :] = Image.fromarray(seg_img[0, :, :]).resize((256, 256))
            img[63, :, :] = Image.fromarray(in_img[depth-1, :, :]).resize((256, 256))
            seg[63, :, :] = Image.fromarray(seg_img[depth-1, :, :]).resize((256, 256))

            for k in range(1, 63):
                y1 = np.zeros((width, height), dtype=np.float32)
                y2 = np.zeros((width, height), dtype=np.float32)
                # We have (x1, y1) and (x2, y2)
                # where x1 = int(k/63*(depth-1)),     x2 = int(k/63*(depth-1)) + 1
                #       y1 = in_img[x1, :, :], y2 = in_img[x2, :, :]
                # line: Y = (y2-y1)/(x2-x1) * (X - x1) + y1
                x1 = int(k/63*(depth-1))
                x2 = int(k/63*(depth-1)) + 1
                y1[:, :] = in_img[x1, :, :]
                y2[:, :] = in_img[x2, :, :]
                kk = k/63*(depth-1)
                temp_img = Image.fromarray((y2-y1)/(x2-x1) * (kk-x1) + y1).resize((256, 256))
                img[k, :, :] = temp_img

                y1[:, :] = seg_img[x1, :, :]
                y2[:, :] = seg_img[x2, :, :]
                temp_seg = Image.fromarray((y2-y1)/(x2-x1) * (kk-x1) + y1).resize((256, 256))
                seg[k, :, :] = temp_seg

            '''
            x = np.arange(depth)
            for ii in range(256):
                for jj in range(256):
                    y1 = in_img[:, ii, jj]
                    li = interp1d(x, y1, kind='cubic')
                    y2 = seg_img[:, ii, jj]
                    ls = interp1d(x, y2, kind='cubic')
                    for k in range(64):
                        img[k, ii, jj] = li(k/63*(depth-1))
                        seg[k, ii, jj] = ls(k/63*(depth-1))
            '''


            sitk.WriteImage(sitk.GetImageFromArray(img.astype(np.uint16)), os.path.join(data_path, 'image', 't{:03d}.tif'.format(i + temp)))
            sitk.WriteImage(sitk.GetImageFromArray(seg.astype(np.uint16)), os.path.join(data_path, 'mask', 'man_seg{:03d}.tif'.format(i + temp)))

        temp += total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str,
                        default='D:\\LitData\\Fluo-C3DH-A549\\TRAIN')
    parser.add_argument('--sequence', type=list,
                        default=['01', '02'])

    args = parser.parse_args()

    main(args)