import click
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import argparse

from scipy import ndimage


def main(path, output_path, threshold=0.15):

    # check if path exists
    if not os.path.isdir(path):
        print('input path is not a valid path')
        return

    names = os.listdir(path)
    names = [name for name in names if '.tif' in name and 'mask' in name]
    names.sort()

    # cv2.IMREAD_ANYDEPTH: read image with ANY DEPTH
    img = cv2.imread(os.path.join(path, names[0]), cv2.IMREAD_ANYDEPTH)
    mi, ni = img.shape
    print('Relabelling the segmentation masks.')
    records = {}

    old = np.zeros((mi, ni))
    index = 1
    n_images = len(names)

    for i, name in enumerate(names):
        result = np.zeros((mi, ni), np.uint16)

        img = cv2.imread(os.path.join(path, name), cv2.IMREAD_ANYDEPTH)

        labels = np.unique(img)[1:]

        parent_cells = []

        for label in labels:
            # mask[i, j] = 1 MEANS img[i, j] == label
            mask = (img == label) * 1

            # old: the mask of label last
            # mask: the mask of label now
            # overlap: the intersection of last and now (include other cells possibly)
            mask_size = np.sum(mask)
            overlap = mask * old
            candidates = np.unique(overlap)[1:]

            max_score = 0
            max_candidate = 0

            for candidate in candidates:
                # select the most possible label for the cell
                score = np.sum(overlap == candidate * 1) / mask_size
                if score > max_score:
                    max_score = score
                    max_candidate = candidate

            if max_score < threshold:
                # no parent cell detected, create new track
                # index means the contemporary label
                records[index] = [i, i, 0]
                result = result + mask * index
                index += 1
            else:

                if max_candidate not in parent_cells:
                    # prolonging track
                    records[max_candidate][1] = i
                    result = result + mask * max_candidate

                else:
                    # split operations
                    # if have not been done yet, modify original record
                    if records[max_candidate][1] == i:
                        records[max_candidate][1] = i - 1
                        # find mask with max_candidate label in the result and rewrite it to index
                        m_mask = (result == max_candidate) * 1
                        result = result - m_mask * max_candidate + m_mask * index

                        records[index] = [i, i, max_candidate.astype(np.uint16)]
                        index += 1

                    # create new record with parent cell max_candidate
                    records[index] = [i, i, max_candidate.astype(np.uint16)]
                    result = result + mask * index
                    index += 1

                # update of used parent cells
                parent_cells.append(max_candidate)
        # store result
        cv2.imwrite(os.path.join(output_path, name), result.astype(np.uint16))
        old = result

    # store tracking
    print('Generating the tracking file.')
    with open(os.path.join(output_path, 'res_track.txt'), "w") as file:
        for key in records.keys():
            file.write('{} {} {} {}\n'.format(key, records[key][0], records[key][1], records[key][2]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str,
                        default='D:\\LitData\\Win\\DIC-C2DH-HeLa\\01_RES',
                        help='path to the result segmentation')
    parser.add_argument('--output_path', type=str,
                        default='D:\\LitData\\Win\\DIC-C2DH-HeLa\\01_RES')

    args = parser.parse_args()

    main(args.path, args.output_path)
