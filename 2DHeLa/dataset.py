from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import os
import torch
from PIL import Image

class litDataset(Dataset):
    '''
    root path: /src/path/image/

    img path: /src/path/image/a.tif
    mask path /src/path/mask/a.tif
    '''
    def __init__(self, path):
        self.path = path
        self.path_list = self.get_img_list(path)
        self.transforms = transforms.Compose([
            # transforms.Resize(256),
            # transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.],    # normalize with (x - mean) / std
                                 std=[1.])
        ])
        self.uneven_illumination = False
        self.erode_kernel = 10
        self.dim = 1
        self.target_dim = 4

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        img_path = self.path_list[index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if self.uneven_illumination:
            img = np.minimum(img, 255).astype(np.uint8)
            img = remove_uneven_illumination(img)

        img = hist_equalization(img) - 0.5
        # img = median_normalization(img) - 0.5
        img = Image.fromarray(np.array(img))
        img = self.transforms(img)
        # cv2.imshow('img', img.squeeze().numpy()), cv2.waitKey(0)
        _, w, h = img.size()

        img_name = os.path.split(img_path)[-1]
        mask_path = os.path.join(self.path, 'mask', img_name)
        mask_path = mask_path.replace(img_name, "man_seg"+img_name[1:])

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (w, h))
        mask = np.array(mask).astype(np.uint8)
        ## also you can use the ground truth of the tracking data as marker.
        marker = np.zeros((w, h), dtype=np.uint8)
        labels = np.unique(mask[mask != 0]).astype(np.uint8)
        for label in labels:
            tmp = np.zeros((w, h), dtype=np.uint8)
            tmp[mask == label] = label
            kernel = np.ones((3, 3), dtype=np.uint8)
            tmp = cv2.erode(tmp, kernel, iterations=5)
            marker = cv2.addWeighted(marker, 1, tmp, 1, 0).reshape((w, h))
        # mask[mask != 0] = 255
        # marker[marker != 0] = 255

        mask_tensor = torch.zeros((2, w, h))
        marker_tensor = torch.zeros((2, w, h))

        ww = 255 - np.max(mask)
        mask[mask != 0] += ww
        marker[marker != 0] += ww

        mask_tensor[0, :, :] = torch.from_numpy(mask / np.max(mask))
        mask_tensor[1, :, :] = torch.from_numpy(1 - mask / np.max(mask))
        marker_tensor[0, :, :] = torch.from_numpy(marker / np.max(marker))
        marker_tensor[1, :, :] = torch.from_numpy(1 - marker / np.max(marker))
        gt = torch.cat([marker_tensor, mask_tensor], dim=0)

        # cv2.imshow('mask_tensor', mask_tensor[0, :, :].data.numpy()), cv2.waitKey(0)
        # cv2.imshow('marker_tensor', marker_tensor[0, :, :].data.numpy()), cv2.waitKey(0)
        '''
        temp = np.zeros((w, h, 4))
        for channel in range(4):
            temp[:, :, channel] = gt[channel, :, :].numpy()
            cv2.imshow(img_name + 'c{}'.format(channel), temp[:, :, channel]), cv2.waitKey(0)
        cv2.imwrite('D:\\LitData\\DIC-C2DH-HeLa\\TRAIN\\01_target\\{}'.format(img_name), temp.astype(np.float32))
        '''
        return img, gt

    def get_img_list(self, path):
        '''
        A folder that contains all the training data.
        '''
        print('Load images from path:\n    {}'.format(path))
        path_list = []
        names = os.listdir(os.path.join(path, 'image'))
        for name in names:
            path_list.append(os.path.join(path, 'image', name))
        return path_list


def remove_uneven_illumination(img, blur_kernel_size = 50):
    '''
    uses LPF to remove uneven illumination
    '''
    img_f = img.astype(np.float32)
    img_mean = np.mean(img_f)
    img_blur = cv2.GaussianBlur(img_f,(blur_kernel_size, blur_kernel_size), 0)
    result = np.maximum(np.minimum((img_f - img_blur) + img_mean, 255), 0).astype(np.int32)
    return result

def median_normalization(image):
    image_ =  image / 255 + (.5 - np.median(image / 255))
    return np.maximum(np.minimum(image_, 1.), .0)

def hist_equalization(image):
    # 直方均衡化
    return cv2.equalizeHist(image) / 255

