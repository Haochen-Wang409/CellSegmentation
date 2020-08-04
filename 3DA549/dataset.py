from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import os
import torch
from PIL import Image
import SimpleITK as sitk

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
            transforms.Resize((256, 256)),
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

        img_name = os.path.split(img_path)[-1]
        mask_path = os.path.join(self.path, 'mask', img_name)
        mask_path = mask_path.replace(img_name, "man_seg" + img_name[1:])

        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        itk_img = sitk.ReadImage(img_path)
        imgs = sitk.GetArrayFromImage(itk_img)
        itk_mask = sitk.ReadImage(mask_path)
        masks = sitk.GetArrayFromImage(itk_mask)

        temp_img = median_normalization(imgs[0, :, :])
        temp_img = Image.fromarray(np.array(temp_img))
        temp_img = self.transforms(temp_img)

        num = imgs.shape[0]
        _, width, height = temp_img.shape

        img = np.zeros((1, 64, width, height))
        gt = torch.zeros((4, 64, width, height))

        for i in range(64):
            temp_img = imgs[i, :, :]

            if self.uneven_illumination:
                temp_img = np.minimum(temp_img, 255).astype(np.uint8)
                temp_img = remove_uneven_illumination(temp_img)

            # temp_img = hist_equalization(temp_img) - 0.5
            temp_img = median_normalization(temp_img) - 0.5
            temp_img = Image.fromarray(np.array(temp_img))
            temp_img = self.transforms(temp_img)
            # cv2.imshow('img', img.squeeze().numpy()), cv2.waitKey(0)
            _, w, h = temp_img.size()
            img[0, i, :, :] = temp_img

            temp_mask = np.array(masks[i, :, :]).astype(np.float32)
            temp_mask = Image.fromarray(temp_mask)
            temp_mask = np.array(self.transforms(temp_mask)).squeeze()

            ## also you can use the ground truth of the tracking data as marker.
            temp_marker = np.zeros((w, h))
            labels = np.unique(temp_mask[temp_mask != 0])
            for label in labels:
                tmp = np.zeros((w, h))
                tmp[temp_mask == label] = label
                kernel = np.ones((3, 3))
                tmp = cv2.erode(tmp, kernel, iterations=5)
                temp_marker = cv2.addWeighted(temp_marker, 1, tmp, 1, 0).reshape((w, h))
            temp_mask[temp_mask != 0] = 1
            temp_marker[temp_marker != 0] = 1

            temp_mask_tensor = torch.zeros((2, w, h))
            temp_marker_tensor = torch.zeros((2, w, h))

            # ww = 255 - np.max(temp_mask)
            # temp_mask[temp_mask != 0] += ww
            # temp_marker[temp_marker != 0] += ww

            if (np.max(temp_mask) > 0) and (np.max(temp_marker) > 0):
                temp_mask_tensor[0, :, :] = torch.from_numpy(temp_mask / np.max(temp_mask))
                temp_mask_tensor[1, :, :] = torch.from_numpy(1 - temp_mask / np.max(temp_mask))
                temp_marker_tensor[0, :, :] = torch.from_numpy(temp_marker / np.max(temp_marker))
                temp_marker_tensor[1, :, :] = torch.from_numpy(1 - temp_marker / np.max(temp_marker))
            else:
                temp_mask_tensor[0, :, :] = torch.from_numpy(temp_mask)
                temp_mask_tensor[1, :, :] = torch.from_numpy(1 - temp_mask)
                temp_marker_tensor[0, :, :] = torch.from_numpy(temp_marker)
                temp_marker_tensor[1, :, :] = torch.from_numpy(1 - temp_marker)

            gt[:, i, :, :] = torch.cat([temp_marker_tensor, temp_mask_tensor], dim=0)

        # print(img.shape, gt.shape)

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

