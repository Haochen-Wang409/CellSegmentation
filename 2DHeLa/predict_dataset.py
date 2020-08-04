import numpy as np
import os
import cv2
import click
import torchvision
from tqdm import tqdm
import tensorflow as tf
from lit.model import unet
import torch
from PIL import Image

from skimage import exposure
from skimage.segmentation import watershed

from scipy.ndimage.morphology import binary_fill_holes

import argparse

BATCH_SIZE = 8
C_MASK_THRESHOLD = 230
MARKER_THRESHOLD = 240

# [marker_threshold, cell_mask_thershold]
THRESHOLDS = {
    'DIC-C2DH-HeLa' : [240, 230],
    'Fluo-N2DH-SIM+' : [240, 230],
    'PhC-C2DL-PSC' : [235, 90]}

def median_normalization(image):
    # 中位数归一化
    image_ =  image / 255 + (.5 - np.median(image / 255))
    return np.maximum(np.minimum(image_, 1.), .0)

def hist_equalization(image):
    # 直方均衡化
    return cv2.equalizeHist(image) / 255
     
def get_normal_fce(normalization):
    if normalization == 'HE':
        return hist_equalization 
    if normalization == 'MEDIAN':
        return median_normalization
    else:
        error('normalization function was not picked')
        return None


def remove_uneven_illumination(img, blur_kernel_size = 501):
    '''
    uses LPF to remove uneven illumination
    '''
   
    img_f = img.astype(np.float32)
    img_mean = np.mean(img_f)
    # 高斯滤波 (平滑处理)
    img_blur = cv2.GaussianBlur(img_f,(blur_kernel_size, blur_kernel_size), 0)
    result = np.maximum(np.minimum((img_f - img_blur) + img_mean, 255), 0).astype(np.int32)
    
    return result


def get_image_size(path):
    '''
    returns size of the given image
    '''
    names = os.listdir(path)
    name = names[0]
    o = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
    return o.shape[0:2]


def get_new_value(mi, divisor=16):
    if mi % divisor == 0:
        return mi
    else:
        return mi + (divisor - mi % divisor)


def read_image(path):
    if 'Fluo' in path:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if 'Fluo-N2DL-HeLa' in path:
            img = (img / 255).astype(np.uint8)
        if 'Fluo-N2DH-SIM+' in path:
            img = np.minimum(img, 255).astype(np.uint8)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return img


# read images
def load_images(path, cut=False, new_mi=0, new_ni=0, normalization='HE', uneven_illumination=False, label=False):
    names = os.listdir(path)
    names.sort()

    transform = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(256),
        # torchvision.transforms.RandomCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.],  # normalize with (x - mean) / std
                                         std=[1.])
    ])

    temp = read_image(os.path.join(path, names[0]))
    temp = Image.fromarray(np.array(temp))
    temp = transform(temp)

    _, mi, ni = temp.shape

    dm = (mi % 16) // 2
    mi16 = mi - mi % 16
    dn = (ni % 16) // 2
    ni16 = ni - ni % 16

    if args.phase == 'TRAIN':
        total = len(names)
    else:
        total = len(names) // 2

    normalization_fce = get_normal_fce(normalization)

    image = np.empty((total, mi, ni, 1), dtype=np.float32)
    print('Begin to load images ...')
    for i in range(total):
        name = names[i]
        o = read_image(os.path.join(path, name))
        if o is None:
            print('image {} was not loaded'.format(name))

        if uneven_illumination:
            o = np.minimum(o, 255).astype(np.uint8)
            o = remove_uneven_illumination(o) 

        image_ = hist_equalization(o) - 0.5
        # image_ = median_normalization(o) - 0.5
        # cv2.imshow('image_', image_), cv2.waitKey(0)

        image_ = Image.fromarray(np.array(image_))
        image_ = transform(image_)

        image_ = image_.reshape((1, mi, ni, 1))
        image[i, :, :, :] = image_

    print('loaded images from directory {} to shape {}'.format(path, image.shape))
    return image


def create_model(model_path):
    model = unet.UNet(
        num_kernel=8,
        kernel_size=3,
        dim=1,
        target_dim=4
    )
    model = model.to('cpu')
    print('Model has been created.')
    model.load_state_dict(torch.load(model_path))
    print('Model parameters has been loaded from path:\n    {}'.format(model_path))
    return model

# postprocess markers
def postprocess_markers(m, threshold=10, erosion_size=12, circular=False, step=4):
    # threshold, 像素值大于threshold的像素点设置为255
    m = m.astype(np.uint8)
    _, new_m = cv2.threshold(m, threshold, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh', new_m), cv2.waitKey(0)

    # distance transform | only for circular objects
    if circular:
        dist_m = (cv2.distanceTransform(new_m, cv2.DIST_L2, 5) * 5).astype(np.uint8)
        new_m = hmax(dist_m, step=step).astype(np.uint8)

    # filling gaps
    hol = binary_fill_holes(new_m).astype(np.uint8)

    # morphological opening
    # cv2.getStructuringElement: 返回指定形状和尺寸的结构元素
    # cv2.morphologyEx: 形态学变换
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    clos = cv2.morphologyEx(hol, cv2.MORPH_OPEN, kernel)

    # label connected components
    # cv2.connectedComponents: 连通分析，去除极小连通块，标记大连通块 (降噪)
    idx, res = cv2.connectedComponents(clos)

    return idx, res

    
def hmax(ml, step=50):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # cv2.blur: 均值滤波
    ml = cv2.blur(ml, (3, 3))
    
    rec1 = np.maximum(ml.astype(np.int32) - step, 0).astype(np.uint8)

    for i in range(255):
        rec0 = rec1
        # cv2.dilate: 图像膨胀 (形态学操作)
        rec1 = np.minimum(cv2.dilate(rec0, kernel), ml.astype(np.uint8))
        if np.sum(rec0 - rec1) == 0:
            break

    return ml - rec1 > 0 


# postprocess cell mask
def postprocess_cell_mask(b, threshold=10):

    # tresholding
    # cv2.inRange: 设置阈值, 低于threshold或高于255均变成0
    b = b.astype(np.uint8)
    bt = cv2.inRange(b, threshold, 255)
    return bt


def threshold_and_store(predictions, \
                        input_images, \
                        res_path, \
                        thr_markers=240, \
                        thr_cell_mask=230, \
                        viz=False, \
                        circular=False, \
                        erosion_size=12, \
                        step='4'):
    det_store = True

    print('predictions.shape: {}'.format(predictions.shape))
    print('input_images.shape: {}'.format(input_images.shape))
    viz_path = res_path.replace('_RES', '_VIZ')
    flg = 0
    print('Begin to postprocess ...')
    for i in tqdm(range(predictions.shape[0])):
        # shape[0] means the number of images
        flg += 1
        # m: marker
        # c: mask (cover)

        m = predictions[i, :, :, 0] * 255
        c = predictions[i, :, :, 2] * 255
        o = (input_images[i, :, :, 0] + .5) * 255
        # cv2.cvtColor: 颜色转换
        o_rgb = cv2.cvtColor(o.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        c_rgb = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        m_rgb = cv2.cvtColor(m.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # postprocess the result of prediction
        idx, markers = postprocess_markers(m, erosion_size=erosion_size, circular=circular, step=step)
        cell_mask = postprocess_cell_mask(c)

        # correct border
        cell_mask = np.maximum(cell_mask, markers)

        labels = watershed(-c, markers, mask=cell_mask)
        # labels = markers

        # cv2.applyCorlorMap: 给图片上色
        # cv2.addWeighted: 图像混合加权 (图像叠加)
        labels_rgb = cv2.applyColorMap(labels.astype(np.uint8) * 15, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(o_rgb.astype(np.uint8), 0.5, labels_rgb, 0.5, 0)

        # store result
        result = np.concatenate((m_rgb, c_rgb, overlay), 1)
        cv2.imwrite('{}/res{:03d}.tif'.format(res_path, i), result)
        cv2.imwrite('{}/markers{:03d}.tif'.format(res_path, i), markers.astype(np.uint8) * 16)
        cv2.imwrite('{}/mask{:03d}.tif'.format(res_path, i), labels.astype(np.uint16))

        # if viz:

def my_reshape(img):
    number, ki, mi, ni = img.shape
    new_img = np.zeros((number, mi, ni, ki))
    for i in range(number):
        for k in range(ki):
            temp = img[i, k, :, :]
            new_img[i, :, :, k] = temp
    return new_img.astype(np.float32)

def my_reshapee(img):
    number, mi, ni, ki = img.shape
    new_img = np.zeros((number, ki, mi, ni))
    for i in range(number):
        for k in range(ki):
            temp = img[i, :, :, k]
            new_img[i, k, :, :] = temp
    return new_img.astype(np.float32)

def predict_dataset(name, phase, sequence, viz=True):
    """
    reads images from the path and converts them to the np array
    path example:
        name/phase/sequence/t000.tif
    """

    dataset_path = os.path.join(name, phase, sequence)
    print('Load images from path:\n    {}'.format(dataset_path))
    print('There are images:\n    {}\n in the path above.'.format(os.listdir(dataset_path)))

    # check if there is a model for this dataset
    if not os.path.isdir(dataset_path):
        print('there is no solution for this dataset')
        exit()

    erosion_size = 1
    if 'DIC-C2DH-HeLa' in name:
        erosion_size = 8
        NORMALIZATION = 'HE'
        MARKER_THRESHOLD, C_MASK_THRESHOLD= THRESHOLDS['DIC-C2DH-HeLa']
        UNEVEN_ILLUMINATION = False
        CIRCULAR = False
        STEP = 0
                
    elif 'Fluo-N2DH-SIM+' in name:
        erosion_size = 1
        NORMALIZATION = 'HE'
        MARKER_THRESHOLD, C_MASK_THRESHOLD= THRESHOLDS['Fluo-N2DH-SIM+']
        UNEVEN_ILLUMINATION = False
        CIRCULAR = False
        STEP = 30

    elif 'PhC-C2DL-PSC' in name:
        erosion_size = 1
        NORMALIZATION = 'MEDIAN'
        MARKER_THRESHOLD, C_MASK_THRESHOLD= THRESHOLDS['PhC-C2DL-PSC']
        UNEVEN_ILLUMINATION = True
        CIRCULAR = True
        STEP = 3

    else:
        print('unknown dataset')
        return


    # load model
    model_path = name
    model_name = [name for name in os.listdir(model_path) if 'PREDICT' in name]
    assert len(model_name) == 1, 'ambiguous choice of nn model, use keyword "PREDICT" exactly for one ".h5" file'

    model_init_path = os.path.join(name, model_name[0])
    store_path = os.path.join(name, phase, '{}_RES'.format(sequence))
    
    if not os.path.isdir(store_path):
        os.mkdir(store_path)
        print('directory {} was created'.format(store_path))

    if not os.path.isfile(model_init_path):
        print('there is no init model for this dataset')
        exit()

    img_path = os.path.join(name, phase, sequence)
    if not os.path.isdir(img_path):
        print('given name of dataset or the sequence is not valid')
        exit()

    mi, ni = get_image_size(img_path)
    new_mi = get_new_value(mi)
    new_ni = get_new_value(ni)

    model_init = create_model(model_init_path)

    input_img = load_images(img_path, \
                            new_mi=new_mi, \
                            new_ni=new_ni, \
                            normalization=NORMALIZATION, \
                            uneven_illumination=UNEVEN_ILLUMINATION)
    img_number = len(input_img)
    print('bft, input_img shape: {}'.format(input_img.shape))
    # cv2.imshow('t000.tif', input_img[0, :, :, :]), cv2.waitKey(0)
    input_img = torch.from_numpy(my_reshapee(input_img).astype(np.float32))
    # (height, width, channel) --->> (channel, height, width)
    print('aft, input_img shape: {}'.format(input_img.shape))

    pred_img = model_init(input_img)
    pred_img = my_reshape(pred_img.data.numpy().astype(np.float32))
    # (channel, height, width) --->> (height, width, channel)
    print('pred shape: {}'.format(pred_img.shape))
    # cv2.imshow('train000.tif', pred_img[0, :, :, :]), cv2.waitKey(0)

    org_img = load_images(img_path, label=True)

    print('Result path: {}'.format(store_path))
    threshold_and_store(pred_img, \
                        org_img, \
                        store_path, \
                        thr_markers=MARKER_THRESHOLD, \
                        thr_cell_mask=C_MASK_THRESHOLD, \
                        viz=viz, \
                        circular=CIRCULAR, \
                        erosion_size=erosion_size, \
                        step=STEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='D:\\LitData\\DIC-C2DH-HeLa', help='the path for dataset')
    parser.add_argument('--phase', type=str, default='TEST', help='the type of dataset (TRAIN / TEST)')
    parser.add_argument('--sequence', type=str, default='01', help='the number of dataset to be predicted')
    parser.add_argument('--viz', type=bool, default=False, help='true if produces also viz images.')

    args = parser.parse_args()

    predict_dataset(args.name, args.phase, args.sequence, args.viz)
