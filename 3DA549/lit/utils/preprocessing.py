import numpy as np
import os
import cv2
import click
from tqdm import tqdm

from skimage import exposure
from skimage.morphology import watershed

from scipy.ndimage.morphology import binary_fill_holes


BATCH_SIZE = 8
C_MASK_THRESHOLD = 230
MARKER_THRESHOLD = 240


# [marker_threshold, cell_mask_thershold]
THRESHOLDS = {
    'DIC-C2DH-HeLa' : [240, 230],
    'Fluo-N2DH-SIM+' : [240, 230],
    'PhC-C2DL-PSC' : [235, 90]}


# first channel -- normalized input
def median_normalization(image):
    image_ =  image / 255 + (.5 - np.median(image / 255))
    return np.maximum(np.minimum(image_, 1.), .0)

def hist_equalization(image):
    return cv2.equalizeHist(image) / 255
     
def get_normal_fce(normalization):
    if normalization == 'HE':
        return hist_equalization 
    if normalization == 'MEDIAN':
        return median_normalization
    else:
        # error('normalization function was not picked')
        return None

# second channel -- cell markers
def preprocess_markers(m, threshold=240, erosion_size=12, circular=False, step=4):
    # threshold 去背景
    m = m.astype(np.uint8)
    _, new_m = cv2.threshold(m, threshold, 255, cv2.THRESH_BINARY)
    
    # distance transform | only for circular objects
    if circular:
        dist_m = (cv2.distanceTransform(new_m, cv2.DIST_L2, 5) * 5).astype(np.uint8)
        new_m = hmax(dist_m, step=step).astype(np.uint8)

    # filling gaps
    # 填充hole
    hol = binary_fill_holes(new_m).astype(np.uint8)

    # morphological opening
    # 先腐蚀，后膨胀。去除图像中小的亮点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    clos = cv2.morphologyEx(hol, cv2.MORPH_OPEN, kernel)

    # label connected components
    idx, res = cv2.connectedComponents(clos)

    return idx, res


def hmax(ml, step=50):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    ml = cv2.blur(ml, (3, 3))
    
    rec1 = np.maximum(ml.astype(np.int32) - step, 0).astype(np.uint8)

    for i in range(255):
        rec0 = rec1
        rec1 = np.minimum(cv2.dilate(rec0, kernel), ml.astype(np.uint8))
        if np.sum(rec0 - rec1) == 0:
            break

    return ml - rec1 > 0 


# third channel - cell mask
def preprocess_cell_mask(b, threshold=230):
    # tresholding
    b = b.astype(np.uint8)
    bt = cv2.inRange(b, threshold, 255)
    return bt

def remove_uneven_illumination(img, blur_kernel_size = 501):
    '''
    uses LPF to remove uneven illumination
    '''
   
    img_f = img.astype(np.float32)
    img_mean = np.mean(img_f)
    
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




# read images
def load_images(path, cut=False, new_mi=0, new_ni=0, normalization='HE', uneven_illumination=False):

    names = os.listdir(path)
    names.sort()

    mi, ni = get_image_size(path)

    dm = (mi % 16) // 2
    mi16 = mi - mi % 16
    dn = (ni % 16) // 2
    ni16 = ni - ni % 16

    total = len(names)
    normalization_fce = get_normal_fce(normalization)

    image = np.empty((total, mi, ni, 1), dtype=np.float32)

    for i, name in enumerate(names):

        o = read_image(os.path.join(path, name))

        if o is None:
            print('image {} was not loaded'.format(name))

        if uneven_illumination:
            o = np.minimum(o, 255).astype(np.uint8)
            o = remove_uneven_illumination(o) 

        image_ = normalization_fce(o)

        image_ = image_.reshape((1, mi, ni, 1)) - .5
        image[i, :, :, :] = image_

    if cut:
        image = image[:, dm:mi16+dm, dn:ni16+dn, :]
    if new_ni > 0 and new_ni > 0:
        image2 = np.zeros((total, new_mi, new_ni, 1), dtype=np.float32)
        image2[:, :mi, :ni, :] = image
        image = image2

    print('loaded images from directory {} to shape {}'.format(path, image.shape))
    return image


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

    print(predictions.shape)
    print(input_images.shape)
    viz_path = res_path.replace('_RES', '_VIZ')

    for i in range(predictions.shape[0]):

        m = predictions[i, :, :, 1] * 255
        c = predictions[i, :, :, 3] * 255
        o = (input_images[i, :, :, 0] + .5) * 255
        o_rgb = cv2.cvtColor(o, cv2.COLOR_GRAY2RGB)

        # postprocess the result of prediction
        idx, markers = postprocess_markers(m, threshold=thr_markers, erosion_size=erosion_size, circular=circular, step=step)
        cell_mask = postprocess_cell_mask(c, threshold=thr_cell_mask)

        # correct border
        cell_mask = np.maximum(cell_mask, markers)

        labels = watershed(-c, markers, mask=cell_mask)
        # labels = markers

        labels_rgb = cv2.applyColorMap(labels.astype(np.uint8)*15, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(o_rgb.astype(np.uint8), 0.5, labels_rgb, 0.5, 0)

        # store result
        cv2.imwrite('{}/mask{:03d}.tif'.format(res_path, i), labels.astype(np.uint16))

        if viz:

            m_rgb = cv2.cvtColor(m.astype(np.uint8),cv2.COLOR_GRAY2RGB)
            c_rgb = cv2.cvtColor(c.astype(np.uint8),cv2.COLOR_GRAY2RGB)

            result = np.concatenate((m_rgb, c_rgb, overlay), 1)
            cv2.imwrite( '{}/res{:03d}.tif'.format(viz_path, i), result)
            cv2.imwrite('{}/markers{:03d}.tif'.format(viz_path, i), markers.astype(np.uint8) * 16)



@click.command()
@click.option('--name', prompt='path to dataset',
              help='Path to the dataset.')
@click.option('--sequence', prompt='sequence number',
              help='Dataset number.')
@click.option('--viz/--no-viz', default=True,
              help='true if produces also viz images.')
def predict_dataset(name, sequence, viz=True):
    """
    reads images from the path and converts them to the np array
    """

    dataset_path = name

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
    model_name = [name for name in os.listdir(dataset_path) if 'PREDICT' in name]
    assert len(model_name) == 1, 'ambiguous choice of nn model, use keyword "PREDICT" exactly for one ".h5" file'
    
    model_init_path = os.path.join(dataset_path, model_name[0])
    store_path = os.path.join('.', name, '{}_RES'.format(sequence))
    
    if not os.path.isdir(store_path):
        os.mkdir(store_path)
        print('directory {} was created'.format(store_path))

    if not os.path.isfile(model_init_path):
        print('there is no init model for this dataset')
        exit()

    img_path = os.path.join('.', name, sequence)
    if not os.path.isdir(img_path):
        print('given name of dataset or the sequence is not valid')
        exit()

    if not os.path.isfile(os.path.join(name, 'model_init.h5')):
        print('directory {} do not contain model file'.format(name))

    mi, ni = get_image_size(img_path)
    new_mi = get_new_value(mi)
    new_ni = get_new_value(ni)

    print(mi, ni)
    print(new_mi, new_ni)

    model_init = create_model(model_init_path, new_mi, new_ni)

    input_img = load_images(img_path, \
                            new_mi=new_mi, \
                            new_ni=new_ni, \
                            normalization=NORMALIZATION, \
                            uneven_illumination=UNEVEN_ILLUMINATION)
                            
    img_number = len(input_img)

    pred_img = model_init.predict(input_img, batch_size=BATCH_SIZE)
    print('pred shape: {}'.format(pred_img.shape))

    org_img = load_images(img_path)
    pred_img = pred_img[:, :mi, :ni, :]

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
    m = cv2.imread('raw.tif', cv2.IMREAD_GRAYSCALE)
    idx, markers = preprocess_markers(m, 100)
    cell_mask = preprocess_cell_mask(m, threshold=90)
    cv2.imwrite('{}/markers_001.tif'.format('.'), markers.astype(np.uint8) * 16)
    cv2.imwrite('{}/mask_001.tif'.format('.'), cell_mask.astype(np.uint8) * 16)