import os
from glob import glob
import numpy as np
import scipy.misc
import cv2
from sklearn.utils import shuffle


def gen_batch_function(data_folder='tl_data', batch_size=32, 
                       image_shape=(40, 40)):
 
    labels = []
    image_paths = []
    gre_image_paths = glob(os.path.join(data_folder, 'green', '*.jpg'))
    image_paths.extend(gre_image_paths)
    labels = [2 for _ in range(len(gre_image_paths))]
    red_image_paths = glob(os.path.join(data_folder, 'red', '*.jpg'))
    image_paths.extend(red_image_paths)
    labels.extend([0 for _ in range(len(red_image_paths))])
    ylw_image_paths = glob(os.path.join(data_folder, 'yellow', '*.jpg'))
    image_paths.extend(ylw_image_paths)
    labels.extend([1 for _ in range(len(ylw_image_paths))])
    unk_image_paths = glob(os.path.join(data_folder, 'unknown', '*.jpg'))
    image_paths.extend(unk_image_paths)
    labels.extend([3 for _ in range(len(unk_image_paths))])

    image_paths, labels = shuffle(image_paths, labels)
    images = []
    for image_file in image_paths:
        image = cv2.resize(cv2.imread(image_file), image_shape)
        image = cv2.normalize(image.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        images.append(image)
    num_images = len(images)
    train = int(num_images * 0.7)
    valid = int(num_images * 0.85)
    train_images = images[:train]
    train_labels = labels[:train]
    valid_images = images[train:valid]
    valid_labels = labels[train:valid]
    test_images = images[valid:]
    test_labels = labels[valid:]
    train_images, train_labels = batching(train_images, train_labels, batch_size)
    valid_images, valid_labels = batching(valid_images, valid_labels, batch_size)
    test_images, test_labels = batching(test_images, test_labels, batch_size)
    to_return = (train_images, train_labels, 
                 valid_images, valid_labels, 
                 test_images, test_labels)
    return to_return


def batching(Xs, Ys, batch_size):
    batched_Xs = []
    batched_Ys = []
    for batch_i in range(0, len(Xs), batch_size):
        batched_Xs.append(Xs[batch_i:batch_i+batch_size])
        batched_Ys.append(Ys[batch_i:batch_i+batch_size])
    return np.array(batched_Xs), np.array(batched_Ys)


if __name__ == '__main__':
    gen_batch_function()


