import csv

import cv2
import numpy as np
import torch


def read_csv(path):
    data = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


def write_csv(data, path, verbose=True):
    if type(data) != list:
        raise ValueError("type(data) should be list. Gotten type(data)=={}".format(type(data)))
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    if verbose:
        print("{} written".format(path))


def save_pt(d, path):
    torch.save(d, path)
    print("{} saved".format(path))


def read_img(img_path):
    """
    Read the image which is in the path of img_path. The function is written based on  https://github.com/JDAI-CV/FaceX-Zoo/blob/main/data_processor/test_dataset.py which is under the LICENSE of https://github.com/JDAI-CV/FaceX-Zoo/blob/main/LICENSE
    :param img_path: The path of the input image
    :return: The image that has been read. BGR. CHW. non-normalized. type: torch.Tensor
    """
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img = torch.tensor(img, dtype=torch.float)
    if img.ndim == 2:
        img = torch.stack([img, img, img], dim=2)
    img = img.permute((2, 0, 1))
    return img
