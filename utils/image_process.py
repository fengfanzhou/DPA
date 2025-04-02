import torch.nn.functional as F

def bgr2rgb(img):
    permute = [2, 1, 0]
    result = img[:, permute, :, :]
    return result


def img_normalize(img):
    img = img / 255
    mean = 0.5
    std = 0.5
    img = (img - mean) / std
    return img



def preprocess_batch_img(x):
    """
    Preprocess batch image.
    :param x: BCHW, non-normalized, BGR
    :return: x: BCHW, normalized, RGB
    """
    x = bgr2rgb(x)
    x = img_normalize(x)
    return x


