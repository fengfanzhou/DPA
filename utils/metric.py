import torch
import torch.nn.functional as F


def cal_dis(embedding1, embedding2, distance_metric):
    if embedding1.shape != embedding2.shape:
        raise RuntimeError(
            "embedding1.shape should be equal to embedding2.shape. However, gotten embedding1.shape=={}, embedding2.shape=={}".format(
                embedding1.shape, embedding2.shape))
    if distance_metric == "norm_L2_square":
        if len(embedding1.shape) == 1:
            nembedding1 = F.normalize(embedding1, p=2.0, dim=0)
            nembedding2 = F.normalize(embedding2, p=2.0, dim=0)
            dis = torch.sum((nembedding2 - nembedding1) * (nembedding2 - nembedding1))
        elif len(embedding1.shape) == 2:
            nembedding1 = F.normalize(embedding1, p=2.0, dim=1)
            nembedding2 = F.normalize(embedding2, p=2.0, dim=1)
            dis = torch.sum((nembedding2 - nembedding1) * (nembedding2 - nembedding1), dim=1)
        else:
            raise ValueError(
                "embedding1.shape should be 1 or 2. Gotten embedding1.shape == {}".format(embedding1.shape))
    else:
        raise ValueError(f"distance_metric should be 'norm_L2_square'. Got {distance_metric}")
    return dis