import torch

from utils.fr_model.irse import MobileFaceNet
from utils.fr_model.margin import ArcMarginProduct


def get_fr_model(args, pretrained):
    fr_model = MobileFaceNet(512)
    if pretrained:
        fr_model.load_state_dict(torch.load(args.fr_model_ckpt_path))
    fr_model = fr_model.cuda()
    fr_model.eval()
    return fr_model



def get_margin(args, trainset_class_nums, pretrained_model_path=None, margin_m=0.5):
    margin = ArcMarginProduct(args.fr_train_feature_dim, trainset_class_nums, s=args.fr_train_scale_size, m=margin_m)
    if pretrained_model_path is not None:
        margin.load_state_dict(torch.load(pretrained_model_path))
        print("Loaded pretrained weight for the margin")
    margin = margin.to(args.device)
    return margin