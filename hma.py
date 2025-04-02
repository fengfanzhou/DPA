import copy
import os.path
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import get_args
from utils.fr_model.fr_interface import get_fr_model
from utils.image_process import bgr2rgb, img_normalize, preprocess_batch_img
from utils.image_transform import image_transform_process
from utils.loss_ import get_loss_func


def cal_dis(embedding1, embedding2):
    dis = F.cosine_similarity(embedding1, embedding2)
    return dis

def preprocess(x):
    x = bgr2rgb(x)
    x = img_normalize(x)
    return x


def read_img(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img = torch.tensor(img, dtype=torch.float)
    img = img.permute((2, 0, 1))
    return img


def save_img(img_path, img):
    dir_name = os.path.dirname(img_path)
    os.makedirs(dir_name, exist_ok=True)
    img = img.permute((1, 2, 0)).detach().cpu().numpy()
    cv2.imwrite(img_path, img)


def get_module_by_name(model, module_name):
    """
    Get the child module of a PyTorch model by the name of the child model. The name of the child module should have a format of the 0th element that model.named_modules() yields.
    The function is fetched from the https://blog.csdn.net/Augurlee/article/details/117934834 which is under the LICENSE of https://creativecommons.org/licenses/by-sa/4.0/
    :param model: The PyTorch model. Type:torch.nn.Module
    :param module_name: The name of the child module. It should have a format of the 0th element that model.named_modules() yields.
    :return: The super module of the module that has a name of module_name, the module that has a name of module_name
    """
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        if hasattr(model, name):
            model = getattr(model, name)
        else:
            return None, None
    if hasattr(model, name_list[-1]):
        leaf_module = getattr(model, name_list[-1])
        return model, leaf_module
    else:
        return None, None


def update_module(model, module_name, new_module):
    """
    Update the child module of model by the name of the child module and the new module to be used for updating
    :param model: The model to be updated
    :param module_name: The name of the child module to be updated. It should have a format of the 0th element that model.named_modules() yields.
    :param new_module: The new module to be updated
    :return: Empty
    """
    super_module, leaf_module = get_module_by_name(model, module_name)
    setattr(super_module, module_name.split('.')[-1], new_module)


def change_into_hard_model(args, model):
    """
    Previous get_adv_feat_model
    """
    args = copy.deepcopy(args)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            update_module(model, name, HardModelBlock(args, module))


def get_hard_model(args, ckpt_path=None):
    args = copy.deepcopy(args)
    if ckpt_path is None:
        model = get_fr_model(args, pretrained=True)
    else:
        args.fr_model_ckpt_path = ckpt_path
        model = get_fr_model(args, pretrained=True)
    change_into_hard_model(args, model)
    return model


class HardModelBlock(nn.Module):
    def __init__(self, args, input_module):
        super(HardModelBlock, self).__init__()
        self.args = args
        self.input_module = input_module
        self.grad_last = torch.tensor(0, dtype=torch.float, device="cuda")

    def forward(self, x):
        output_of_input_module = self.input_module(x).clone()
        output = self.forward_add_benign_ax(output_of_input_module)
        return output

    def forward_add_benign_ax(self, output_of_input_module):
        output_of_input_module.register_hook(lambda grad: self.set_grad_last(grad))
        if self.grad_last.shape != output_of_input_module.shape:
            output = output_of_input_module
        else:
            stepsize = self.args.adv_feat_step
            p = stepsize * torch.sign(self.grad_last)
            output = output_of_input_module + p
        return output

    def set_grad_last(self, grad):
        self.grad_last=grad.detach().clone()
        return grad


def get_surrogate_models(args):
    attacker_model_list = []
    args = copy.deepcopy(args)
    args.adv_feat = "all_conv"
    interval = int(np.sqrt(args.ens_end_epoch))
    attacker_model_list.append(get_hard_model(args))
    for i in range(args.ens_end_epoch):
        if i == 0 or i == (args.ens_end_epoch - 1) or i % interval == 0:
            path0 = os.path.join(args.path_of_generated_asset, "pt", "pretrained", f"net{i}.pt")
            if os.path.exists(path0):
                attacker_model_list.append(get_hard_model(args, ckpt_path=path0))
                print(f"Added model with parameters stored in {path0} into attacker_model_list")
            path1 = os.path.join(args.path_of_generated_asset, "pt", "randomly_initialized", f"net{i}.pt")
            if os.path.exists(path1):
                attacker_model_list.append(get_hard_model(args, ckpt_path=path1))
                print(f"Added model with parameters stored in {path1} into attacker_model_list")
    return attacker_model_list


def hma(args):
    # args = kwargs["args"]
    attacker_img = read_img(args.attacker_img_path).to(args.device)
    attacker_img = attacker_img[None, ...]
    victim_img = read_img(args.victim_img_path).to(args.device)
    victim_img = victim_img[None, ...]
    # writer = kwargs["writer"]
    attacker_model_list = get_surrogate_models(args)
    loss_func = get_loss_func(args)
    adv_image = attacker_img.detach().clone().requires_grad_(True)
    victim_feat_list = []
    for attacker_model in attacker_model_list:
        victim_feat_list.append(attacker_model(preprocess_batch_img(victim_img)).detach())
    for i in range(args.round):
        loss = 0.0
        for i_attacker_model, attacker_model in enumerate(attacker_model_list):
            std_proj = random.uniform(0.01, 0.1)
            std_rotate = random.uniform(0.01, 0.1)
            adv_feat = attacker_model(
                preprocess_batch_img(image_transform_process(adv_image, std_proj, std_rotate)))
            loss += loss_func(victim_feat_list[i_attacker_model], adv_feat).mean()
            # loss = torch.mean(loss)
        loss = loss / len(attacker_model_list)
        loss.backward()
        grad = adv_image.grad
        adv_image = adv_image - grad.sign()
        adv_image = torch.clamp(adv_image, attacker_img - args.epsilon, attacker_img + args.epsilon)
        adv_image = torch.clamp(adv_image, 0, 255)
        adv_image = adv_image.detach().clone().requires_grad_(True)
    return adv_image

if __name__ == '__main__':
    args = get_args()
    adv_image = hma(args)
    save_img(args.adv_img_path, adv_image[0])
    print(f"Saved the adversarial example in {args.adv_img_path}")