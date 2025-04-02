import os

import numpy as np
import torch
import torch.optim as optim
from config import get_args
from rich.progress import track
from utils.dataset import get_balancedface_dataloader

from utils.fr_model.fr_interface import get_fr_model, get_margin
from utils.image_process import preprocess_batch_img

from utils.io import save_pt


def dpa_process(args, pretrained=True):
    dataloader = get_balancedface_dataloader(args)
    net = get_fr_model(args, pretrained=pretrained)
    margin = get_margin(args, dataloader.dataset.class_nums)
    net = net.cuda()
    net.train()
    margin = margin.cuda()
    margin.train()
    tag = "pretrained" if pretrained else "randomly_initialized"
    pt_save_path = os.path.join(args.path_of_generated_asset, "pt", tag)
    os.makedirs(pt_save_path, exist_ok=True)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer_ft = optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}
    ], lr=args.lr, momentum=0.9, nesterov=True)
    start_epoch = 0
    total_iters = 1
    interval = int(np.sqrt(args.ens_end_epoch))
    for epoch in range(start_epoch, start_epoch + args.ens_end_epoch):
        print("Start epoch {}".format(epoch))
        for img, label in track(total=len(dataloader), sequence=dataloader,
                                description=""):
            img = img.cuda()
            label = label.cuda()
            optimizer_ft.zero_grad()
            raw_logits = net(preprocess_batch_img(img))
            output = margin(raw_logits, label)
            total_loss = criterion(output, label)
            total_loss.backward()
            optimizer_ft.step()

            total_iters += 1
        if epoch == 0 or epoch == (args.ens_end_epoch - 1) or epoch % interval == 0:
            save_pt(net.state_dict(), os.path.join(pt_save_path, "net{}.pt".format(epoch)))


def dpa(args):
    dpa_process(args, pretrained=True)
    dpa_process(args, pretrained=False)


if __name__ == '__main__':
    args = get_args()
    dpa(args)