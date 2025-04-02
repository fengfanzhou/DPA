import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.io import read_csv, write_csv, read_img


class Balancedface(Dataset):
    def __init__(self, args):
        self.args = args
        self.id_dir_name_mapping_path = os.path.join(args.path_of_generated_asset, "balancedface_id_dir_name_mapping.csv")
        self.img_list = read_csv(self.id_dir_name_mapping_path)[1:]
        self.img_label_all_dataset = np.unique(np.array(self.img_list)[:, 0].astype(int))
        self.class_nums = self.img_label_all_dataset.shape[0]

    def __getitem__(self, item):
        img_path = os.path.join(self.args.dataset_path, self.img_list[item][1], self.img_list[item][2], self.img_list[item][3])
        img_path = f"{os.path.splitext(img_path)[0]}.jpg"
        if not os.path.exists(img_path):
            img_path = f"{os.path.splitext(img_path)[0]}.png"
        img = read_img(img_path)
        label = int(self.img_list[item][0])
        return img, label

    def __len__(self):
        return len(self.img_list)


def get_balancedface_dataloader(args):
    dataset = Balancedface(args)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.fr_train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    return dataloader