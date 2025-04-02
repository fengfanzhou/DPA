import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fr_model_ckpt_path', type=str, default='./generated_asset/mobile_face.pth')
    parser.add_argument('--attacker_img_path', type=str)
    parser.add_argument('--victim_img_path', type=str)
    parser.add_argument('--adv_img_path', type=str, default='./generated_asset/adv_img.png')
    parser.add_argument('--path_of_generated_asset', type=str, default='./generated_asset')
    parser.add_argument('--round', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--fr_train_batch_size', type=int, default=512)
    parser.add_argument('--fr_train_feature_dim', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=40)
    parser.add_argument('--fr_train_scale_size', type=float, default=32.0)
    parser.add_argument('--epsilon', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ens_end_epoch', type=int, default=35)
    parser.add_argument('--dataset_path', type=str, default="")
    parser.add_argument('--attack_type', type=str, default="targeted")
    parser.add_argument('--adv_feat_step', type=float, default=8e-4)
    args = parser.parse_args()
    return args
