from utils.metric import cal_dis


def targeted_loss(victim_feat, adv_feat):
    loss = cal_dis(victim_feat, adv_feat, "norm_L2_square")
    return loss


def untargeted_loss(attacker_like_feat, adv_feat):
    loss = -cal_dis(attacker_like_feat, adv_feat, "norm_L2_square")
    return loss


def get_loss_func(args):
    if args.attack_type == "targeted":
        return targeted_loss
    elif args.attack_type == "untargeted":
        return untargeted_loss
    else:
        raise NotImplementedError("--attack_type == {} is not supported".format(args.attack_type))
