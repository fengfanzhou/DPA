# Improving the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation

This is the PyTorch implementation of our paper:

[Paper] Improving the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation

Fengfan Zhou, Bangjie Yin, Hefei Ling, Qianyu Zhou, Wenxuan Wang.

The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR), 2025


**[[Arxiv]](https://arxiv.org/pdf/2411.15555)**


## Installation

### Requirements

* Ubuntu, CUDA>=11.7

* Python>=3.8.12
    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n dpa python=3.8.12
    ```
    Then, activate the environment:
    ```bash
    conda activate dpa
    ```
* PyTorch>=2.0.1, torchvision>=0.15.2 (following instructions [here](https://pytorch.org/)

    For example, if your CUDA version is 11.7, you could install pytorch and torchvision as following:
    ```bash
    pip install torch==2.0.1 torchvision==0.15.2
    ```
  
* Other requirements
```requirements
opencv-python~=4.6.0.66
rich~=13.3.1
```

## Diverse Parameters Optimization (DPO)

You can use the following command:
```bash
CUDA_VISIBLE_DEVICES=[gpu id] python dpo.py --dataset_path=[path of the dataset]
```

## Hard Model Aggregation (HMA)
Run the following command to craft the adversarial examples:
```bash
CUDA_VISIBLE_DEVICES=2 python hma.py --attacker_img_path=[your attacker image path] --victim_img_path=[your victim image path]
```

## Acknowledgements
This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [Adv-Makeup](https://github.com/TencentYoutuResearch/Adv-Makeup)
* [AMT-GAN](https://github.com/CGCL-codes/AMT-GAN)
* [TIP-IM](https://github.com/ShawnXYang/TIP-IM)
* [Face_Pytorch](https://github.com/wujiyang/Face_Pytorch)
* [IADG](https://github.com/qianyuzqy/IADG)
* [FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo)

## Citing DPA
If you find DPA useful in your research, please consider citing:
```bibtex
@inproceedings{zhou2025improving,
  title={Improving the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation},
  author={Zhou, Fengfan and Yin, Bangjie and Ling, Hefei and Zhou, Qianyu and Wang, Wenxuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

