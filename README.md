# Exploring Latent Dimensions of Crowd-sourced Creativity

(Accepted at Machine Learning for Creativity and Design (NeurIPS Workshop), 2021) 

*Our code is based on the official [GANalyze repository](https://github.com/LoreGoetschalckx/GANalyze)*

## Overview
- [Requirements and Installation](#requirements-and-installation)
- [Training](#training)
- [Testing](#testing)
- [Reference](#reference)

## Requirements and Installation

We have tested our framework and produced the official results with the libraries and their corresponding versions below. However, newer versions will very likely work without a problem. If you experience a problem, don't hesitate to open an issue.

- CUDA 10.2
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.6.0 and [torchvision](https://github.com/pytorch/vision) >= 0.7.0
- efficientnet-pytorch
- numpy, scipy, PIL

One possible way of installing the libraries on a linux server is as follows:

```bash
# CUDA 10.2
pip install torch==1.6.0 torchvision==0.7.0

# Efficientnet Pytorch
pip install efficientnet-pytorch==0.7.0
```

Finally, to clone this repo, run:

```bash
git clone https://github.com/catlab-team/latentcreative.git
```

## Training

Before training, pretrained BigGAN models should be downloaded using the script below:

```bash
cd GANalyze
sh download_pretrained.sh
```
After download, you can use the training script below to train the proposed framework:


```bash
python train_pytorch.py --assessor_path ../CreativeClassifier/models/best.pth \
--experiment_name experiment --artbreeder_class 0 --class_direction 1 --num_samples 400000 \
--clipped_step_size 1 --transformer AdaptiveDirectionZAdaptiveDirectionY_noise_nonlinear class_reg=1000,noise_dim=8 \
--batch_size 8 --learning_rate 0.0001 --multiway_linear 0
```

## Testing

After training, you can use the testing script below to produce images.

```bash
python test_artbreed.py --checkpoint 400000 --checkpoint_dir Checkpoints/experiment/biggan512_None/creative_classifier_best.pth/AdaptiveDirectionZAdaptiveDirectionY_noise_nonlinear_class_reg=1000,noise_dim=8/artbreeder_class:0/[HASH]
```



## Reference

BibTeX reference of our work is as follows:

```markdown
@article{kocasari2021exploring,
  title={Exploring Latent Dimensions of Crowd-sourced Creativity},
  author={Kocasari, Umut and Bag, Alperen and Atici, Efehan and Yanardag, Pinar},
  journal={arXiv preprint arXiv:2112.06978},
  year={2021}
}
```

