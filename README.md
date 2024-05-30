# DDDM

This is the codebase for [Directly Denoising Diffusion Model (ICML 2024)](https://arxiv.org/abs/2405.13540).

# Getting Started


## Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `DDDM` python package that the scripts depend on.

## Preparing Data

The training code reads images from a directory of image files. In the [datasets](datasets) folder, we have provided instructions/scripts for preparing these directories for CIFAR-10 and ImageNet.

For creating your own dataset, simply dump all of your images into a directory with ".jpg", ".jpeg", or ".png" extensions. If you wish to train a class-conditional model, name the files like "mylabel1_XXX.jpg", "mylabel2_YYY.jpg", etc., so that the data loader knows that "mylabel1" and "mylabel2" are the labels. Subdirectories will automatically be enumerated as well, so the images can be organized into a recursive structure (although the directory names will be ignored, and the underscore prefixes are used as names).

The images will automatically be scaled and center-cropped by the data-loading pipeline. Simply pass `--data_dir path/to/images` to the training script, and it will take care of the rest.

## Training

To train your model, you should first decide some hyperparameters. We will split up our hyperparameters into three groups: model architecture, diffusion process, and training flags. Here are some reasonable defaults for a baseline:

```
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
```

Here are some changes we experiment with, and how to set them in the flags:

 * **Variance Preserving Diffusion:** add `--VP ` to `DIFFUSION_FLAGS` (Default)
 * **Variance Exploding Diffusion:** set `--VP True` to `--VP False`
 * **Use Pseudo-Huber metric:** add `--use_ph True` to `DIFFUSION_FLAGS`
 * **Use Pseudo-LPIPS metric:** add `--use_pl True` to `DIFFUSION_FLAGS`

Once you have setup your hyper-parameters, you can run an experiment like so:

```
torchrun \
--standalone \
--nproc_per_node=8 \
./scripts/image_train.py --data_dir datasets/cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

```


The logs and saved models will be written to a logging directory determined by the `DDDM_LOGDIR` environment variable. If it is not set, then a temporary directory will be created in `/tmp`.

## Sampling

The above training script saves checkpoints to `.pt` files in the logging directory.ou will likely want to sample from the EMA models, since those produce much better samples.

Once you have a path to your model, you can generate a large batch of samples like so:

```
python scripts/image_sample.py --model_path /path/to/model.pt $MODEL_FLAGS $DIFFUSION_FLAGS
```

Our model generates samples in one step. To enalbe multi-step sampling, change the value of `--sample_steps`.


## **Citation**
If you find the code useful for your work, please star this repo and consider citing:
```
@misc{zhang2024directly,
    title={Directly Denoising Diffusion Model},
    author={Dan Zhang and Jingjing Wang and Feng Luo},
    year={2024},
    eprint={2405.13540},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

```


## **Acknowledgement**
The code is based on[improved-diffusion](https://github.com/openai/improved-diffusion)