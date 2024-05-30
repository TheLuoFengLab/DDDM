#!/bin/bash

MASTER=$1

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"



cd /project/dzhang4/vast_data/improved-diffusion/
conda activate ldm
OPENAI_LOGDIR="./logs"

torchrun \
--nproc_per_node=2 \
--nnodes=8 \
--node_rank= $PBS_NODENUM \
--rdzv_id=123 \
--rdzv_backend=c10d \
--rdzv_endpoint= $MASTER \
./scripts/image_train.py --data_dir datasets/cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS


