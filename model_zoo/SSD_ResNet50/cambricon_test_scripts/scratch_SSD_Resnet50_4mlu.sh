#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
SSD_DIR=$(cd ${CUR_DIR}/../;pwd)

pushd $SSD_DIR

export COCO2017_TRAIN_DATASET=$PYTORCH_TRAIN_DATASET/COCO2017
python -m torch.distributed.launch --nproc_per_node=4 main.py --backbone resnet50 --warmup 300 --bs 32 --data ${COCO2017_TRAIN_DATASET} --save ./models --iterations -1 

popd