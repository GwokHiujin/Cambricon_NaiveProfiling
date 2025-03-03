#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

BACKBONE="resnet50"
TARGET_MAP="0.2502"
START_EVAL_AT=32
EVALUATE_EVERY=2
WARMUP=300
ITERS="-1"
BATCH_SIZE=32
SAVEDIR="./models_test_4mlu"

CUR_DIR=$(cd $(dirname $0);pwd)
SSD_DIR=$(cd ${CUR_DIR}/../;pwd)
pushd $SSD_DIR

if [ ! -d "$SAVEDIR" ]; then  
    mkdir -p "$SAVEDIR"  
    echo "Create SAVEDIR: $SAVEDIR"  
else  
    echo "SAVEDIR $SAVEDIR is existed! "  
fi  

export MLU_VISIBLE_DEVICES=0,1,2,3
if [ -d ${PYTORCH_TRAIN_DATASET} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"
    
    # export COCO2017_TRAIN_DATASET=$PYTORCH_TRAIN_DATASET/COCO2017
    export COCO2017_TRAIN_DATASET="/coco"
    
    /torch/venv3/pytorch/bin/python -m torch.distributed.launch --nproc_per_node=4 main.py\
        --max_bitwidth
        --backbone $BACKBONE \
    	--warmup $WARMUP \
        --bs $BATCH_SIZE \
    	--ebs $BATCH_SIZE \
    	--data ${COCO2017_TRAIN_DATASET} \
    	--save $SAVEDIR \
    	--iterations $ITERS \
    	--start_eval_at $START_EVAL_AT \
        --evaluate_every $EVALUATE_EVERY \
        --target_map $TARGET_MAP
    
    # end timing
    end=$(date +%s)
    end_fmt=$(date +%Y-%m-%d\ %r)
    echo "ENDING TIMING RUN AT $end_fmt"
    
    
    # report result
    result=$(( $end - $start ))
    result_name="SSD_resnet50"
    
    echo "RESULT,$result_name,$result,$USER,$start_fmt"
else
    echo "Directory ${PYTORCH_TRAIN_DATASET} does not exist"
fi
popd
