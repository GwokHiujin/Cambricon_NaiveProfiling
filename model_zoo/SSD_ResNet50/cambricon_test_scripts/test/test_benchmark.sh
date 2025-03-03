#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
SSD_DIR=$(cd ${CUR_DIR}/../../;pwd)

# 帮助函数
function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  bash $0 precision-device-[options...]"
    echo "|      Supported options:"
    echo "|             precision: fp32, amp"
    echo "|             device: mlu, gpu"
    echo "|             option1(multicards): ddp"
    echo "|                                                   "
    echo "|  eg.1. bash test_benchmark.sh fp32-mlu"
    echo "|      which means running on single MLU card with fp32 precision."
    echo "|                                                   "
    echo "|  eg.2. export MLU_VISIBLE_DEVICES=0,1,2,3 && bash test_benchmark.sh O1-mlu-ddp"
    echo "|      which means running on 4 MLU cards with O1 precision."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

# 获取用户指定config函数并执行,得到对应config的参数配置
while getopts 'h:' opt; do
   case "$opt" in
       h)  usage ; exit 1 ;;
       ?)  echo "unrecognized optional arg : "; $opt; usage; exit 1;;
   esac
done
## 加载参数配置
config=$1
source ${CUR_DIR}/params_config.sh
set_configs "$config"

args_cmd=" --data $COCO2017_TRAIN_DATASET --bs $batch_size \
    --checkpoint $PYTORCH_TRAIN_CHECKPOINT/ssd/epoch_31.pt \
    --seed $seed --epochs 33 --iterations $iters \
    --backbone resnet50 \
    --device $device \
    --num-workers $num_workers \
    --save ./models --backbone-path $PYTORCH_TRAIN_CHECKPOINT/ssd/resnet50-19c8e357.pth "
val_cmd="bash examples/SSD300_FP32_EVAL.sh ./ $COCO2017_TRAIN_DATASET ./models/epoch_32.pt $device --eval_iters $eval_iters --backbone-path $PYTORCH_TRAIN_CHECKPOINT/ssd/resnet50-19c8e357.pth "



# config配置到网络脚本的转换
main() {

    pushd $SSD_DIR
    pip install -r requirements.txt

    # 配置DDP相关参数
    if [[ $ddp == "True"  ]]; then
      train_cmd="python -m torch.distributed.launch --nproc_per_node=$DEVICE_COUNT main.py --warmup 300  $args_cmd"
    else
      train_cmd="python main.py --warmup 32 $args_cmd"
    fi

    # 配置混合精度相关参数
    if [[ ${precision} == "amp" ]]; then
      train_cmd="${train_cmd} --pyamp"
    fi

    if [[ ${cur_platform%_*} != "MLU370" ]]; then
      train_cmd="${train_cmd} --data-backend dali-mlu"
    fi

    # 参数配置完毕，运行脚本
    echo "$train_cmd"
    eval "${train_cmd}"

    # 是否执行推理部分
    if [[ ${evaluate} == "Ture" ]]; then
         echo "val_cmd: $val_cmd"
		 eval "$val_cmd"
    fi

    popd
}



pushd $CUR_DIR
main
popd
