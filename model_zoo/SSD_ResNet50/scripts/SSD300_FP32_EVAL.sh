# This script evaluates SSD300 model in FP32 using 32 batch size on 1 MLU or GPU
# Usage: ./SSD300_FP32_EVAL.sh <path to this repository> <path to dataset> <path to checkpoint> <device type,0)MLU, 1)GPU> additional flags>

#param
device='MLU'
if [[ $4 -eq 1 ]];then
    device='GPU'
fi

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

/torch/venv3/pytorch/bin/python $1/main.py --max_bitwidth --backbone resnet50 --ebs 32 --data $2 --mode evaluation --checkpoint $3 --device $device ${@:5}

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="SSD_resnet50"
echo "RESULT: $result_name,$result s,$start_fmt"