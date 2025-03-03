# This script launches SSD300 training in FP32 on 1 MLUs or GPUs using 32 batch size
# Usage: ./SSD300_FP32_1MLU.sh <path to this repository> <path to dataset> <device type,0)MLU, 1)GPU> <additional flags>

#param
CUR_DIR=$(cd $(dirname $0);pwd)
device='MLU'
if [[ $3 -eq 1 ]];then
    device='GPU'
fi

pip install -r ${CUR_DIR}/../requirements.txt

device_type=`cat /proc/driver/*/*/*/information | grep "Device name" | uniq | awk -F ":" '{print $2}'`
if [[ $BENCHMARK_LOG && "$device_type" == *"MLU370"* ]]; then
    bs=80
else
    bs=32
fi

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

/torch/venv3/pytorch/bin/python $1/main.py --backbone resnet18 --bs $bs --warmup 32 --data $2 --device $device ${@:4}

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="SSD_resnet"
echo "RESULT: $result_name,$result s,$start_fmt"
