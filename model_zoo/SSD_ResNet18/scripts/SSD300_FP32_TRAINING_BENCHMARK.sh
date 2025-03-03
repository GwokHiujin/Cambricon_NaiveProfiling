# This script launches SSD300 training benchmark in FP32 on 1 MLU with 32 batch size
# Usage: bash SSD300_FP32_INFERENCE_BENCHMARK.sh <path to this repository> <path to dataset> <additional flags>

/torch/venv3/pytorch/bin/python $1/main.py --backbone resnet18 --warmup 300 --mode benchmark-training --bs 32 --data $2 ${@:3}
