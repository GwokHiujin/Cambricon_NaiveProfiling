#!/bin/bash
set +e
CUR_DIR=$(cd $(dirname $0);pwd)

function usage
{
    echo "Usage:"
    echo "-------------------------------------------------------------"
    echo "|  $0 [options]"
    echo "|  optional params:"
    echo "|               -i  is_ddp, 0)single card, 1)ddp defaultly be 0."
    echo "|               -d  device, 0)mlu, 1)mlu defaultly be 0."
    echo "|               -c  mixed precision, -1) fp32, 0)O0, 1)O1, 2)O2, 3)O3. 4)amp defaultly be 0."
    echo "|  eg. ./test_traing_benchmarks.sh -i 0 -c 1"
    echo "|      which means running benchmark test on single MLU card with cnmix_O1"
    echo "|  eg. export MLU_VISIBLE_DEVICE=0,1,2,3 && ./test_traing_benchmarks.sh -i 1 -c 0"
    echo "|      which means running benchmark test on 4 MLU cards with cnmix_O0"
    echo "-------------------------------------------------------------"

}

# 获得用户输入功能编号
is_ddp=0
device=0
mixed_index=0
while getopts "i:d:c:" opt; do
case "$opt" in
    i) is_ddp=$OPTARG;;
    d) device=$OPTARG;;
    c) mixed_index=$OPTARG;;
    ?) echo "there is unrecognized optional parameter."; usage; exit 1;;
esac
done

if [[ $is_ddp =~ ^[0-1]{1}$ && $device =~ ^[0-1]{1}$ && $mixed_index =~ ^-?[0-4]{1}$ && $mixed_index -ge -1 && $mixed_index -le 4 ]]; then
    echo "Parameters Exact."
else
    echo "[ERROR] Unknown Parameter."
    usage
    exit 1
fi

# 根据用户输入编号确定config params
config_params=""
case "$mixed_index" in
    -1) config_params="fp32" ;;
    0)  config_params="O0"   ;;
    1)  config_params="O1"   ;;
    2)  config_params="O2"   ;;
    3)  config_params="O3"   ;;
    4)  config_params="amp"  ;;
    ?)  echo "unrecognized precision index " $mixed_index; usage; exit 1 ;;
esac


if [[ $device == 1 ]]; then
    config_params=${config_params}-gpu
else 
    config_params=${config_params}-mlu
fi

if [[ $is_ddp == 1 ]]; then
    config_params=${config_params}-ddp
fi

function travel_networks_benchmark () {
    local runnable_script=$1
    local net_max_id=$2
    local config_params=$3

    for ((net_id=0; net_id<=${net_max_id}; net_id++))
    do
        bash $runnable_script $net_id $config_params
    done

}

# 分类网络遍历执行
max_network_id=18
script="${CUR_DIR}/../Classification/test_classify.sh"
travel_networks_benchmark $script $max_network_id $config_params

# 检测网络遍历执行
max_network_id=8
script="${CUR_DIR}/../Detection/test_detection.sh"
travel_networks_benchmark $script $max_network_id $config_params

# 语言网络遍历执行
max_network_id=5
script="${CUR_DIR}/../LanguageModeling/test_language.sh"
travel_networks_benchmark $script $max_network_id $config_params

# 推荐网络遍历执行
max_network_id=0
script="${CUR_DIR}/../Recommendation/test_recommendation.sh"
travel_networks_benchmark $script $max_network_id $config_params

# 分割网络遍历执行
max_network_id=0
script="${CUR_DIR}/../Segmentation/test_segmentation.sh"
travel_networks_benchmark $script $max_network_id $config_params

# 语音合成网络遍历执行
max_network_id=1
script="${CUR_DIR}/../SpeechSynthesis/test_speech.sh"
travel_networks_benchmark $script $max_network_id $config_params

# 生成对坑网络遍历执行
max_network_id=0
script="${CUR_DIR}/../GAN/test_gan.sh"
travel_networks_benchmark $script $max_network_id $config_params


