#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

base_params () {
    device="MLU"

    batch_size="32"
    iters="2"
    eval_iters="2"
    precision="fp32"
    ddp="False"
    evaluate="False";
    seed=6503
    num_workers=12
    DEVICE_COUNT="1"

    benchmark_mode="True"
    max_batch_size_MLU290="32"
    max_batch_size_MLU370="80"
    max_batch_size_MLU590_M9="320"
    max_batch_size_MLU590_M9U="320"
    max_batch_size_MLU590_H8="280"
    max_batch_size_MLU590_M9B="280"
    max_batch_size_MLU590_H8_AMP="560"
    max_batch_size_MLU590_M9B_AMP="560"
    max_batch_size_MLU590_M9_AMP="660"
    max_batch_size_MLU590_M9U_AMP="660"
    max_batch_size_MLU580="152"
    max_batch_size_MLU580_AMP="296"
    max_batch_size_MLU590_M9C="112"
    max_batch_size_MLU590_M9C_AMP="222"
    max_batch_size_MLU570="112"
    max_batch_size_MLU570_AMP="222"
    max_batch_size_MLU370_AMP="128"
    max_batch_size_MLU370_ECC="64"
    max_batch_size_V100="32"

}

set_configs () {
    params=$1

    # 调用网络的base_params
    base_params

    # 根据每个字段的功能, overide对应参数
    params_array=(${params//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            amp)    precision="amp" ;;
            mlu)    ;;
            gpu)    device="gpu" ;;
            ddp)    ddp="True" ;;
            ci)     benchmark_mode=False;
                    evaluate="True" ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    ## 加载公用方法
    source ${CONFIG_DIR}/../../../../tools/utils/common_utils.sh
    get_visible_cards DEVICE_COUNT
    # 处理benchmark_mode所需的参数
    if [[ $benchmark_mode == "True" ]]; then

        ## 获取benchmark_mode计数规则,配置迭代数
        iters=-1
        perf_iters_rule iters
        ## 设置benchmark_mode log路径
        export BENCHMARK_LOG=${CONFIG_DIR}/../../../../benchmark_log

        ## 获取平台类型，配置最大batch_size
        cur_platform=""
        get_platform_with_flag_name cur_platform
        mbs_name=max_batch_size_${cur_platform}

        cur_ecc_status=""
        get_ecc_status cur_ecc_status
        if [[ ${cur_ecc_status} == "ON" ]]; then
            mbs_name=max_batch_size_${cur_platform}_ECC
        fi

        if [[ $precision == "amp" ]]; then
            mbs_name=max_batch_size_${cur_platform}_AMP
	    if [[ ${cur_platform%_*} == "MLU590" && ${ddp} == "True" ]]; then
                if [[ ${DEVICE_COUNT} -le 8 ]]; then
	            total_iters=18
		    cutdown_iters=10
                    iters=$total_iters
                    export MLU_ADAPTIVE_STRATEGY_COUNT=$cutdown_iters
		fi
            fi
        fi

        batch_size=${!mbs_name}
        if [ ! -z ${USER_MAX_BATCH_SIZE} ]; then
            batch_size=${USER_MAX_BATCH_SIZE}
        fi

        ## 检查性能模式软硬件环境
        pushd ${CONFIG_DIR}/../../../../tools/mlu_performance_check/; ./check_mlu_perf.sh; popd;
    fi
}

