#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "cnrt.h"
#include "../common.h"

extern void AddKernel(float *output, const float *a,
                   const float *b, int32_t data_num);

float src1_cpu[ELEM_NUM];
float src2_cpu[ELEM_NUM];
float dst_cpu[ELEM_NUM];

int main()
{
    CNRT_CHECK(cnrtSetDevice(0));
    cnrtNotifier_t notifier_start, notifier_end;
    CNRT_CHECK(cnrtNotifierCreate(&notifier_start));
    CNRT_CHECK(cnrtNotifierCreate(&notifier_end));
    cnrtQueue_t queue;
    CNRT_CHECK(cnrtQueueCreate(&queue));

    cnrtDim3_t dim;
    cnrtFunctionType_t func_type;
    get_policy_function_block(&dim, &func_type);
    dim.x = 16;

    for (int32_t i = 0; i < ELEM_NUM; ++i)
    {
        src1_cpu[i] = 1.0f;
        src2_cpu[i] = 1.0f;
    }

    float *src1_mlu = NULL;
    float *src2_mlu = NULL;
    float *dst_mlu = NULL;
    int32_t elements_num = ELEM_NUM / dim.x;
    CNRT_CHECK(cnrtMalloc((void **)&src1_mlu, ELEM_NUM * sizeof(float)));
    CNRT_CHECK(cnrtMalloc((void **)&src2_mlu, ELEM_NUM * sizeof(float)));
    CNRT_CHECK(cnrtMalloc((void **)&dst_mlu, ELEM_NUM * sizeof(float)));

    CNRT_CHECK(cnrtMemcpy(src1_mlu, src1_cpu, ELEM_NUM * sizeof(float),
                          cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpy(src2_mlu, src2_cpu, ELEM_NUM * sizeof(float),
                          cnrtMemcpyHostToDev));

    void *args[] = {&dst_mlu, &src1_mlu, &src2_mlu, &elements_num};
    CNRT_CHECK(cnrtPlaceNotifier(notifier_start, queue));
    CNRT_CHECK(cnrtInvokeKernel((void *)&AddKernel, dim, func_type, args, 0, queue));
    CNRT_CHECK(cnrtPlaceNotifier(notifier_end, queue));
    CNRT_CHECK(cnrtQueueSync(queue));

    float latency = 0.0;
    CNRT_CHECK(cnrtNotifierDuration(notifier_start, notifier_end, &latency));

    CNRT_CHECK(cnrtMemcpy(dst_cpu, dst_mlu, ELEM_NUM * sizeof(float),
                          cnrtMemcpyDevToHost));

    CNRT_CHECK(cnrtFree(src1_mlu));
    CNRT_CHECK(cnrtFree(src2_mlu));
    CNRT_CHECK(cnrtFree(dst_mlu));
    CNRT_CHECK(cnrtQueueDestroy(queue));

    float diff = 0.0;
    float baseline = 2.0f; // 1.0f + 1.0f = 2.0f
    for (int32_t i = 0; i < ELEM_NUM; ++i)
    {
        diff += fabs(dst_cpu[i] - baseline);
    }

    double theory_io = ELEM_NUM * sizeof(float) * 3.0;
    double theory_ops = ELEM_NUM;
    double peak_compute_force = get_peak_compute_force();
    double io_bandwidth = get_io_bandwidth();
    double io_efficiency = theory_io / (latency * 1000) / io_bandwidth;
    double compute_efficiency = theory_ops / (latency * 1000) / peak_compute_force;
    printf("[MLU Hardware Time ]: %.3f ms\n", latency / 1000);
    printf("[MLU IO Efficiency ]: %f\n", io_efficiency);
    printf("[MLU Compute Efficiency]: %f\n", compute_efficiency);
    printf("[MLU Diff Rate ]: %f\n", diff);
    printf(diff == 0 ? "PASSED\n" : "FAILED\n");

    return 0;
}
